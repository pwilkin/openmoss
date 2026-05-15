// SPDX-License-Identifier: Apache-2.0
//
// Audio codec (MOSS-Audio-Tokenizer) GGML graph: codes → waveform.
//
// Architecture (decoder side; encoder mirrors it but is not implemented yet):
//
//   codes (32, T)
//     │
//     ▼  per-quantizer LFQ.decode_code:
//        codebook lookup → out_proj  (Conv1d 8 → 512, kernel=1, weight-normed)
//     │
//     ▼  Σ over 32 quantizers
//   emb (512, T)  ─→  quantizer.output_proj (Conv1d 512 → 768, weight-normed)
//     │
//     ▼  emb (768, T)
//   dec.0:  ProjectedTransformer  in=768  d=1280 nh=20 dff=5120  nl=32  out=1280
//   dec.1:  patch=2 upsample      → (640, 2·T)
//   dec.2:  ProjectedTransformer  in=640  d=768  nh=12 dff=3072  nl=12  out=768
//   dec.3:  patch=2 upsample      → (384, 4·T)
//   dec.4:  ProjectedTransformer  in=384  d=768  nh=12 dff=3072  nl=12  out=768
//   dec.5:  patch=2 upsample      → (384, 8·T)
//   dec.6:  ProjectedTransformer  in=384  d=768  nh=12 dff=3072  nl=12  out=240
//   dec.7:  patch=240 upsample    → (1, 8·240·T) = waveform
//
// All transformer layers use pre-LN, fused QKV via in_projs.0.weight, RoPE
// (interleaved-pair, max_period=10000), causal self-attention, GELU FFN, and
// per-channel LayerScale on the residual branches.
//
// Weight-norm reconstruction: the upstream model decorates the quantizer
// projections with `torch.nn.utils.parametrizations.weight_norm`, so the
// safetensors carry `wp0` (magnitude, shape (out, 1, 1)) and `wp1` (direction,
// shape (out, in, 1)) instead of a plain weight. We materialise the effective
// weight `w[o,i] = wp0[o] * wp1[o,i] / sqrt(Σi wp1[o,i]^2)` on the host once at
// codec init and upload it as a fresh f16 tensor on the aux backend.

#include "openmoss/codec.h"
#include "openmoss/model.h"
#include "openmoss/tokenizer.h"

#include "aux_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace openmoss {

namespace {

constexpr float NORM_EPS         = 1e-5f;
constexpr float ROPE_FREQ_BASE   = 10000.0f;
constexpr int   CODEC_NUM_VQ     = 32;
constexpr int   CODEC_CB_DIM     = 8;
constexpr int   CODEC_RVQ_DIM    = 512;
constexpr int   CODEC_OUT_DIM    = 768;

struct StageSpec {
    int input_dim;
    int d_model;
    int n_heads;
    int dim_ff;
    int n_layers;
    int output_dim;
    int patch_after;     // upsample factor of the PatchedPretransform that follows; 0 if none
    int gguf_idx;        // dec.<this>
};

constexpr std::array<StageSpec, 4> DECODER_STAGES = {{
    //   in   d   nh   dff  nl  out  patch  gguf
    {  768, 1280, 20, 5120, 32, 1280,    2,    0 },  // dec.0 + dec.1 patch
    {  640,  768, 12, 3072, 12,  768,    2,    2 },  // dec.2 + dec.3 patch
    {  384,  768, 12, 3072, 12,  768,    2,    4 },  // dec.4 + dec.5 patch
    {  384,  768, 12, 3072, 12,  240,  240,    6 },  // dec.6 + dec.7 patch
}};

// Encoder mirrors the decoder. The initial patch=240 happens BEFORE the first
// transformer stage; `patch_after` here is the patch that follows the stage.
constexpr int CODEC_PRE_PATCH = 240;
constexpr std::array<StageSpec, 4> ENCODER_STAGES = {{
    //   in   d   nh   dff  nl  out  patch  gguf
    {  240,  768, 12, 3072, 12,  384,    2,    1 },  // enc.1 + enc.2 patch
    {  768,  768, 12, 3072, 12,  384,    2,    3 },  // enc.3 + enc.4 patch
    {  768,  768, 12, 3072, 12,  640,    2,    5 },  // enc.5 + enc.6 patch
    { 1280, 1280, 20, 5120, 32,  768,    0,    7 },  // enc.7 (no patch after)
}};

// f16 representation helpers (we store everything as f16 on the backend).
inline uint16_t f32_to_f16_bits(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 0x1;
    int32_t  exp  = int32_t((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffff;
    uint16_t out;
    if (exp <= 0) {
        // subnormal or zero
        if (exp < -10) { out = uint16_t(sign << 15); }
        else {
            mant = (mant | 0x800000) >> uint32_t(1 - exp);
            // round to nearest even
            if (mant & 0x1000) mant += 0x2000;
            out = uint16_t((sign << 15) | (mant >> 13));
        }
    } else if (exp >= 0x1f) {
        // overflow → inf (or NaN if mantissa)
        out = uint16_t((sign << 15) | (0x1f << 10) | (mant ? 0x200 : 0));
    } else {
        if (mant & 0x1000) {
            mant += 0x2000;
            if (mant & 0x800000) { mant = 0; exp += 1; }
            if (exp >= 0x1f) {
                out = uint16_t((sign << 15) | (0x1f << 10));
                return out;
            }
        }
        out = uint16_t((sign << 15) | (uint32_t(exp) << 10) | (mant >> 13));
    }
    return out;
}

inline float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h >> 15) & 0x1;
    int32_t  exp  = int32_t((h >> 10) & 0x1f);
    uint32_t mant = uint32_t(h & 0x3ff);
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign << 31;
        } else {
            // subnormal
            int32_t e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
            u = (sign << 31) | (uint32_t(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        u = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        u = (sign << 31) | (uint32_t(exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &u, 4); return f;
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// CodecGraphs
// ───────────────────────────────────────────────────────────────────────────

class CodecGraphs {
public:
    explicit CodecGraphs(Model & owner);
    ~CodecGraphs();

    std::vector<float>   decode(const int32_t * codes, int32_t n_vq, int32_t T_audio);
    std::vector<int32_t> encode(const float * waveform, int64_t n_samples, int32_t & T_audio_out);

private:
    Model & m_owner;

    // Effective weights (post-weight-norm) for the quantizer projections.
    // Held in their own ggml_context + backend buffer.
    ggml_context        * m_w_ctx = nullptr;
    ggml_backend_buffer_t m_w_buf = nullptr;

    // Per-quantizer oproj: takes (8, T) → (512, T). Stored as (in=8, out=512) f16.
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_q_oproj_w {};
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_q_oproj_b {};
    // Final quantizer.oproj: takes (512, T) → (768, T). (in=512, out=768) f16.
    ggml_tensor * m_quant_oproj_w = nullptr;
    ggml_tensor * m_quant_oproj_b = nullptr;

    // Encoder-only weights:
    //   q.{i}.iproj: (in=512, out=8) f16, plus bias (8,)
    //   quantizer.iproj: (in=768, out=512) f16, plus bias (512,)
    //   codebook_normed[i]: (8, 1024) f16 — L2-normalized rows of codebook[i]
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_q_iproj_w {};
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_q_iproj_b {};
    ggml_tensor * m_quant_iproj_w = nullptr;
    ggml_tensor * m_quant_iproj_b = nullptr;
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_codebook_normed {};

    // Codebook tensors (already loaded in aux); cached for fast lookup.
    std::array<ggml_tensor *, CODEC_NUM_VQ> m_codebook {};

    struct Layer {
        ggml_tensor *norm1_w, *norm1_b;
        ggml_tensor *norm2_w, *norm2_b;
        ggml_tensor *attn_in;       // (d_model, 3*d_model)  fused QKV
        ggml_tensor *attn_out;      // (d_model, d_model)
        ggml_tensor *linear1;       // (d_model, dim_ff)
        ggml_tensor *linear2;       // (dim_ff, d_model)
        ggml_tensor *layer_scale_1; // (d_model,)
        ggml_tensor *layer_scale_2; // (d_model,)
    };
    struct Stage {
        StageSpec            spec;
        ggml_tensor        * iproj = nullptr;   // optional (null when d_model == input_dim)
        ggml_tensor        * oproj = nullptr;   // optional (null when d_model == output_dim)
        std::vector<Layer>   layers;
    };
    std::array<Stage, 4> m_stages;     // decoder stages (dec.0/2/4/6)
    std::array<Stage, 4> m_enc_stages; // encoder stages (enc.1/3/5/7)

    ggml_gallocr_t m_galloc = nullptr;

    // ── private helpers ──
    ggml_tensor * tensor_(const std::string & name) const;
    ggml_tensor * tensor_or_null_(const std::string & name) const;
    void resolve_decoder_();
    void resolve_encoder_();
    void compute_effective_weights_();
    void compute_normalized_codebooks_();
    ggml_tensor * read_f16_to_host_(ggml_tensor * t, std::vector<uint16_t> & out) const;

    // Graph construction:
    ggml_tensor * build_layer_norm_(ggml_context * gctx, ggml_tensor * x,
                                    ggml_tensor * w, ggml_tensor * b) const;
    ggml_tensor * build_attention_(ggml_context * gctx, ggml_tensor * x,
                                    const Layer & L, int d_model, int n_heads,
                                    ggml_tensor * pos) const;
    ggml_tensor * build_ffn_(ggml_context * gctx, ggml_tensor * x,
                              const Layer & L) const;
    ggml_tensor * build_layer_(ggml_context * gctx, ggml_tensor * x,
                                const Layer & L, int d_model, int n_heads,
                                ggml_tensor * pos) const;
    ggml_tensor * build_stage_(ggml_context * gctx, ggml_tensor * x,
                                const Stage & S, ggml_tensor * pos) const;
    static ggml_tensor * patch_upsample_(ggml_context * gctx,
                                          ggml_tensor * x, int patch);
    static ggml_tensor * patch_downsample_(ggml_context * gctx,
                                            ggml_tensor * x, int patch);
};

// ───────────────────────────────────────────────────────────────────────────
// Construction
// ───────────────────────────────────────────────────────────────────────────

CodecGraphs::CodecGraphs(Model & owner) : m_owner(owner) {
    if (!m_owner.codec_present()) {
        // Allow a no-codec build: leave everything null. decode() will throw.
        return;
    }
    auto * aux = m_owner.aux();
    if (!aux || !aux->backend) {
        throw std::runtime_error("CodecGraphs: aux backend not initialised");
    }

    resolve_decoder_();
    resolve_encoder_();
    compute_effective_weights_();
    compute_normalized_codebooks_();

    m_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
    if (!m_galloc) throw std::runtime_error("CodecGraphs: gallocr_new failed");
}

CodecGraphs::~CodecGraphs() {
    if (m_galloc) ggml_gallocr_free(m_galloc);
    if (m_w_buf)  ggml_backend_buffer_free(m_w_buf);
    if (m_w_ctx)  ggml_free(m_w_ctx);
}

ggml_tensor * CodecGraphs::tensor_(const std::string & name) const {
    auto * aux = m_owner.aux();
    auto it = aux->tensors.find(name);
    if (it == aux->tensors.end()) {
        throw std::runtime_error("CodecGraphs: missing codec tensor: " + name);
    }
    return it->second;
}

ggml_tensor * CodecGraphs::tensor_or_null_(const std::string & name) const {
    auto * aux = m_owner.aux();
    auto it = aux->tensors.find(name);
    return it == aux->tensors.end() ? nullptr : it->second;
}

// Resolve every decoder weight into our cached tensor pointers.
void CodecGraphs::resolve_decoder_() {
    // codebooks
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        m_codebook[i] = tensor_("moss.codec.quantizer.q." + std::to_string(i)
                                + ".codebook.weight");
    }

    // decoder stages
    for (size_t s = 0; s < m_stages.size(); ++s) {
        Stage & S = m_stages[s];
        S.spec = DECODER_STAGES[s];
        const std::string base = "moss.codec.dec." + std::to_string(S.spec.gguf_idx) + ".";

        // Optional input/output projections (Identity in PyTorch when dim matches).
        S.iproj = (S.spec.d_model != S.spec.input_dim)
            ? tensor_(base + "iproj.weight") : nullptr;
        S.oproj = (S.spec.d_model != S.spec.output_dim)
            ? tensor_(base + "oproj.weight") : nullptr;

        S.layers.resize(size_t(S.spec.n_layers));
        for (int li = 0; li < S.spec.n_layers; ++li) {
            const std::string lb = base + "tr.l." + std::to_string(li) + ".";
            Layer & L = S.layers[size_t(li)];
            L.norm1_w       = tensor_(lb + "norm1.weight");
            L.norm1_b       = tensor_(lb + "norm1.bias");
            L.norm2_w       = tensor_(lb + "norm2.weight");
            L.norm2_b       = tensor_(lb + "norm2.bias");
            L.attn_in       = tensor_(lb + "attn.inp.0.weight");
            L.attn_out      = tensor_(lb + "attn.outp.0.weight");
            L.linear1       = tensor_(lb + "linear1.weight");
            L.linear2       = tensor_(lb + "linear2.weight");
            L.layer_scale_1 = tensor_(lb + "layer_scale_1.scale");
            L.layer_scale_2 = tensor_(lb + "layer_scale_2.scale");
        }
    }
}

// Resolve encoder weights (mirror of resolve_decoder_).
void CodecGraphs::resolve_encoder_() {
    for (size_t s = 0; s < m_enc_stages.size(); ++s) {
        Stage & S = m_enc_stages[s];
        S.spec = ENCODER_STAGES[s];
        const std::string base = "moss.codec.enc." + std::to_string(S.spec.gguf_idx) + ".";

        S.iproj = (S.spec.d_model != S.spec.input_dim)
            ? tensor_(base + "iproj.weight") : nullptr;
        S.oproj = (S.spec.d_model != S.spec.output_dim)
            ? tensor_(base + "oproj.weight") : nullptr;

        S.layers.resize(size_t(S.spec.n_layers));
        for (int li = 0; li < S.spec.n_layers; ++li) {
            const std::string lb = base + "tr.l." + std::to_string(li) + ".";
            Layer & L = S.layers[size_t(li)];
            L.norm1_w       = tensor_(lb + "norm1.weight");
            L.norm1_b       = tensor_(lb + "norm1.bias");
            L.norm2_w       = tensor_(lb + "norm2.weight");
            L.norm2_b       = tensor_(lb + "norm2.bias");
            L.attn_in       = tensor_(lb + "attn.inp.0.weight");
            L.attn_out      = tensor_(lb + "attn.outp.0.weight");
            L.linear1       = tensor_(lb + "linear1.weight");
            L.linear2       = tensor_(lb + "linear2.weight");
            L.layer_scale_1 = tensor_(lb + "layer_scale_1.scale");
            L.layer_scale_2 = tensor_(lb + "layer_scale_2.scale");
        }
    }
}

// Read a backend tensor (f16) into a host f16 buffer. Returns the tensor
// (unchanged) for convenience.
ggml_tensor * CodecGraphs::read_f16_to_host_(ggml_tensor * t,
                                              std::vector<uint16_t> & out) const {
    if (t->type != GGML_TYPE_F16) {
        throw std::runtime_error(std::string("read_f16_to_host_: expected f16 for ")
                                 + (t->name[0] ? t->name : "<unnamed>"));
    }
    const size_t n = ggml_nelements(t);
    out.resize(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(uint16_t));
    return t;
}

// Materialise effective post-weight-norm Conv1d weights for every quantizer
// projection used during decoding (per-quantizer oproj + the global oproj).
//
// All source tensors are 3D (out, in, k=1) f16 in GGML ne order (1, in, out).
// We compute the effective 2D weight (in, out) on the host and upload.
void CodecGraphs::compute_effective_weights_() {
    auto * aux = m_owner.aux();

    // Build a fresh ggml_context big enough for all the descriptors (33 weights
    // + 33 biases ≈ 66 tensors). Allocate as f16 on the same backend.
    {
        ggml_init_params ip{};
        ip.mem_size   = ggml_tensor_overhead() * 200;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        m_w_ctx = ggml_init(ip);
        if (!m_w_ctx) throw std::runtime_error("CodecGraphs: ggml_init for weight ctx failed");
    }

    auto make_w = [&](const std::string & name, int in_dim, int out_dim) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_2d(m_w_ctx, GGML_TYPE_F16, in_dim, out_dim);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto make_b = [&](const std::string & name, int out_dim) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_1d(m_w_ctx, GGML_TYPE_F16, out_dim);
        ggml_set_name(t, name.c_str());
        return t;
    };

    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        const std::string n = "q.oproj." + std::to_string(i);
        m_q_oproj_w[i] = make_w(n + ".w", CODEC_CB_DIM, CODEC_RVQ_DIM);
        m_q_oproj_b[i] = make_b(n + ".b", CODEC_RVQ_DIM);
    }
    m_quant_oproj_w = make_w("quant.oproj.w", CODEC_RVQ_DIM, CODEC_OUT_DIM);
    m_quant_oproj_b = make_b("quant.oproj.b", CODEC_OUT_DIM);

    // Encoder iprojs (q.{i}.iproj: 512 → 8;  quantizer.iproj: 768 → 512)
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        const std::string n = "q.iproj." + std::to_string(i);
        m_q_iproj_w[i] = make_w(n + ".w", CODEC_RVQ_DIM, CODEC_CB_DIM);
        m_q_iproj_b[i] = make_b(n + ".b", CODEC_CB_DIM);
    }
    m_quant_iproj_w = make_w("quant.iproj.w", CODEC_OUT_DIM, CODEC_RVQ_DIM);
    m_quant_iproj_b = make_b("quant.iproj.b", CODEC_RVQ_DIM);

    // Pre-allocate normalized codebooks (filled in compute_normalized_codebooks_)
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(m_w_ctx, GGML_TYPE_F16,
                                              CODEC_CB_DIM, /*codebook_size=*/1024);
        ggml_set_name(t, ("cb_norm." + std::to_string(i)).c_str());
        m_codebook_normed[i] = t;
    }

    // Allocate backing storage on the backend.
    m_w_buf = ggml_backend_alloc_ctx_tensors(m_w_ctx, aux->backend);
    if (!m_w_buf) throw std::runtime_error("CodecGraphs: alloc_ctx_tensors for weight buf failed");

    // Helper: reconstruct effective weight from wp0, wp1 (both f16 on device)
    // and upload to dst (f16 on device).
    auto reconstruct = [&](const std::string & wp0_name,
                           const std::string & wp1_name,
                           const std::string & bias_name,
                           ggml_tensor * dst_w,
                           ggml_tensor * dst_b,
                           int in_dim,
                           int out_dim) {
        ggml_tensor * wp0_t  = tensor_(wp0_name);
        ggml_tensor * wp1_t  = tensor_(wp1_name);
        ggml_tensor * bias_t = tensor_or_null_(bias_name);

        // wp0_t shape (out, 1, 1)  → ne=(1, 1, out), nelements = out
        // wp1_t shape (out, in, 1) → ne=(1, in, out), nelements = out*in
        if (ggml_nelements(wp0_t) != out_dim) {
            throw std::runtime_error("compute_effective_weights_: wp0 shape mismatch for " + wp0_name);
        }
        if (ggml_nelements(wp1_t) != int64_t(in_dim) * int64_t(out_dim)) {
            throw std::runtime_error("compute_effective_weights_: wp1 shape mismatch for " + wp1_name);
        }

        std::vector<uint16_t> wp0_f16, wp1_f16, bias_f16;
        read_f16_to_host_(wp0_t, wp0_f16);
        read_f16_to_host_(wp1_t, wp1_f16);
        if (bias_t) read_f16_to_host_(bias_t, bias_f16);

        // Compute: weight[o, i] = wp0[o] * wp1[o, i] / sqrt(Σi wp1[o, i]^2)
        //
        // wp1 is laid out as (out, in, 1) row-major in PyTorch; equivalently
        // ne=(1, in, out) in GGML. The contiguous memory order is the same:
        // varying along `in` is the inner stride. So wp1_f16 indexed linearly
        // in slot (o*in + i) corresponds to wp1[o, i].
        std::vector<uint16_t> w_eff(size_t(in_dim) * size_t(out_dim));
        for (int o = 0; o < out_dim; ++o) {
            float g = f16_bits_to_f32(wp0_f16[size_t(o)]);
            float ssq = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                float v = f16_bits_to_f32(wp1_f16[size_t(o) * size_t(in_dim) + size_t(i)]);
                ssq += v * v;
            }
            float inv = (ssq > 0.0f) ? 1.0f / std::sqrt(ssq) : 0.0f;
            float scale = g * inv;
            for (int i = 0; i < in_dim; ++i) {
                float v = f16_bits_to_f32(wp1_f16[size_t(o) * size_t(in_dim) + size_t(i)]);
                // dst layout: ne=(in, out), index (i, o) → linear = o*in + i.
                w_eff[size_t(o) * size_t(in_dim) + size_t(i)] = f32_to_f16_bits(scale * v);
            }
        }

        ggml_backend_tensor_set(dst_w, w_eff.data(), 0, w_eff.size() * sizeof(uint16_t));
        if (bias_t && dst_b) {
            ggml_backend_tensor_set(dst_b, bias_f16.data(), 0,
                                    bias_f16.size() * sizeof(uint16_t));
        } else if (dst_b) {
            // Zero-fill if the upstream model has no bias (shouldn't happen with
            // WNConv1d, which defaults to bias=True, but be safe).
            std::vector<uint16_t> zero(out_dim, 0);
            ggml_backend_tensor_set(dst_b, zero.data(), 0,
                                    zero.size() * sizeof(uint16_t));
        }
    };

    // Per-quantizer oprojs: Conv1d(8 → 512, kernel=1)
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        const std::string base = "moss.codec.quantizer.q." + std::to_string(i) + ".oproj.";
        reconstruct(base + "wp0", base + "wp1", base + "bias",
                     m_q_oproj_w[i], m_q_oproj_b[i],
                     CODEC_CB_DIM, CODEC_RVQ_DIM);
    }
    // Global quantizer oproj: Conv1d(512 → 768, kernel=1)
    reconstruct("moss.codec.quantizer.oproj.wp0",
                 "moss.codec.quantizer.oproj.wp1",
                 "moss.codec.quantizer.oproj.bias",
                 m_quant_oproj_w, m_quant_oproj_b,
                 CODEC_RVQ_DIM, CODEC_OUT_DIM);

    // Encoder iprojs.
    // Per-quantizer iproj: Conv1d(512 → 8, kernel=1)
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        const std::string base = "moss.codec.quantizer.q." + std::to_string(i) + ".iproj.";
        reconstruct(base + "wp0", base + "wp1", base + "bias",
                     m_q_iproj_w[i], m_q_iproj_b[i],
                     CODEC_RVQ_DIM, CODEC_CB_DIM);
    }
    // Global quantizer iproj: Conv1d(768 → 512, kernel=1)
    reconstruct("moss.codec.quantizer.iproj.wp0",
                 "moss.codec.quantizer.iproj.wp1",
                 "moss.codec.quantizer.iproj.bias",
                 m_quant_iproj_w, m_quant_iproj_b,
                 CODEC_OUT_DIM, CODEC_RVQ_DIM);
}

// Pre-compute L2-normalized codebooks (per-row L2 normalization). Used during
// encoding to compute cosine similarity via a single matmul.
void CodecGraphs::compute_normalized_codebooks_() {
    constexpr int CB_SIZE = 1024;
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        std::vector<uint16_t> cb_f16;
        read_f16_to_host_(m_codebook[i], cb_f16);
        if (cb_f16.size() != size_t(CB_SIZE) * size_t(CODEC_CB_DIM)) {
            throw std::runtime_error("compute_normalized_codebooks_: shape mismatch for q." +
                                      std::to_string(i) + ".codebook");
        }
        std::vector<uint16_t> cb_norm(cb_f16.size());
        // codebook[r, c] is at flat index r*8 + c (rows of length 8).
        for (int r = 0; r < CB_SIZE; ++r) {
            float ssq = 0.0f;
            for (int c = 0; c < CODEC_CB_DIM; ++c) {
                float v = f16_bits_to_f32(cb_f16[size_t(r) * CODEC_CB_DIM + size_t(c)]);
                ssq += v * v;
            }
            float inv = (ssq > 0.0f) ? 1.0f / std::sqrt(ssq) : 0.0f;
            for (int c = 0; c < CODEC_CB_DIM; ++c) {
                float v = f16_bits_to_f32(cb_f16[size_t(r) * CODEC_CB_DIM + size_t(c)]);
                cb_norm[size_t(r) * CODEC_CB_DIM + size_t(c)] = f32_to_f16_bits(v * inv);
            }
        }
        ggml_backend_tensor_set(m_codebook_normed[i], cb_norm.data(), 0,
                                cb_norm.size() * sizeof(uint16_t));
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Graph construction helpers
// ───────────────────────────────────────────────────────────────────────────

// Inline f16 → f32 cast helper. The CUDA bin_bcast kernel only supports
// (F32, F32, F32) for the F32 dst case; mixing f16 weights into an f32
// activation aborts. We cast the weight constants up to f32 at graph time.
static ggml_tensor * to_f32_(ggml_context * gctx, ggml_tensor * t) {
    return (t->type == GGML_TYPE_F32) ? t : ggml_cast(gctx, t, GGML_TYPE_F32);
}

ggml_tensor * CodecGraphs::build_layer_norm_(ggml_context * gctx,
                                              ggml_tensor * x,
                                              ggml_tensor * w,
                                              ggml_tensor * b) const {
    ggml_tensor * y = ggml_norm(gctx, x, NORM_EPS);
    y = ggml_mul(gctx, y, to_f32_(gctx, w));
    y = ggml_add(gctx, y, to_f32_(gctx, b));
    return y;
}

// Self-attention: x is (d_model, T). Returns (d_model, T).
//   - Fused QKV via a single mul_mat: (3*d_model, T)
//   - Split into Q, K, V each (head_dim, n_heads, T) via 3D views into the qkv buffer
//   - RoPE on Q, K (interleaved-pair convention)
//   - Manual scaled dot-product attention with causal mask
//   - Output projection
ggml_tensor * CodecGraphs::build_attention_(ggml_context * gctx,
                                             ggml_tensor * x,
                                             const Layer & L,
                                             int d_model,
                                             int n_heads,
                                             ggml_tensor * pos) const {
    const int head_dim = d_model / n_heads;
    const int T        = int(x->ne[1]);

    ggml_tensor * qkv = ggml_mul_mat(gctx, L.attn_in, x);   // (3*d_model, T)

    const size_t e        = ggml_type_size(qkv->type);
    const size_t row_size = size_t(head_dim) * e;
    const size_t qkv_nb1  = qkv->nb[1];

    ggml_tensor * Q = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, 0);
    ggml_tensor * K = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, size_t(d_model) * e);
    ggml_tensor * V = ggml_view_3d(gctx, qkv, head_dim, n_heads, T,
                                    row_size, qkv_nb1, size_t(2 * d_model) * e);

    // Make Q/K/V contiguous before RoPE so subsequent kernels see clean layouts.
    Q = ggml_cont(gctx, Q);
    K = ggml_cont(gctx, K);
    V = ggml_cont(gctx, V);

    // RoPE — interleaved-pair convention, matches MOSS' `q.view(*, D//2, 2)`.
    Q = ggml_rope_ext(gctx, Q, pos, /*c=*/nullptr,
                      head_dim, GGML_ROPE_TYPE_NORMAL, /*n_ctx_orig=*/T,
                      ROPE_FREQ_BASE, /*freq_scale=*/1.0f,
                      /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                      /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    K = ggml_rope_ext(gctx, K, pos, nullptr,
                      head_dim, GGML_ROPE_TYPE_NORMAL, T,
                      ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Manual SDPA. Permute to put T at ne[1] and heads at ne[2] (= batch).
    ggml_tensor * Qp = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));   // (head_dim, T, n_heads)
    ggml_tensor * Kp = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));   // (head_dim, T, n_heads)
    ggml_tensor * Vp = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));   // (head_dim, T, n_heads)

    // scores = Kᵀ Q  → (T_k, T_q, n_heads)
    ggml_tensor * scores = ggml_mul_mat(gctx, Kp, Qp);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

    // soft_max( scores * scale + causal_mask )
    scores = ggml_scale(gctx, scores, 1.0f / std::sqrt(float(head_dim)));
    scores = ggml_diag_mask_inf_inplace(gctx, scores, /*n_past=*/0);
    scores = ggml_soft_max(gctx, scores);

    // out = V · scores. Need T_k contracted, so transpose V's first two axes.
    ggml_tensor * Vp_t = ggml_cont(gctx, ggml_permute(gctx, Vp, 1, 0, 2, 3)); // (T, head_dim, n_heads)
    ggml_tensor * attn = ggml_mul_mat(gctx, Vp_t, scores);                    // (head_dim, T_q, n_heads)

    // back to (head_dim, n_heads, T) so we can flatten heads into d_model
    attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(gctx, attn, d_model, T);

    // output projection
    attn = ggml_mul_mat(gctx, L.attn_out, attn);   // (d_model, T)
    return attn;
}

ggml_tensor * CodecGraphs::build_ffn_(ggml_context * gctx,
                                       ggml_tensor * x,
                                       const Layer & L) const {
    ggml_tensor * y = ggml_mul_mat(gctx, L.linear1, x);
    y = ggml_gelu(gctx, y);
    y = ggml_mul_mat(gctx, L.linear2, y);
    return y;
}

ggml_tensor * CodecGraphs::build_layer_(ggml_context * gctx,
                                         ggml_tensor * x,
                                         const Layer & L,
                                         int d_model,
                                         int n_heads,
                                         ggml_tensor * pos) const {
    // attn block
    ggml_tensor * y = build_layer_norm_(gctx, x, L.norm1_w, L.norm1_b);
    y = build_attention_(gctx, y, L, d_model, n_heads, pos);
    y = ggml_mul(gctx, y, to_f32_(gctx, L.layer_scale_1));
    x = ggml_add(gctx, x, y);

    // ffn block
    y = build_layer_norm_(gctx, x, L.norm2_w, L.norm2_b);
    y = build_ffn_(gctx, y, L);
    y = ggml_mul(gctx, y, to_f32_(gctx, L.layer_scale_2));
    x = ggml_add(gctx, x, y);

    return x;
}

// Run a ProjectedTransformer stage: optional input proj → 12/32 layers → optional output proj.
ggml_tensor * CodecGraphs::build_stage_(ggml_context * gctx,
                                         ggml_tensor * x,
                                         const Stage & S,
                                         ggml_tensor * pos) const {
    if (S.iproj) x = ggml_mul_mat(gctx, S.iproj, x);
    for (const Layer & L : S.layers) {
        x = build_layer_(gctx, x, L, S.spec.d_model, S.spec.n_heads, pos);
    }
    if (S.oproj) x = ggml_mul_mat(gctx, S.oproj, x);
    return x;
}

// PyTorch:  x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l*h)
//   input  ne=(d*h, T_in)
//   output ne=(d,   T_in*h)
//
// In GGML terms:
//   1. reshape (d*h, T_in)         → (h, d, T_in)
//   2. permute (1, 0, 2)            → (d, h, T_in) (non-contig)
//   3. cont                          → (d, h, T_in) contiguous
//   4. reshape (d, h*T_in)          (memory: h innermost after d, then T_in)
ggml_tensor * CodecGraphs::patch_upsample_(ggml_context * gctx,
                                            ggml_tensor * x, int patch) {
    const int64_t dh = x->ne[0];
    const int64_t T_in = x->ne[1];
    if (dh % patch != 0) {
        throw std::runtime_error("patch_upsample_: channel count " + std::to_string(dh)
                                 + " not divisible by patch=" + std::to_string(patch));
    }
    const int64_t d = dh / patch;
    ggml_tensor * y = ggml_reshape_3d(gctx, x, patch, d, T_in);
    y = ggml_permute(gctx, y, 1, 0, 2, 3);                  // (d, h, T_in)
    y = ggml_cont(gctx, y);                                  // contiguous (d, h, T_in)
    y = ggml_reshape_2d(gctx, y, d, T_in * patch);
    return y;
}

// Inverse of patch_upsample_. PyTorch encode:
//   x.reshape(b, d, T_in/h, h).permute(0, 1, 3, 2).reshape(b, d*h, T_in/h)
//   ⇒ out[d_out = d*h + h_idx, t_out] = in[d, t_out*h + h_idx]
// In GGML (D innermost):
//   1. ggml_reshape_3d(x, D, h, T_out)        [view ne=(D, T_in) as ne=(D, h, T_out)]
//   2. ggml_permute(_, 1, 0, 2, 3)            → ne=(h, D, T_out)
//   3. ggml_cont(_)                            → contiguous (h, D, T_out)
//   4. ggml_reshape_2d(_, D*h, T_out)
ggml_tensor * CodecGraphs::patch_downsample_(ggml_context * gctx,
                                              ggml_tensor * x, int patch) {
    const int64_t D = x->ne[0];
    const int64_t T_in = x->ne[1];
    if (T_in % patch != 0) {
        throw std::runtime_error("patch_downsample_: T_in " + std::to_string(T_in)
                                 + " not divisible by patch=" + std::to_string(patch));
    }
    const int64_t T_out = T_in / patch;
    ggml_tensor * y = ggml_reshape_3d(gctx, x, D, patch, T_out);
    y = ggml_permute(gctx, y, 1, 0, 2, 3);                  // (h, D, T_out)
    y = ggml_cont(gctx, y);                                  // contiguous (h, D, T_out)
    y = ggml_reshape_2d(gctx, y, D * patch, T_out);
    return y;
}

// ───────────────────────────────────────────────────────────────────────────
// Decode entry point
// ───────────────────────────────────────────────────────────────────────────

std::vector<float> CodecGraphs::decode(const int32_t * codes,
                                        int32_t n_vq,
                                        int32_t T_audio) {
    auto * aux = m_owner.aux();
    if (!aux || !aux->backend) {
        throw std::runtime_error("CodecGraphs::decode: aux backend not initialised");
    }
    if (!m_owner.codec_present()) {
        throw std::runtime_error("CodecGraphs::decode: codec tensors not loaded "
                                  "(rebuild with --codec, and don't pass --skip-codec)");
    }
    if (n_vq != CODEC_NUM_VQ) {
        throw std::runtime_error("CodecGraphs::decode: expected n_vq=32, got "
                                  + std::to_string(n_vq));
    }
    if (T_audio <= 0) return {};

    const int n_samples = T_audio * 1920;

    // ── Build graph ────────────────────────────────────────────────────────
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("CodecGraphs::decode: ggml_init failed");

    // Inputs:
    //   - codes_in[i]: (T_audio,) int32, one per quantizer
    //   - pos_T[s]:    (T_at_stage_s,) int32, one per decoder stage (T grows)
    std::vector<ggml_tensor *> codes_in(CODEC_NUM_VQ);
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        codes_in[i] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_audio);
        ggml_set_name(codes_in[i], ("codes_" + std::to_string(i)).c_str());
        ggml_set_input(codes_in[i]);
    }

    // Per-stage RoPE position tensors. Each stage operates at a different T.
    int T_at[4];
    T_at[0] = T_audio;
    T_at[1] = T_at[0] * DECODER_STAGES[0].patch_after;
    T_at[2] = T_at[1] * DECODER_STAGES[1].patch_after;
    T_at[3] = T_at[2] * DECODER_STAGES[2].patch_after;
    std::array<ggml_tensor *, 4> pos_T {};
    for (int s = 0; s < 4; ++s) {
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
    }

    // ── Quantizer.decode_codes ─────────────────────────────────────────────
    ggml_tensor * sum = nullptr;
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        // (codebook_dim=8, T) via embedding lookup → f32 (get_rows promotes)
        ggml_tensor * z = ggml_get_rows(gctx, m_codebook[i], codes_in[i]);
        // Conv1d 8 → 512 (kernel=1). w stored as (in=8, out=512) f16.
        z = ggml_mul_mat(gctx, m_q_oproj_w[i], z);                   // (512, T) f32
        z = ggml_add(gctx, z, to_f32_(gctx, m_q_oproj_b[i]));        // broadcast bias
        sum = sum ? ggml_add(gctx, sum, z) : z;
    }
    // Final rvq oproj: 512 → 768
    ggml_tensor * x = ggml_mul_mat(gctx, m_quant_oproj_w, sum);     // (768, T) f32
    x = ggml_add(gctx, x, to_f32_(gctx, m_quant_oproj_b));          // (768, T)

    // ── 4 transformer stages, each followed by a patch upsample ───────────
    for (int s = 0; s < 4; ++s) {
        x = build_stage_(gctx, x, m_stages[s], pos_T[s]);
        if (m_stages[s].spec.patch_after > 0) {
            x = patch_upsample_(gctx, x, m_stages[s].spec.patch_after);
        }
    }

    // After dec.7 the channel dim is 1 and time is T_audio*1920. View as 1D.
    ggml_tensor * waveform = ggml_reshape_1d(gctx, x, n_samples);
    ggml_set_name(waveform, "waveform");
    ggml_set_output(waveform);

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, 65536, false);
    ggml_build_forward_expand(graph, waveform);

    if (!ggml_gallocr_alloc_graph(m_galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("CodecGraphs::decode: gallocr_alloc_graph failed");
    }

    // ── Upload inputs ──────────────────────────────────────────────────────
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        std::vector<int32_t> col(T_audio);
        for (int t = 0; t < T_audio; ++t) col[size_t(t)] = codes[i * T_audio + t];
        ggml_backend_tensor_set(codes_in[i], col.data(), 0,
                                size_t(T_audio) * sizeof(int32_t));
    }
    for (int s = 0; s < 4; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0,
                                p.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("CodecGraphs::decode: graph_compute failed");
    }

    std::vector<float> wav;
    wav.resize(size_t(n_samples));
    ggml_backend_tensor_get(waveform, wav.data(), 0, wav.size() * sizeof(float));

    ggml_free(gctx);
    return wav;
}

// ───────────────────────────────────────────────────────────────────────────
// Encode entry point
// ───────────────────────────────────────────────────────────────────────────

std::vector<int32_t> CodecGraphs::encode(const float * waveform,
                                          int64_t       n_samples,
                                          int32_t &     T_audio_out) {
    auto * aux = m_owner.aux();
    if (!aux || !aux->backend) {
        throw std::runtime_error("CodecGraphs::encode: aux backend not initialised");
    }
    if (!m_owner.codec_loaded()) {
        throw std::runtime_error("CodecGraphs::encode: codec tensors not loaded");
    }
    if (n_samples <= 0) { T_audio_out = 0; return {}; }

    // Pad waveform to a multiple of downsample_rate (1920) — same convention
    // as MossAudioTokenizerModel._encode_frame.
    const int64_t hop = 1920;
    int64_t T_wav = n_samples;
    int64_t pad = (hop - (T_wav % hop)) % hop;
    int64_t T_padded = T_wav + pad;
    const int32_t T_audio = int32_t(T_padded / hop);
    T_audio_out = T_audio;

    // ── Build graph ────────────────────────────────────────────────────────
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("CodecGraphs::encode: ggml_init failed");

    // Input: padded waveform as a 2D tensor (1 channel × T_padded).
    ggml_tensor * wav = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, T_padded);
    ggml_set_name(wav, "waveform");
    ggml_set_input(wav);

    // Per-stage RoPE positions, one per encoder transformer stage.
    int T_at[4];
    T_at[0] = int(T_padded / CODEC_PRE_PATCH);                       // input to enc.1
    T_at[1] = T_at[0] / ENCODER_STAGES[0].patch_after;               // input to enc.3
    T_at[2] = T_at[1] / ENCODER_STAGES[1].patch_after;               // input to enc.5
    T_at[3] = T_at[2] / ENCODER_STAGES[2].patch_after;               // input to enc.7
    std::array<ggml_tensor *, 4> pos_T {};
    for (int s = 0; s < 4; ++s) {
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("enc_pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
    }

    // Initial patch=240 downsample: (1, T_padded) → (240, T_padded/240)
    ggml_tensor * x = patch_downsample_(gctx, wav, CODEC_PRE_PATCH);

    // Four encoder stages, each followed by a patch downsample (except the last).
    for (int s = 0; s < 4; ++s) {
        x = build_stage_(gctx, x, m_enc_stages[s], pos_T[s]);
        if (m_enc_stages[s].spec.patch_after > 0) {
            x = patch_downsample_(gctx, x, m_enc_stages[s].spec.patch_after);
        }
    }

    // ── Quantizer: input_proj → 32-step residual LFQ encoding ─────────────
    // x: (768, T_audio).
    ggml_tensor * residual = ggml_mul_mat(gctx, m_quant_iproj_w, x);          // (512, T)
    residual = ggml_add(gctx, residual, to_f32_(gctx, m_quant_iproj_b));      // (512, T)

    std::array<ggml_tensor *, CODEC_NUM_VQ> indices {};
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        // z_e = q[i].iproj(residual)
        ggml_tensor * z_e = ggml_mul_mat(gctx, m_q_iproj_w[i], residual);     // (8, T)
        z_e = ggml_add(gctx, z_e, to_f32_(gctx, m_q_iproj_b[i]));             // (8, T)

        // L2-normalize per timestep (along the 8-dim, which is ne[0]).
        ggml_tensor * z_e_n = ggml_l2_norm(gctx, z_e, /*eps=*/1e-12f);        // (8, T)

        // similarity = codebook_normed[i] @ z_e_n  → (1024, T)
        // codebook_normed has ne=(8, 1024); mul_mat contracts ne[0]=8.
        ggml_tensor * sim = ggml_mul_mat(gctx, m_codebook_normed[i], z_e_n);  // (1024, T)
        ggml_mul_mat_set_prec(sim, GGML_PREC_F32);

        // Argmax along ne[0] → (T,) i32 — cosine-similarity nearest neighbour.
        ggml_tensor * idx = ggml_argmax(gctx, sim);                            // (T,) i32
        ggml_set_name(idx, ("idx_" + std::to_string(i)).c_str());
        ggml_set_output(idx);
        indices[i] = idx;

        // Residual update: residual -= q[i].oproj( codebook[i][idx] )
        // (raw codebook here, not the normalized version — matches LFQ.decode_code_wo_out_proj)
        ggml_tensor * z_q = ggml_get_rows(gctx, m_codebook[i], idx);          // (8, T) f32
        z_q = ggml_mul_mat(gctx, m_q_oproj_w[i], z_q);                        // (512, T)
        z_q = ggml_add(gctx, z_q, to_f32_(gctx, m_q_oproj_b[i]));             // (512, T)
        residual = ggml_sub(gctx, residual, z_q);                              // (512, T)
    }

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, 65536, false);
    for (int i = 0; i < CODEC_NUM_VQ; ++i) ggml_build_forward_expand(graph, indices[i]);

    if (!ggml_gallocr_alloc_graph(m_galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("CodecGraphs::encode: gallocr_alloc_graph failed");
    }

    // ── Upload inputs ─────────────────────────────────────────────────────
    {
        std::vector<float> wpad;
        wpad.assign(size_t(T_padded), 0.0f);
        std::memcpy(wpad.data(), waveform, size_t(T_wav) * sizeof(float));
        ggml_backend_tensor_set(wav, wpad.data(), 0, wpad.size() * sizeof(float));
    }
    for (int s = 0; s < 4; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("CodecGraphs::encode: graph_compute failed");
    }

    // ── Read indices back: assemble (n_vq, T_audio) row-major i32 ─────────
    std::vector<int32_t> out;
    out.assign(size_t(CODEC_NUM_VQ) * size_t(T_audio), 0);
    for (int i = 0; i < CODEC_NUM_VQ; ++i) {
        ggml_backend_tensor_get(indices[i], out.data() + size_t(i) * size_t(T_audio),
                                0, size_t(T_audio) * sizeof(int32_t));
    }

    ggml_free(gctx);
    return out;
}

// ───────────────────────────────────────────────────────────────────────────
// Model ctor/dtor + CodecGraphs accessor.
//
// Defined here so the unique_ptr<CodecGraphs> and unique_ptr<Tokenizer>
// destructors see the complete types (both forward-declared in model.h).
// ───────────────────────────────────────────────────────────────────────────

Model::Model()  = default;
Model::~Model() = default;

CodecGraphs * Model::codec() {
    if (!m_codec) m_codec = std::make_unique<CodecGraphs>(*this);
    return m_codec.get();
}

// ───────────────────────────────────────────────────────────────────────────
// Public free functions
// ───────────────────────────────────────────────────────────────────────────

std::vector<int32_t> codec_encode(Model & model,
                                   const float * waveform,
                                   int64_t       n_samples,
                                   int32_t &     n_vq_out,
                                   int32_t &     t_audio_out) {
    n_vq_out = CODEC_NUM_VQ;
    CodecGraphs * cg = model.codec();
    if (!cg) throw std::runtime_error("codec_encode: codec not available on this model");
    return cg->encode(waveform, n_samples, t_audio_out);
}

std::vector<float> codec_decode(Model & model,
                                 const int32_t * codes,
                                 int32_t         n_vq,
                                 int32_t         t_audio) {
    CodecGraphs * cg = model.codec();
    if (!cg) throw std::runtime_error("codec_decode: codec not available on this model");
    return cg->decode(codes, n_vq, t_audio);
}

} // namespace openmoss
