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
    int context;         // causal attention window in keys:
                         // stage frame rate × causal_transformer_context_duration (10 s)
};

// MOSS-Audio-Tokenizer v1 — 24 kHz mono, hop 1920, 4 transformer stages/side.
constexpr std::array<StageSpec, 4> DECODER_STAGES_V1 = {{
    //   in    d   nh   dff  nl  out  patch  gguf  ctx
    {  768, 1280, 20, 5120, 32, 1280,    2,    0,  125 },  // dec.0 @ 12.5 Hz + dec.1 patch
    {  640,  768, 12, 3072, 12,  768,    2,    2,  250 },  // dec.2 @ 25 Hz   + dec.3 patch
    {  384,  768, 12, 3072, 12,  768,    2,    4,  500 },  // dec.4 @ 50 Hz   + dec.5 patch
    {  384,  768, 12, 3072, 12,  240,  240,    6, 1000 },  // dec.6 @ 100 Hz  + dec.7 patch
}};

constexpr std::array<StageSpec, 4> ENCODER_STAGES_V1 = {{
    //   in    d   nh   dff  nl  out  patch  gguf  ctx
    {  240,  768, 12, 3072, 12,  384,    2,    1, 1000 },  // enc.1 @ 100 Hz  + enc.2 patch
    {  768,  768, 12, 3072, 12,  384,    2,    3,  500 },  // enc.3 @ 50 Hz   + enc.4 patch
    {  768,  768, 12, 3072, 12,  640,    2,    5,  250 },  // enc.5 @ 25 Hz   + enc.6 patch
    { 1280, 1280, 20, 5120, 32,  768,    0,    7,  125 },  // enc.7 @ 12.5 Hz (no patch after)
}};

// MOSS-Audio-Tokenizer v2 — 48 kHz stereo, hop 3840/channel, 6 stages/side.
//
// Same layer math as v1, two more stages per side, and per-stage attention
// windows instead of a single 10 s one. Upstream computes each window as
// round(stage_frame_rate × context_duration) where the rate is measured in
// *interleaved* elements (48000 × 2 channels = 96000 at the waveform end) and
// context_duration comes from that stage's kwargs — {10,10,8,4,2,1} s from the
// frame end outwards. The rates below are the interleaved-element rates.
constexpr std::array<StageSpec, 6> DECODER_STAGES_V2 = {{
    //   in    d   nh   dff  nl  out  patch  gguf  ctx
    {  768, 1280, 20, 5120, 32, 1280,    2,    0,  125 },  // dec.0  @ 12.5 Hz × 10 s
    {  640,  768, 12, 3072, 12,  768,    2,    2,  250 },  // dec.2  @ 25 Hz   × 10 s
    {  384,  768, 12, 3072, 12,  768,    2,    4,  400 },  // dec.4  @ 50 Hz   ×  8 s
    {  384,  768, 12, 3072, 12,  768,    2,    6,  400 },  // dec.6  @ 100 Hz  ×  4 s
    {  384,  768, 12, 3072, 12,  768,    2,    8,  400 },  // dec.8  @ 200 Hz  ×  2 s
    {  384,  768, 12, 3072, 12,  240,  240,   10,  400 },  // dec.10 @ 400 Hz  ×  1 s
}};

constexpr std::array<StageSpec, 6> ENCODER_STAGES_V2 = {{
    //   in    d   nh   dff  nl  out  patch  gguf  ctx
    {  240,  768, 12, 3072, 12,  384,    2,    1,  400 },  // enc.1  @ 400 Hz  ×  1 s
    {  768,  768, 12, 3072, 12,  384,    2,    3,  400 },  // enc.3  @ 200 Hz  ×  2 s
    {  768,  768, 12, 3072, 12,  384,    2,    5,  400 },  // enc.5  @ 100 Hz  ×  4 s
    {  768,  768, 12, 3072, 12,  384,    2,    7,  400 },  // enc.7  @ 50 Hz   ×  8 s
    {  768,  768, 12, 3072, 12,  640,    2,    9,  250 },  // enc.9  @ 25 Hz   × 10 s
    { 1280, 1280, 20, 5120, 32,  768,    0,   11,  125 },  // enc.11 @ 12.5 Hz × 10 s
}};

// The initial patch=240 happens BEFORE the first encoder transformer stage;
// `patch_after` is the patch that follows a stage. Same in both generations.
constexpr int CODEC_PRE_PATCH = 240;

std::vector<StageSpec> stages_for(int codec_version, bool decoder) {
    switch (codec_version) {
        case 1: return decoder
            ? std::vector<StageSpec>(DECODER_STAGES_V1.begin(), DECODER_STAGES_V1.end())
            : std::vector<StageSpec>(ENCODER_STAGES_V1.begin(), ENCODER_STAGES_V1.end());
        case 2: return decoder
            ? std::vector<StageSpec>(DECODER_STAGES_V2.begin(), DECODER_STAGES_V2.end())
            : std::vector<StageSpec>(ENCODER_STAGES_V2.begin(), ENCODER_STAGES_V2.end());
        default:
            throw std::runtime_error("unsupported moss.codec.version: "
                                     + std::to_string(codec_version));
    }
}

// Every matmul here accumulates in f32.
//
// ggml's Vulkan backend picks an f16-accumulate pipeline whenever a node is left
// at GGML_PREC_DEFAULT, which over these dot products costs ~1e-2 relative on the
// decoded waveform — sixteen times what f32 accumulation gives, and far more than
// the f16 weights themselves contribute. It hides well because matrix-*vector*
// products take a different, f32-accumulating path, so the audio heads and the
// depth transformer (which run one token at a time) are unaffected and only this
// file, which pushes thousands of frames through 68 transformer layers, suffers.
ggml_tensor * mm_(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

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
    // Sized from moss.codec.version at construction: 4 stages/side for v1,
    // 6 for v2.
    std::vector<Stage> m_stages;      // decoder stages (dec.0/2/4/… )
    std::vector<Stage> m_enc_stages;  // encoder stages (enc.1/3/5/… )

    ggml_gallocr_t m_galloc = nullptr;

    // ── private helpers ──
    ggml_tensor * tensor_(const std::string & name) const;
    ggml_tensor * tensor_or_null_(const std::string & name) const;
    void resolve_decoder_();
    void resolve_encoder_();
    void validate_stage_(const Stage & S, const char * side) const;
    void compute_effective_weights_();
    void compute_normalized_codebooks_();
    ggml_tensor * read_f16_to_host_(ggml_tensor * t, std::vector<uint16_t> & out) const;

    // Graph construction:
    ggml_tensor * build_layer_norm_(ggml_context * gctx, ggml_tensor * x,
                                    ggml_tensor * w, ggml_tensor * b) const;
    // How one stage's attention is carved up over the time axis.
    //
    // Every stage is banded: query q attends only to keys in (q - context, q].
    // A dense (T, T) mask therefore wastes ~98% of its entries on -inf, and at
    // v2's deepest decoder stage — 32x the frame rate — the masks across all
    // stages come to ~2730 * T_audio^2 bytes: 384 MiB at 30 s, 1.5 GiB at 60 s,
    // in host RAM *and* backend memory.
    //
    // Instead, split the queries into chunks and give each chunk a mask over
    // only the keys it can reach: [q0 - context + 1, q1). Memory falls to
    // ~T * (chunk + context), and the attention FLOPs drop from O(T^2) to
    // O(T * context). One block covering everything reproduces the old
    // behaviour exactly, which is what short stages get.
    struct AttnPlan {
        struct Block {
            int64_t q0 = 0, q1 = 0;   // query range
            int64_t k0 = 0;           // first reachable key
            ggml_tensor * mask = nullptr;   // (n_kv, n_q) f16
        };
        int64_t T = 0;
        int64_t context = 0;
        std::vector<Block> blocks;
    };

    ggml_tensor * build_attention_(ggml_context * gctx, ggml_tensor * x,
                                    const Layer & L, int d_model, int n_heads,
                                    ggml_tensor * pos, const AttnPlan & plan) const;
    ggml_tensor * build_ffn_(ggml_context * gctx, ggml_tensor * x,
                              const Layer & L) const;
    ggml_tensor * build_layer_(ggml_context * gctx, ggml_tensor * x,
                                const Layer & L, int d_model, int n_heads,
                                ggml_tensor * pos, const AttnPlan & plan) const;
    ggml_tensor * build_stage_(ggml_context * gctx, ggml_tensor * x,
                                const Stage & S, ggml_tensor * pos,
                                const AttnPlan & plan) const;
    static AttnPlan make_attn_plan_(ggml_context * gctx, int64_t T, int64_t context);
    static void     fill_attn_plan_(const AttnPlan & plan);
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
    const auto specs = stages_for(m_owner.dims().codec_version, /*decoder=*/true);
    m_stages.resize(specs.size());
    for (size_t s = 0; s < m_stages.size(); ++s) {
        Stage & S = m_stages[s];
        S.spec = specs[s];
        const std::string base = "moss.codec.dec." + std::to_string(S.spec.gguf_idx) + ".";

        // Look these up unconditionally rather than inferring their existence
        // from the dimensions. v1 used nn.Identity() when d_model matched, so
        // the tensors genuinely are absent there; v2 always materialises an
        // nn.Linear, including ten *square* learned projections. Keying off
        // `d_model != input_dim` would skip those silently and produce subtly
        // wrong audio instead of a load error.
        S.iproj = tensor_or_null_(base + "iproj.weight");
        S.oproj = tensor_or_null_(base + "oproj.weight");

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
        validate_stage_(S, "dec");
    }
}

// Resolve encoder weights (mirror of resolve_decoder_).
void CodecGraphs::resolve_encoder_() {
    const auto specs = stages_for(m_owner.dims().codec_version, /*decoder=*/false);
    m_enc_stages.resize(specs.size());
    for (size_t s = 0; s < m_enc_stages.size(); ++s) {
        Stage & S = m_enc_stages[s];
        S.spec = specs[s];
        const std::string base = "moss.codec.enc." + std::to_string(S.spec.gguf_idx) + ".";

        S.iproj = tensor_or_null_(base + "iproj.weight");   // see resolve_decoder_
        S.oproj = tensor_or_null_(base + "oproj.weight");

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
        validate_stage_(S, "enc");
    }
}

// A stage projection may legitimately be absent only when it would have been an
// nn.Identity() upstream — i.e. when the dimensions it maps between are equal.
// Anything else means the sidecar is missing a learned matrix, which would
// otherwise degrade the audio silently rather than failing. Shapes are checked
// too, since GGUF carries no type information beyond the name.
void CodecGraphs::validate_stage_(const Stage & S, const char * side) const {
    const std::string where = std::string(side) + "." + std::to_string(S.spec.gguf_idx);

    auto check = [&](ggml_tensor * t, const char * what, int in_dim, int out_dim) {
        if (!t) {
            if (in_dim != out_dim) {
                throw std::runtime_error(
                    "CodecGraphs: " + where + "." + what + " is missing from the sidecar, "
                    "but it maps " + std::to_string(in_dim) + " → " + std::to_string(out_dim)
                    + " so it cannot be an identity. Reconvert the model.");
            }
            return;   // genuinely nn.Identity() upstream
        }
        if (t->ne[0] != in_dim || t->ne[1] != out_dim) {
            throw std::runtime_error(
                "CodecGraphs: " + where + "." + what + " has shape ("
                + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1])
                + "), expected (" + std::to_string(in_dim) + ", "
                + std::to_string(out_dim) + ")");
        }
    };

    check(S.iproj, "iproj", S.spec.input_dim, S.spec.d_model);
    check(S.oproj, "oproj", S.spec.d_model,   S.spec.output_dim);
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
                                             ggml_tensor * pos,
                                             const AttnPlan & plan) const {
    const int head_dim = d_model / n_heads;
    const int T        = int(x->ne[1]);

    ggml_tensor * qkv = mm_(gctx, L.attn_in, x);   // (3*d_model, T)

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

    // Flash attention (causal). We previously did a manual SDPA
    // (scores = KᵀQ → scale → diag_mask_inf → soft_max → ·V), but ggml_soft_max
    // aborts on some CUDA builds ("SOFT_MAX failed: invalid argument") and the
    // explicit (T×T×n_heads) scores buffer is O(T²) memory — at long T that
    // alone needs tens of GB. ggml_flash_attn_ext streams the softmax (no
    // materialised scores) and runs the same well-exercised kernel the llama
    // backbone uses, so it works on every backend we target.
    //
    // Layout flash_attn_ext expects: q/k/v = (head_dim, T, n_heads).
    ggml_tensor * Qp = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));   // (head_dim, T, n_heads) f32
    ggml_tensor * Kp = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
    ggml_tensor * Vp = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

    // The kernel wants F16 K/V (matches the backbone's KV path); the mask is
    // already F16. Q stays F32.
    ggml_tensor * Kh = ggml_cast(gctx, Kp, GGML_TYPE_F16);
    ggml_tensor * Vh = ggml_cast(gctx, Vp, GGML_TYPE_F16);

    const float scale = 1.0f / std::sqrt(float(head_dim));

    // One flash-attention call per query chunk, each seeing only the keys its
    // queries can actually reach. Results concatenate along the time axis.
    ggml_tensor * attn = nullptr;
    for (const auto & b : plan.blocks) {
        const int64_t n_q  = b.q1 - b.q0;
        const int64_t n_kv = b.q1 - b.k0;

        ggml_tensor * q = Qp;
        ggml_tensor * k = Kh;
        ggml_tensor * v = Vh;
        if (n_q != T) {
            q = ggml_cont(gctx, ggml_view_3d(gctx, Qp, head_dim, n_q, n_heads,
                                             Qp->nb[1], Qp->nb[2],
                                             size_t(b.q0) * Qp->nb[1]));
        }
        if (n_kv != T) {
            k = ggml_cont(gctx, ggml_view_3d(gctx, Kh, head_dim, n_kv, n_heads,
                                             Kh->nb[1], Kh->nb[2],
                                             size_t(b.k0) * Kh->nb[1]));
            v = ggml_cont(gctx, ggml_view_3d(gctx, Vh, head_dim, n_kv, n_heads,
                                             Vh->nb[1], Vh->nb[2],
                                             size_t(b.k0) * Vh->nb[1]));
        }

        ggml_tensor * a = ggml_flash_attn_ext(gctx, q, k, v, b.mask, scale,
                                              /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
        // (head_dim, n_heads, n_q)
        attn = attn ? ggml_concat(gctx, attn, a, 2) : a;
    }
    // → flatten heads into d_model.
    attn = ggml_reshape_2d(gctx, attn, d_model, T);

    // output projection
    attn = mm_(gctx, L.attn_out, attn);   // (d_model, T)
    return attn;
}

ggml_tensor * CodecGraphs::build_ffn_(ggml_context * gctx,
                                       ggml_tensor * x,
                                       const Layer & L) const {
    ggml_tensor * y = mm_(gctx, L.linear1, x);
    y = ggml_gelu(gctx, y);
    y = mm_(gctx, L.linear2, y);
    return y;
}

ggml_tensor * CodecGraphs::build_layer_(ggml_context * gctx,
                                         ggml_tensor * x,
                                         const Layer & L,
                                         int d_model,
                                         int n_heads,
                                         ggml_tensor * pos,
                                         const AttnPlan & plan) const {
    // attn block
    ggml_tensor * y = build_layer_norm_(gctx, x, L.norm1_w, L.norm1_b);
    y = build_attention_(gctx, y, L, d_model, n_heads, pos, plan);
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
                                         ggml_tensor * pos,
                                         const AttnPlan & plan) const {
    if (S.iproj) x = mm_(gctx, S.iproj, x);
    for (const Layer & L : S.layers) {
        x = build_layer_(gctx, x, L, S.spec.d_model, S.spec.n_heads, pos, plan);
    }
    if (S.oproj) x = mm_(gctx, S.oproj, x);
    return x;
}

// Chunk width, in queries. Bigger chunks mean fewer flash-attention calls but a
// wider mask per chunk (memory ~ T * (chunk + context)). 1024 keeps both small:
// at T = 16000 and context = 400 the masks total ~46 MiB instead of ~512 MiB.
static constexpr int64_t ATTN_CHUNK = 1024;

CodecGraphs::AttnPlan CodecGraphs::make_attn_plan_(ggml_context * gctx,
                                                   int64_t T, int64_t context) {
    AttnPlan plan;
    plan.T       = T;
    plan.context = context;
    if (T <= 0) return plan;

    // Short stages stay in one block, which is bit-for-bit the previous
    // behaviour: a single (T, T) mask and one flash-attention call.
    const int64_t chunk = (T <= ATTN_CHUNK) ? T : ATTN_CHUNK;

    for (int64_t q0 = 0; q0 < T; q0 += chunk) {
        AttnPlan::Block b;
        b.q0 = q0;
        b.q1 = std::min(q0 + chunk, T);
        // Query q reaches back to q - context + 1; the earliest query in this
        // block therefore bounds the key window.
        b.k0 = (context > 0) ? std::max<int64_t>(0, q0 - context + 1) : 0;
        b.mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, b.q1 - b.k0, b.q1 - b.q0);
        ggml_set_input(b.mask);
        plan.blocks.push_back(b);
    }
    return plan;
}

// The codec transformers are trained with a *sliding-window* causal mask
// (per-stage `context_duration` x the stage's frame rate; upstream:
// `attn_bias = (delta >= 0) & (delta < context)`). Attending past that window
// feeds the model RoPE distances it never saw in training and audio quality
// collapses progressively past the window (issue #7), so the band limit is not
// an optimisation — it is part of the model.
void CodecGraphs::fill_attn_plan_(const AttnPlan & plan) {
    static const uint16_t F16_ZERO    = 0x0000;
    static const uint16_t F16_NEG_INF = 0xFC00;
    std::vector<uint16_t> buf;
    for (const auto & b : plan.blocks) {
        const int64_t n_kv = b.q1 - b.k0;
        const int64_t n_q  = b.q1 - b.q0;
        buf.assign(size_t(n_kv) * size_t(n_q), F16_NEG_INF);
        for (int64_t qi = 0; qi < n_q; ++qi) {
            const int64_t q = b.q0 + qi;
            uint16_t * row = buf.data() + size_t(qi) * size_t(n_kv);
            // Global key k is visible iff k <= q and q - k < context.
            const int64_t lo = (plan.context > 0)
                             ? std::max(b.k0, q - plan.context + 1) : b.k0;
            for (int64_t k = lo; k <= q; ++k) row[k - b.k0] = F16_ZERO;
        }
        ggml_backend_tensor_set(b.mask, buf.data(), 0, buf.size() * sizeof(uint16_t));
    }
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
    // Models may use a prefix of the RVQ codebooks (MOSS-TTS = all 32; smaller
    // variants like MOSS-VoiceGenerator generate 16). Decode the first n_vq
    // codebooks; the codec sums residuals coarse-to-fine, so a prefix is valid.
    if (n_vq < 1 || n_vq > CODEC_NUM_VQ) {
        throw std::runtime_error("CodecGraphs::decode: n_vq must be 1.."
                                  + std::to_string(CODEC_NUM_VQ) + ", got " + std::to_string(n_vq));
    }
    if (T_audio <= 0) return {};

    // Flash attention streams the softmax, so there's no O(T²) scores buffer
    // anymore — but each stage still needs an (T×T) f16 causal mask, and T grows
    // by the patch factor at every stage. Sum over all of them: for v2 the last
    // decoder stage runs at 32·T_audio, so the masks total ≈ 2730·T_audio² bytes
    // (384 MiB at 30 s, 1.5 GiB at 60 s). A runaway generation — the model never
    // emitting an end-of-speech token — would balloon this to many GB, so refuse
    // early with an actionable message instead of attempting a doomed allocation.
    {
        int64_t T = T_audio;
        int64_t mask_bytes = 0;
        for (const auto & S : m_stages) {
            // Chunked plan: ceil(T/chunk) blocks, each (chunk + context) x chunk.
            const int64_t chunk = std::min<int64_t>(T, 1024);
            if (chunk > 0) {
                const int64_t nblk = (T + chunk - 1) / chunk;
                mask_bytes += nblk * (chunk + S.spec.context) * chunk
                            * int64_t(sizeof(uint16_t));
            }
            if (S.spec.patch_after > 0) T *= S.spec.patch_after;
        }
        const int64_t cap_bytes = int64_t(8) * 1024 * 1024 * 1024;  // ~8 GiB
        if (mask_bytes > cap_bytes) {
            const double secs = double(T_audio) * double(m_owner.dims().downsample_rate)
                              / double(m_owner.dims().sampling_rate);
            throw std::runtime_error(
                "CodecGraphs::decode: refusing to decode " + std::to_string(T_audio) +
                " frames (~" + std::to_string(int(secs)) + "s): the codec attention "
                "masks alone would need ~" + std::to_string(mask_bytes >> 30) +
                " GiB. This usually means generation never emitted an end-of-speech "
                "token (try a lower --max-new-tokens / different sampling).");
        }
    }

    // Samples the final unpatch emits, across all channels. Stereo is carried as
    // L/R time-interleaving through a 1-channel network, so this buffer is
    // already in interleaved order — exactly what a WAV writer wants.
    const int64_t n_samples = int64_t(T_audio) * m_owner.dims().samples_per_frame();

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
    std::vector<ggml_tensor *> codes_in(n_vq);
    for (int i = 0; i < n_vq; ++i) {
        codes_in[i] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_audio);
        ggml_set_name(codes_in[i], ("codes_" + std::to_string(i)).c_str());
        ggml_set_input(codes_in[i]);
    }

    // Per-stage RoPE position tensors. Each stage operates at a different T,
    // growing by the previous stage's patch factor.
    const size_t n_stages = m_stages.size();
    std::vector<int>           T_at(n_stages);
    std::vector<ggml_tensor *> pos_T(n_stages, nullptr);
    std::vector<AttnPlan> plan_T(n_stages);
    for (size_t s = 0; s < n_stages; ++s) {
        T_at[s] = (s == 0) ? T_audio : T_at[s - 1] * m_stages[s - 1].spec.patch_after;
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
        plan_T[s] = make_attn_plan_(gctx, T_at[s], m_stages[s].spec.context);
    }

    // ── Quantizer.decode_codes ─────────────────────────────────────────────
    ggml_tensor * sum = nullptr;
    for (int i = 0; i < n_vq; ++i) {
        // (codebook_dim=8, T) via embedding lookup → f32 (get_rows promotes)
        ggml_tensor * z = ggml_get_rows(gctx, m_codebook[i], codes_in[i]);
        // Conv1d 8 → 512 (kernel=1). w stored as (in=8, out=512) f16.
        z = mm_(gctx, m_q_oproj_w[i], z);                   // (512, T) f32
        z = ggml_add(gctx, z, to_f32_(gctx, m_q_oproj_b[i]));        // broadcast bias
        sum = sum ? ggml_add(gctx, sum, z) : z;
    }
    // Final rvq oproj: 512 → 768
    ggml_tensor * x = mm_(gctx, m_quant_oproj_w, sum);     // (768, T) f32
    x = ggml_add(gctx, x, to_f32_(gctx, m_quant_oproj_b));          // (768, T)

    // Transformer stages, each followed by a patch upsample.
    for (size_t s = 0; s < n_stages; ++s) {
        x = build_stage_(gctx, x, m_stages[s], pos_T[s], plan_T[s]);
        if (m_stages[s].spec.patch_after > 0) {
            x = patch_upsample_(gctx, x, m_stages[s].spec.patch_after);
        }
    }

    // After the final unpatch the channel dim is 1 and time is n_samples.
    // View as 1D.
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
    for (int i = 0; i < n_vq; ++i) {
        std::vector<int32_t> col(T_audio);
        for (int t = 0; t < T_audio; ++t) col[size_t(t)] = codes[i * T_audio + t];
        ggml_backend_tensor_set(codes_in[i], col.data(), 0,
                                size_t(T_audio) * sizeof(int32_t));
    }
    for (size_t s = 0; s < n_stages; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0,
                                p.size() * sizeof(int32_t));
        fill_attn_plan_(plan_T[s]);
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

    // Encode only as many codebooks as the *model* consumes. RVQ is greedy —
    // code i depends solely on the residual left by codes 0..i-1 — so the first
    // n_vq codes are bit-identical whether or not the deeper ones are computed.
    // MOSS-TTS-Delay wants all 32; MOSS-TTS-Local wants 12.
    const int n_vq = std::min(m_owner.dims().n_vq, CODEC_NUM_VQ);
    if (n_vq < 1) {
        throw std::runtime_error("CodecGraphs::encode: model reports n_vq=" +
                                 std::to_string(m_owner.dims().n_vq));
    }

    // Pad the waveform to a whole number of codec frames — same convention as
    // MossAudioTokenizerModel._encode_frame. `waveform` is interleaved across
    // channels, so the hop is measured in interleaved elements too.
    const int64_t hop = m_owner.dims().samples_per_frame();
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

    // Per-stage RoPE positions, one per encoder transformer stage. T shrinks by
    // the previous stage's patch factor.
    const size_t n_stages = m_enc_stages.size();
    std::vector<int>           T_at(n_stages);
    std::vector<ggml_tensor *> pos_T(n_stages, nullptr);
    std::vector<AttnPlan> plan_T(n_stages);
    for (size_t s = 0; s < n_stages; ++s) {
        T_at[s] = (s == 0) ? int(T_padded / CODEC_PRE_PATCH)
                           : T_at[s - 1] / m_enc_stages[s - 1].spec.patch_after;
        pos_T[s] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_at[s]);
        ggml_set_name(pos_T[s], ("enc_pos_" + std::to_string(s)).c_str());
        ggml_set_input(pos_T[s]);
        plan_T[s] = make_attn_plan_(gctx, T_at[s], m_enc_stages[s].spec.context);
    }

    // Initial patch=240 downsample: (1, T_padded) → (240, T_padded/240)
    ggml_tensor * x = patch_downsample_(gctx, wav, CODEC_PRE_PATCH);

    // Encoder stages, each followed by a patch downsample (except the last).
    for (size_t s = 0; s < n_stages; ++s) {
        x = build_stage_(gctx, x, m_enc_stages[s], pos_T[s], plan_T[s]);
        if (m_enc_stages[s].spec.patch_after > 0) {
            x = patch_downsample_(gctx, x, m_enc_stages[s].spec.patch_after);
        }
    }

    // ── Quantizer: input_proj → 32-step residual LFQ encoding ─────────────
    // x: (768, T_audio).
    ggml_tensor * residual = mm_(gctx, m_quant_iproj_w, x);          // (512, T)
    residual = ggml_add(gctx, residual, to_f32_(gctx, m_quant_iproj_b));      // (512, T)

    std::array<ggml_tensor *, CODEC_NUM_VQ> indices {};
    for (int i = 0; i < n_vq; ++i) {
        // z_e = q[i].iproj(residual)
        ggml_tensor * z_e = mm_(gctx, m_q_iproj_w[i], residual);     // (8, T)
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
        z_q = mm_(gctx, m_q_oproj_w[i], z_q);                        // (512, T)
        z_q = ggml_add(gctx, z_q, to_f32_(gctx, m_q_oproj_b[i]));             // (512, T)
        residual = ggml_sub(gctx, residual, z_q);                              // (512, T)
    }

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, 65536, false);
    for (int i = 0; i < n_vq; ++i) ggml_build_forward_expand(graph, indices[i]);

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
    for (size_t s = 0; s < n_stages; ++s) {
        std::vector<int32_t> p;
        p.resize(size_t(T_at[s]));
        for (int t = 0; t < T_at[s]; ++t) p[size_t(t)] = t;
        ggml_backend_tensor_set(pos_T[s], p.data(), 0, p.size() * sizeof(int32_t));
        fill_attn_plan_(plan_T[s]);
    }

    if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("CodecGraphs::encode: graph_compute failed");
    }

    // ── Read indices back: assemble (n_vq, T_audio) row-major i32 ─────────
    std::vector<int32_t> out;
    out.assign(size_t(n_vq) * size_t(T_audio), 0);
    for (int i = 0; i < n_vq; ++i) {
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
    n_vq_out = std::min(model.dims().n_vq, CODEC_NUM_VQ);
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
