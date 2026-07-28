// SPDX-License-Identifier: Apache-2.0
//
// WanAudioModel — the Diffusion Transformer behind MOSS-SoundEffect-v2.0.
//
// 30 blocks, dim 1536, 12 heads (head_dim 128), FFN 8960, eps 1e-6 everywhere.
// Per block, with the six modulation vectors sliced out of a shared sum:
//
//   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//       = (blk.mod[6, dim] + t_mod[6, dim]).chunk(6)
//
//   x = x + gate_msa * self_attn ( norm1(x) * (1 + scale_msa) + shift_msa )
//   x = x +            cross_attn( norm3(x), context )      ← no gate here
//   x = x + gate_mlp * ffn       ( norm2(x) * (1 + scale_mlp) + shift_mlp )
//
// Things that are easy to get wrong, all checked against the reference:
//
//   * norm1 and norm2 are LayerNorm(elementwise_affine=False) — they have no
//     parameters and are genuinely absent from the checkpoint. Only norm3 is
//     affine, and the diffusers export stores it under the name `norm2`; the
//     converter already renames it, so here it really is norm3.
//
//   * norm_q / norm_k are RMSNorm over the *whole* 1536 axis, not per-head.
//     They therefore run before the reshape into heads. v is neither normed
//     nor rotated.
//
//   * RoPE is the interleaved-pair variant (GGML_ROPE_TYPE_NORMAL), theta
//     10000, over all 128 head dims. The reference splits `freqs_cis` into
//     three chunks and concatenates them straight back, which for vae_type
//     "dac" is an exact identity — so this is a plain 1-D RoPE.
//
//   * Cross-attention has no RoPE and no mask: all 512 text slots are attended,
//     padding included. The padding rows were zeroed by the text encoder, which
//     is what makes that harmless.
//
//   * The FFN activation is GELU with approximate='tanh'. ggml_gelu *is* the
//     tanh approximation, so ggml_gelu_erf would be wrong here.
//
//   * The final head takes the plain time embedding `t`, not `t_mod`, with its
//     own 2-way modulation: shift = index 0, scale = index 1.
//
// Layout convention: every activation is channel-innermost, ne = (channels,
// frames). PyTorch's latents are [B, C, T] (frame-innermost), so the host
// transposes at the boundary — see the doc comment on DiTGraph::forward.
//
// patch_size is [1], so patchify is a per-frame affine map (a kernel-1 Conv1d)
// and unpatchify is a no-op in this layout.

#include "openmoss/soundeffect.h"
#include "openmoss/model.h"

#include "aux_internal.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace openmoss {

namespace {

// The DiT graph is ~45 nodes per block plus the embedding and head paths.
constexpr size_t DIT_GRAPH_NODES = 16384;

// f16 constants must be promoted before elementwise ops: several backends only
// implement bin_bcast for (F32, F32, F32). Same reason as codec.cpp's to_f32_.
ggml_tensor * f32_(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// y = W x + b, with W stored as ne = (in, out) and b optional.
//
// The f32 accumulator is not optional. Vulkan picks its f16-accumulate matmul
// pipeline whenever prec is GGML_PREC_DEFAULT, which over a 1536- or 2048-long
// dot product costs ~5e-3 relative — twenty times the error f16 *weights* alone
// contribute, and enough to swamp the model's real per-frame signal once the
// head's LayerNorm amplifies it. Matrix-vector products take a different,
// f32-accumulating path, so the damage is invisible until a seam with more than
// one column is compared.
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                     ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return b ? ggml_add(ctx, y, f32_(ctx, b)) : y;
}

// LayerNorm(elementwise_affine=False) — normalisation and nothing else.
ggml_tensor * layer_norm_plain(ggml_context * ctx, ggml_tensor * x, float eps) {
    return ggml_norm(ctx, x, eps);
}

ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, f32_(ctx, w));
    return ggml_add(ctx, y, f32_(ctx, b));
}

ggml_tensor * rms_norm_w(ggml_context * ctx, ggml_tensor * x,
                         ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), f32_(ctx, w));
}

// The `modulate()` of the reference: x * (1 + scale) + shift.
//
// `ones` is a persistent (dim, 1) tensor of 1.0f rather than an in-graph
// constant so the expression stays literally "1 + scale" — writing it as
// x*scale + x + shift would reassociate and drift from the reference.
ggml_tensor * modulate(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * shift, ggml_tensor * scale,
                       ggml_tensor * ones) {
    ggml_tensor * s = ggml_add(ctx, scale, ones);
    return ggml_add(ctx, ggml_mul(ctx, x, s), shift);
}

// Column `idx` of a (dim, n) f32 tensor, as a (dim, 1) tensor that broadcasts
// against (dim, frames). The slice is a contiguous run, so no cont is needed.
ggml_tensor * column(ggml_context * ctx, ggml_tensor * t, int idx) {
    return ggml_view_2d(ctx, t, t->ne[0], 1, t->nb[1], size_t(idx) * t->nb[1]);
}

// Scaled dot-product attention over pre-split heads.
//   q: (head_dim, n_q,  n_heads) f32
//   k: (head_dim, n_kv, n_heads) f16
//   v: (head_dim, n_kv, n_heads) f16
// Returns (head_dim * n_heads, n_q) — heads already flattened back into `dim`.
//
// No mask in either attention: self-attention over the latent is fully
// bidirectional (this is a diffusion model, not a causal LM), and
// cross-attention deliberately attends to the zeroed padding slots too.
ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                        ggml_tensor * v, int dim, int head_dim, int64_t n_q) {
    const float scale = 1.0f / std::sqrt(float(head_dim));
    ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, scale,
                                          /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
    ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    // (head_dim, n_heads, n_q) → (dim, n_q)
    return ggml_reshape_2d(ctx, a, dim, n_q);
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// DiTContext — per-block cross-attention K/V for one fixed conditioning
// ───────────────────────────────────────────────────────────────────────────

class DiTContext {
public:
    ~DiTContext() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }

    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k;   // per block, (head_dim, n_text, n_heads) f16
    std::vector<ggml_tensor *> v;
    int32_t n_text = 0;
};

void DiTContextDeleter::operator()(DiTContext * p) const { delete p; }

// ───────────────────────────────────────────────────────────────────────────
// DiTGraph::Impl
// ───────────────────────────────────────────────────────────────────────────

struct DiTGraph::Impl {
    explicit Impl(Model & owner_) : owner(owner_) {}

    ~Impl() {
        if (galloc)    ggml_gallocr_free(galloc);
        if (const_buf) ggml_backend_buffer_free(const_buf);
        if (const_ctx) ggml_free(const_ctx);
    }

    Model & owner;

    int   dim = 0, n_layers = 0, n_heads = 0, head_dim = 0, ffn_dim = 0;
    int   in_dim = 0, out_dim = 0, text_dim = 0, freq_dim = 0;
    float eps = 1e-6f, rope_base = 10000.0f;

    struct Block {
        ggml_tensor *sa_q_w, *sa_q_b, *sa_k_w, *sa_k_b, *sa_v_w, *sa_v_b;
        ggml_tensor *sa_o_w, *sa_o_b, *sa_nq, *sa_nk;
        ggml_tensor *ca_q_w, *ca_q_b, *ca_k_w, *ca_k_b, *ca_v_w, *ca_v_b;
        ggml_tensor *ca_o_w, *ca_o_b, *ca_nq, *ca_nk;
        ggml_tensor *ffn1_w, *ffn1_b, *ffn2_w, *ffn2_b;
        ggml_tensor *norm3_w, *norm3_b;
        ggml_tensor *mod;            // (dim, 6) f16
    };
    std::vector<Block> blocks;

    ggml_tensor *t_emb1_w = nullptr, *t_emb1_b = nullptr;
    ggml_tensor *t_emb2_w = nullptr, *t_emb2_b = nullptr;
    ggml_tensor *t_proj_w = nullptr, *t_proj_b = nullptr;
    ggml_tensor *txt1_w   = nullptr, *txt1_b   = nullptr;
    ggml_tensor *txt2_w   = nullptr, *txt2_b   = nullptr;
    ggml_tensor *patch_w  = nullptr, *patch_b  = nullptr;
    ggml_tensor *head_w   = nullptr, *head_b   = nullptr, *head_mod = nullptr;

    // Owned constants: the (dim, 1) vector of ones used by modulate().
    ggml_context        * const_ctx = nullptr;
    ggml_backend_buffer_t const_buf = nullptr;
    ggml_tensor         * ones      = nullptr;

    ggml_gallocr_t galloc = nullptr;

    ggml_tensor * get(const std::string & name) const {
        auto * aux = owner.aux();
        auto it = aux->tensors.find(name);
        if (it == aux->tensors.end()) {
            throw std::runtime_error("DiTGraph: missing tensor " + name);
        }
        return it->second;
    }

    // The time embedding's sinusoidal input, computed on the host in double
    // exactly as sinusoidal_embedding_1d does:
    //   sinusoid[j] = timestep * 10000^(-j / (freq_dim/2))
    //   emb         = [cos(sinusoid) ; sin(sinusoid)]     — cosines FIRST
    std::vector<float> sinusoid(float timestep) const {
        const int half = freq_dim / 2;
        std::vector<float> out(size_t(freq_dim), 0.0f);
        for (int j = 0; j < half; ++j) {
            const double f = std::pow(10000.0, -double(j) / double(half));
            const double s = double(timestep) * f;
            out[size_t(j)]        = float(std::cos(s));
            out[size_t(half + j)] = float(std::sin(s));
        }
        return out;
    }

    // Reshape a linear weight to 2-D. patch.weight arrives as a kernel-1 Conv1d,
    // ne = (1, in, out); with patch_size 1 that is exactly a linear map.
    static ggml_tensor * as_linear_w(ggml_context * ctx, ggml_tensor * w) {
        if (ggml_n_dims(w) <= 2) return w;
        return ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
    }

    // Project `ctx_emb` (dim, n_text) through one block's cross-attention K and
    // V, laid out for flash attention: (head_dim, n_text, n_heads).
    void build_cross_kv(ggml_context * gctx, const Block & B, ggml_tensor * ctx_emb,
                        int64_t n_text, ggml_tensor ** k_out, ggml_tensor ** v_out) const {
        ggml_tensor * k = rms_norm_w(gctx, linear(gctx, B.ca_k_w, B.ca_k_b, ctx_emb),
                                     B.ca_nk, eps);
        ggml_tensor * v = linear(gctx, B.ca_v_w, B.ca_v_b, ctx_emb);
        k = ggml_reshape_3d(gctx, k, head_dim, n_heads, n_text);
        v = ggml_reshape_3d(gctx, v, head_dim, n_heads, n_text);
        *k_out = ggml_cont(gctx, ggml_permute(gctx, k, 0, 2, 1, 3));
        *v_out = ggml_cont(gctx, ggml_permute(gctx, v, 0, 2, 1, 3));
    }
};

// ───────────────────────────────────────────────────────────────────────────
// Construction
// ───────────────────────────────────────────────────────────────────────────

DiTGraph::DiTGraph(Model & owner) : m_impl(std::make_unique<Impl>(owner)) {
    Impl & I = *m_impl;
    const auto & d = owner.dims();

    if (d.arch != Arch::SoundEffect) {
        throw std::runtime_error("DiTGraph: model is not moss_soundeffect");
    }
    auto * aux = owner.aux();
    if (!aux || !aux->backend) {
        throw std::runtime_error("DiTGraph: aux backend not initialised");
    }

    I.dim       = d.dit_dim;
    I.n_layers  = d.dit_n_layers;
    I.n_heads   = d.dit_n_heads;
    I.ffn_dim   = d.dit_ffn_dim;
    I.in_dim    = d.dit_in_dim;
    I.out_dim   = d.dit_out_dim;
    I.text_dim  = d.dit_text_dim;
    I.freq_dim  = d.dit_freq_dim;
    I.eps       = d.dit_eps;
    I.rope_base = d.dit_rope_base;

    if (I.dim <= 0 || I.n_heads <= 0 || I.n_layers <= 0) {
        throw std::runtime_error("DiTGraph: incomplete DiT geometry in the sidecar");
    }
    if (I.dim % I.n_heads != 0) {
        throw std::runtime_error("DiTGraph: dim " + std::to_string(I.dim) +
                                 " not divisible by n_heads " + std::to_string(I.n_heads));
    }
    I.head_dim = I.dim / I.n_heads;
    if (I.freq_dim % 2 != 0) {
        throw std::runtime_error("DiTGraph: freq_dim must be even, got "
                                 + std::to_string(I.freq_dim));
    }

    I.blocks.resize(size_t(I.n_layers));
    for (int i = 0; i < I.n_layers; ++i) {
        const std::string b = "moss.dit.blk." + std::to_string(i) + ".";
        Impl::Block & B = I.blocks[size_t(i)];
        B.sa_q_w = I.get(b + "sa.q.weight");  B.sa_q_b = I.get(b + "sa.q.bias");
        B.sa_k_w = I.get(b + "sa.k.weight");  B.sa_k_b = I.get(b + "sa.k.bias");
        B.sa_v_w = I.get(b + "sa.v.weight");  B.sa_v_b = I.get(b + "sa.v.bias");
        B.sa_o_w = I.get(b + "sa.o.weight");  B.sa_o_b = I.get(b + "sa.o.bias");
        B.sa_nq  = I.get(b + "sa.norm_q.weight");
        B.sa_nk  = I.get(b + "sa.norm_k.weight");
        B.ca_q_w = I.get(b + "ca.q.weight");  B.ca_q_b = I.get(b + "ca.q.bias");
        B.ca_k_w = I.get(b + "ca.k.weight");  B.ca_k_b = I.get(b + "ca.k.bias");
        B.ca_v_w = I.get(b + "ca.v.weight");  B.ca_v_b = I.get(b + "ca.v.bias");
        B.ca_o_w = I.get(b + "ca.o.weight");  B.ca_o_b = I.get(b + "ca.o.bias");
        B.ca_nq  = I.get(b + "ca.norm_q.weight");
        B.ca_nk  = I.get(b + "ca.norm_k.weight");
        B.ffn1_w = I.get(b + "ffn.1.weight"); B.ffn1_b = I.get(b + "ffn.1.bias");
        B.ffn2_w = I.get(b + "ffn.2.weight"); B.ffn2_b = I.get(b + "ffn.2.bias");
        B.norm3_w = I.get(b + "norm3.weight");
        B.norm3_b = I.get(b + "norm3.bias");
        B.mod     = I.get(b + "mod");
        if (ggml_nelements(B.mod) != int64_t(I.dim) * 6) {
            throw std::runtime_error("DiTGraph: " + b + "mod has "
                                     + std::to_string(ggml_nelements(B.mod))
                                     + " elements, expected " + std::to_string(I.dim * 6));
        }
    }

    I.t_emb1_w = I.get("moss.dit.t_emb.1.weight");  I.t_emb1_b = I.get("moss.dit.t_emb.1.bias");
    I.t_emb2_w = I.get("moss.dit.t_emb.2.weight");  I.t_emb2_b = I.get("moss.dit.t_emb.2.bias");
    I.t_proj_w = I.get("moss.dit.t_proj.weight");   I.t_proj_b = I.get("moss.dit.t_proj.bias");
    I.txt1_w   = I.get("moss.dit.txt_emb.1.weight");I.txt1_b   = I.get("moss.dit.txt_emb.1.bias");
    I.txt2_w   = I.get("moss.dit.txt_emb.2.weight");I.txt2_b   = I.get("moss.dit.txt_emb.2.bias");
    I.patch_w  = I.get("moss.dit.patch.weight");    I.patch_b  = I.get("moss.dit.patch.bias");
    I.head_w   = I.get("moss.dit.head.weight");     I.head_b   = I.get("moss.dit.head.bias");
    I.head_mod = I.get("moss.dit.mod");

    if (I.t_emb1_w->ne[0] != I.freq_dim) {
        throw std::runtime_error(
            "DiTGraph: t_emb.1 expects " + std::to_string(I.t_emb1_w->ne[0]) +
            " inputs but moss.dit.freq_dim is " + std::to_string(I.freq_dim));
    }
    if (ggml_nelements(I.head_mod) != int64_t(I.dim) * 2) {
        throw std::runtime_error("DiTGraph: moss.dit.mod has "
                                 + std::to_string(ggml_nelements(I.head_mod))
                                 + " elements, expected " + std::to_string(I.dim * 2));
    }

    // Owned constants.
    {
        ggml_init_params ip{};
        ip.mem_size   = ggml_tensor_overhead() * 8;
        ip.no_alloc   = true;
        I.const_ctx = ggml_init(ip);
        if (!I.const_ctx) throw std::runtime_error("DiTGraph: ggml_init for constants failed");

        I.ones = ggml_new_tensor_2d(I.const_ctx, GGML_TYPE_F32, I.dim, 1);
        ggml_set_name(I.ones, "dit.ones");

        I.const_buf = ggml_backend_alloc_ctx_tensors(I.const_ctx, aux->backend);
        if (!I.const_buf) throw std::runtime_error("DiTGraph: constant buffer alloc failed");

        std::vector<float> ones(size_t(I.dim), 1.0f);
        ggml_backend_tensor_set(I.ones, ones.data(), 0, ones.size() * sizeof(float));
    }

    I.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
    if (!I.galloc) throw std::runtime_error("DiTGraph: gallocr_new failed");
}

DiTGraph::~DiTGraph() = default;

// ───────────────────────────────────────────────────────────────────────────
// encode_context
// ───────────────────────────────────────────────────────────────────────────

DiTContextPtr DiTGraph::encode_context(const float * context,
                                       int32_t       n_text,
                                       float *       ctx_emb_tap) {
    Impl & I = *m_impl;
    auto * aux = I.owner.aux();
    if (n_text <= 0) throw std::runtime_error("DiTGraph::encode_context: n_text must be > 0");

    DiTContextPtr out(new DiTContext());
    out->n_text = n_text;

    // Destination tensors live in their own buffer and outlive the graph.
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * size_t(2 * I.n_layers + 8);
        ip.no_alloc = true;
        out->ctx = ggml_init(ip);
        if (!out->ctx) throw std::runtime_error("DiTGraph::encode_context: ggml_init failed");

        out->k.resize(size_t(I.n_layers));
        out->v.resize(size_t(I.n_layers));
        for (int i = 0; i < I.n_layers; ++i) {
            out->k[size_t(i)] = ggml_new_tensor_3d(out->ctx, GGML_TYPE_F16,
                                                   I.head_dim, n_text, I.n_heads);
            out->v[size_t(i)] = ggml_new_tensor_3d(out->ctx, GGML_TYPE_F16,
                                                   I.head_dim, n_text, I.n_heads);
            ggml_set_name(out->k[size_t(i)], ("ctx.k." + std::to_string(i)).c_str());
            ggml_set_name(out->v[size_t(i)], ("ctx.v." + std::to_string(i)).c_str());
        }
        out->buf = ggml_backend_alloc_ctx_tensors(out->ctx, aux->backend);
        if (!out->buf) {
            throw std::runtime_error("DiTGraph::encode_context: K/V buffer alloc failed");
        }
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (64 * size_t(I.n_layers) + 128)
                + ggml_graph_overhead_custom(DIT_GRAPH_NODES, false);
    ip.no_alloc = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("DiTGraph::encode_context: graph ggml_init failed");

    ggml_tensor * ctx_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, I.text_dim, n_text);
    ggml_set_name(ctx_in, "context");
    ggml_set_input(ctx_in);

    // text_embedding: Linear(text_dim → dim) → GELU(tanh) → Linear(dim → dim)
    ggml_tensor * ctx_emb = linear(gctx, I.txt1_w, I.txt1_b, ctx_in);
    ctx_emb = ggml_gelu(gctx, ctx_emb);
    ctx_emb = linear(gctx, I.txt2_w, I.txt2_b, ctx_emb);
    if (ctx_emb_tap) {
        ggml_set_name(ctx_emb, "ctx_emb");
        ggml_set_output(ctx_emb);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, DIT_GRAPH_NODES, false);
    for (int i = 0; i < I.n_layers; ++i) {
        ggml_tensor *k = nullptr, *v = nullptr;
        I.build_cross_kv(gctx, I.blocks[size_t(i)], ctx_emb, n_text, &k, &v);
        ggml_build_forward_expand(graph, ggml_cpy(gctx, k, out->k[size_t(i)]));
        ggml_build_forward_expand(graph, ggml_cpy(gctx, v, out->v[size_t(i)]));
    }
    if (ctx_emb_tap) ggml_build_forward_expand(graph, ctx_emb);

    if (!ggml_gallocr_alloc_graph(I.galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("DiTGraph::encode_context: gallocr_alloc_graph failed");
    }
    ggml_backend_tensor_set(ctx_in, context, 0,
                            size_t(I.text_dim) * size_t(n_text) * sizeof(float));

    if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("DiTGraph::encode_context: graph_compute failed");
    }
    if (ctx_emb_tap) {
        ggml_backend_tensor_get(ctx_emb, ctx_emb_tap, 0,
                                size_t(I.dim) * size_t(n_text) * sizeof(float));
    }

    ggml_free(gctx);
    return out;
}

// ───────────────────────────────────────────────────────────────────────────
// forward
// ───────────────────────────────────────────────────────────────────────────

void DiTGraph::forward(const float *      latent,
                       int64_t            n_frames,
                       float              timestep,
                       const DiTContext & ctx,
                       float *            out,
                       const DiTTaps &    taps) {
    Impl & I = *m_impl;
    auto * aux = I.owner.aux();
    if (n_frames <= 0) throw std::runtime_error("DiTGraph::forward: n_frames must be > 0");
    if (int(ctx.k.size()) != I.n_layers) {
        throw std::runtime_error("DiTGraph::forward: context was built for a different model");
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (256 * size_t(I.n_layers) + 512)
                + ggml_graph_overhead_custom(DIT_GRAPH_NODES, false);
    ip.no_alloc = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("DiTGraph::forward: ggml_init failed");

    // ── Inputs ─────────────────────────────────────────────────────────────
    ggml_tensor * x_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, I.in_dim, n_frames);
    ggml_set_name(x_in, "latent");
    ggml_set_input(x_in);

    ggml_tensor * t_sin = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, I.freq_dim, 1);
    ggml_set_name(t_sin, "t_sin");
    ggml_set_input(t_sin);

    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_frames);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    // ── Time embedding ─────────────────────────────────────────────────────
    //   t     = t_emb.2( silu( t_emb.1(sinusoid) ) )
    //   t_mod = t_proj( silu(t) ).unflatten(6, dim)
    // time_projection is Sequential(SiLU, Linear), so the SiLU comes first and
    // the stored weight is index .1.
    ggml_tensor * t = linear(gctx, I.t_emb1_w, I.t_emb1_b, t_sin);
    t = ggml_silu(gctx, t);
    t = linear(gctx, I.t_emb2_w, I.t_emb2_b, t);          // (dim, 1)
    if (taps.t) { ggml_set_name(t, "t"); ggml_set_output(t); }

    // Tap the pre-reshape tensor, not the reshape: ggml_set_output on a view
    // leaves the parent's buffer eligible for reuse, so the readback would find
    // whatever the last block wrote there. Same bytes either way — (6*dim, 1)
    // and (dim, 6) are the same memory.
    ggml_tensor * t_proj = linear(gctx, I.t_proj_w, I.t_proj_b, ggml_silu(gctx, t));
    if (taps.t_mod) { ggml_set_name(t_proj, "t_mod"); ggml_set_output(t_proj); }
    ggml_tensor * t_mod = ggml_reshape_2d(gctx, t_proj, I.dim, 6);       // (dim, 6)

    // ── Patchify: kernel-1 Conv1d = a per-frame affine map ────────────────
    ggml_tensor * x = linear(gctx, Impl::as_linear_w(gctx, I.patch_w), I.patch_b, x_in);
    ggml_tensor * x_patched = nullptr;
    if (taps.x_patched) {
        ggml_set_name(x, "x_patched");
        ggml_set_output(x);
        x_patched = x;
    }
    ggml_tensor * blk_sa = nullptr, * blk_ca = nullptr, * blk_out = nullptr;

    // ── Blocks ─────────────────────────────────────────────────────────────
    for (int i = 0; i < I.n_layers; ++i) {
        const Impl::Block & B = I.blocks[size_t(i)];

        ggml_tensor * m = ggml_add(gctx, ggml_reshape_2d(gctx, f32_(gctx, B.mod), I.dim, 6),
                                   t_mod);                          // (dim, 6)
        ggml_tensor * shift_msa = column(gctx, m, 0);
        ggml_tensor * scale_msa = column(gctx, m, 1);
        ggml_tensor * gate_msa  = column(gctx, m, 2);
        ggml_tensor * shift_mlp = column(gctx, m, 3);
        ggml_tensor * scale_mlp = column(gctx, m, 4);
        ggml_tensor * gate_mlp  = column(gctx, m, 5);

        // Self-attention, modulated in and gated out.
        {
            ggml_tensor * h = modulate(gctx, layer_norm_plain(gctx, x, I.eps),
                                       shift_msa, scale_msa, I.ones);

            ggml_tensor * q = rms_norm_w(gctx, linear(gctx, B.sa_q_w, B.sa_q_b, h), B.sa_nq, I.eps);
            ggml_tensor * k = rms_norm_w(gctx, linear(gctx, B.sa_k_w, B.sa_k_b, h), B.sa_nk, I.eps);
            ggml_tensor * v = linear(gctx, B.sa_v_w, B.sa_v_b, h);

            q = ggml_reshape_3d(gctx, q, I.head_dim, I.n_heads, n_frames);
            k = ggml_reshape_3d(gctx, k, I.head_dim, I.n_heads, n_frames);
            v = ggml_reshape_3d(gctx, v, I.head_dim, I.n_heads, n_frames);

            // Interleaved-pair RoPE over all head dims; v is left alone.
            q = ggml_rope_ext(gctx, q, pos, nullptr, I.head_dim, GGML_ROPE_TYPE_NORMAL,
                              /*n_ctx_orig=*/0, I.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(gctx, k, pos, nullptr, I.head_dim, GGML_ROPE_TYPE_NORMAL,
                              /*n_ctx_orig=*/0, I.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * qp = ggml_cont(gctx, ggml_permute(gctx, q, 0, 2, 1, 3));
            ggml_tensor * kp = ggml_cast(gctx, ggml_cont(gctx, ggml_permute(gctx, k, 0, 2, 1, 3)),
                                         GGML_TYPE_F16);
            ggml_tensor * vp = ggml_cast(gctx, ggml_cont(gctx, ggml_permute(gctx, v, 0, 2, 1, 3)),
                                         GGML_TYPE_F16);

            ggml_tensor * a = attention(gctx, qp, kp, vp, I.dim, I.head_dim, n_frames);
            a = linear(gctx, B.sa_o_w, B.sa_o_b, a);
            if (taps.blk_sa && taps.blk_index == i) { ggml_set_output(a); blk_sa = a; }
            x = ggml_add(gctx, x, ggml_mul(gctx, a, gate_msa));
        }

        // Cross-attention onto the cached text K/V. No gate, no modulation —
        // just the affine norm3 on the way in.
        {
            ggml_tensor * h = layer_norm_affine(gctx, x, B.norm3_w, B.norm3_b, I.eps);
            ggml_tensor * q = rms_norm_w(gctx, linear(gctx, B.ca_q_w, B.ca_q_b, h), B.ca_nq, I.eps);
            q = ggml_reshape_3d(gctx, q, I.head_dim, I.n_heads, n_frames);
            ggml_tensor * qp = ggml_cont(gctx, ggml_permute(gctx, q, 0, 2, 1, 3));

            ggml_tensor * a = attention(gctx, qp, ctx.k[size_t(i)], ctx.v[size_t(i)],
                                        I.dim, I.head_dim, n_frames);
            a = linear(gctx, B.ca_o_w, B.ca_o_b, a);
            if (taps.blk_ca && taps.blk_index == i) { ggml_set_output(a); blk_ca = a; }
            x = ggml_add(gctx, x, a);
        }

        // FFN, modulated in and gated out. GELU(approximate='tanh').
        {
            ggml_tensor * h = modulate(gctx, layer_norm_plain(gctx, x, I.eps),
                                       shift_mlp, scale_mlp, I.ones);
            h = linear(gctx, B.ffn1_w, B.ffn1_b, h);
            h = ggml_gelu(gctx, h);
            h = linear(gctx, B.ffn2_w, B.ffn2_b, h);
            x = ggml_add(gctx, x, ggml_mul(gctx, h, gate_mlp));
        }

        if (taps.blk_out && taps.blk_index == i) {
            ggml_set_name(x, "blk_out");
            ggml_set_output(x);
            blk_out = x;
        }
    }

    // ── Head: modulated by the plain time embedding `t`, not by t_mod ─────
    {
        ggml_tensor * hm = ggml_add(gctx, ggml_reshape_2d(gctx, f32_(gctx, I.head_mod), I.dim, 2),
                                    t);                              // (dim, 2)
        ggml_tensor * h = modulate(gctx, layer_norm_plain(gctx, x, I.eps),
                                   column(gctx, hm, 0), column(gctx, hm, 1), I.ones);
        x = linear(gctx, I.head_w, I.head_b, h);                     // (out_dim, n_frames)
    }
    ggml_set_name(x, "velocity");
    ggml_set_output(x);

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, DIT_GRAPH_NODES, false);
    ggml_build_forward_expand(graph, x);

    // Tapped runs get a private allocator. ggml_gallocr_needs_realloc compares
    // node counts, shapes and sources — but *not* GGML_TENSOR_FLAG_OUTPUT — so a
    // tapped graph and an untapped one look identical to it and the second call
    // silently reuses the first's plan. The plan that matters is the one where
    // the tapped tensor was marked an output; without this, taps read whatever
    // later nodes wrote over that buffer. Only the verification path pays for it.
    const bool tapped = taps.t || taps.t_mod || taps.x_patched
                     || taps.blk_sa || taps.blk_ca || taps.blk_out;
    ggml_gallocr_t galloc = I.galloc;
    ggml_gallocr_t tap_galloc = nullptr;
    if (tapped) {
        tap_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
        if (!tap_galloc) {
            ggml_free(gctx);
            throw std::runtime_error("DiTGraph::forward: gallocr_new failed");
        }
        galloc = tap_galloc;
    }
    struct GallocGuard {
        ggml_gallocr_t g;
        ~GallocGuard() { if (g) ggml_gallocr_free(g); }
    } tap_guard{tap_galloc};

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("DiTGraph::forward: gallocr_alloc_graph failed");
    }

    // ── Upload inputs ──────────────────────────────────────────────────────
    ggml_backend_tensor_set(x_in, latent, 0,
                            size_t(I.in_dim) * size_t(n_frames) * sizeof(float));
    {
        const std::vector<float> s = I.sinusoid(timestep);
        ggml_backend_tensor_set(t_sin, s.data(), 0, s.size() * sizeof(float));
    }
    {
        std::vector<int32_t> p(size_t(n_frames), 0);
        for (int64_t i = 0; i < n_frames; ++i) p[size_t(i)] = int32_t(i);
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("DiTGraph::forward: graph_compute failed");
    }

    ggml_backend_tensor_get(x, out, 0,
                            size_t(I.out_dim) * size_t(n_frames) * sizeof(float));

    // Only tensors that were marked as graph outputs above survive gallocr's
    // buffer reuse, so each fetch is gated on the same condition that set it.
    auto fetch = [](ggml_tensor * src, float * dst) {
        if (src && dst) ggml_backend_tensor_get(src, dst, 0, ggml_nbytes(src));
    };
    fetch(t,         taps.t);
    fetch(t_proj,    taps.t_mod);
    fetch(x_patched, taps.x_patched);
    fetch(blk_sa,    taps.blk_sa);
    fetch(blk_ca,    taps.blk_ca);
    fetch(blk_out,   taps.blk_out);

    ggml_free(gctx);
}

// ───────────────────────────────────────────────────────────────────────────
// Lazy accessors on Model
//
// Defined here rather than in model.cpp so the unique_ptr destructors see the
// complete types — the same reason Model::codec() lives in codec.cpp.
// ───────────────────────────────────────────────────────────────────────────

DiTGraph * Model::dit() {
    if (!m_dit) m_dit.reset(new DiTGraph(*this));
    return m_dit.get();
}

void DiTGraphDeleter::operator()(DiTGraph * p) const { delete p; }

} // namespace openmoss
