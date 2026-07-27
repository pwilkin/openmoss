// SPDX-License-Identifier: Apache-2.0
//
// DAC VAE decoder for MOSS-SoundEffect-v2.0: latent (128, T) → mono waveform.
//
//   post_quant   Conv1d(128 → 128, k=1)            plain, not weight-normed
//   dec.in       Conv1d(128 → 2048, k=7, pad=3)
//   dec.{1..5}   Snake → ConvTranspose1d → 3x ResidualUnit(dilation 1, 3, 9)
//   dec.out      Snake → Conv1d(64 → 1, k=7, pad=3) → tanh
//
// ResidualUnit(dim, d) is Snake → Conv1d(k=7, dilation=d, pad=3d) → Snake →
// Conv1d(k=1), added to its input. The pad is exactly what makes the unit
// length-preserving, so the reference's `x[..., pad:-pad]` crop is always a
// no-op here.
//
// Snake, per channel: y = x + sin(a*x)^2 / (a + 1e-9). The sidecar ships both
// `alpha` and a precomputed `inv` = 1/(a + 1e-9), which must stay f32: three of
// the decoder's alphas are ~8e-6, so their reciprocals reach 1.2e5 and would
// round to inf in f16, poisoning the waveform with no error anywhere. Emitting
// the ops as exactly MUL → SIN → SQR → MUL → ADD also lets the Vulkan backend
// collapse them into its fused Snake kernel.
//
// Layout: everything here is *time-innermost*, ne = (time, channels) — the
// opposite of the DiT — because that is what ggml_conv_1d and ggml_im2col
// expect, and what the fused Snake kernel wants (broadcast operand shaped
// (1, C)). decode() transposes the incoming latent once.
//
// Two ggml details drive the implementation:
//
//   * ggml_conv_transpose_1d asserts p0 == 0 and d0 == 1, and Vulkan only has an
//     f32 × f32 pipeline for it. So the transposed convs run unpadded and the
//     padding is applied by cropping afterwards: ggml yields L = (T-1)s + k
//     where PyTorch yields L = (T-1)s - 2p + k + op, so p comes off the left and
//     p - op off the right.
//
//   * ggml_conv_1d hardcodes GGML_TYPE_F16 for its im2col buffer, which both
//     doubles as a precision floor on the activations and, at full length, needs
//     over a gigabyte. conv1d_() below does the im2col in f32 by hand. That
//     needs f32 kernels, which are materialised once at construction.

#include "openmoss/soundeffect.h"
#include "openmoss/model.h"

#include "aux_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace openmoss {

namespace {

constexpr size_t VAE_GRAPH_NODES = 8192;

// Latent frames per graph, before overlap. Peak temporaries land around half a
// gigabyte at this width; the whole 30 s latent at once would need tens of GB.
constexpr int32_t VAE_CHUNK_FRAMES = 128;

// Overlap added to each side of a chunk and then trimmed off. The decoder's
// half-receptive field works out to ~12 latent frames (3 from dec.in, then the
// three dilated k=7 units per block contributing 3+9+27 at that block's rate,
// divided down by each stride). Measured: 12 already makes chunked decoding
// bit-identical to single-shot, and 2 does not (1.5e-2 relative), so 32 is
// ~2.7x margin. It costs almost nothing, since it only widens the chunk.
constexpr int64_t VAE_OVERLAP_FRAMES = 32;

ggml_tensor * f32_(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

// A per-channel vector shaped for broadcasting against (time, channels).
ggml_tensor * per_channel(ggml_context * ctx, ggml_tensor * v) {
    return ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));
}

// Snake: y = x + inv * sin(alpha * x)^2, with alpha and inv per channel.
//
// The five ops must stay adjacent and in this order — Vulkan pattern-matches
// MUL, SIN, SQR, MUL, ADD and needs the ADD's other operand to be x itself.
ggml_tensor * snake_(ggml_context * ctx, ggml_tensor * x,
                     ggml_tensor * alpha, ggml_tensor * inv) {
    ggml_tensor * s = ggml_mul(ctx, x, per_channel(ctx, alpha));
    s = ggml_sin(ctx, s);
    s = ggml_sqr(ctx, s);
    s = ggml_mul(ctx, s, per_channel(ctx, inv));
    return ggml_add(ctx, x, s);
}

// Conv1d with an f32 im2col. `w` is (K, IC, OC) f32, `x` is (T, IC) f32;
// returns (OL, OC).
ggml_tensor * conv1d_(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                      ggml_tensor * x, int stride, int pad, int dilation) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, stride, 0, pad, 0, dilation, 0,
                                    /*is_2D=*/false, GGML_TYPE_F32);   // (IC*K, OL, 1)
    ggml_tensor * y = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));        // (OL, OC)
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);   // see linear() in dit.cpp
    y = ggml_reshape_2d(ctx, y, col->ne[1], w->ne[2]);
    if (b) y = ggml_add(ctx, y, per_channel(ctx, f32_(ctx, b)));
    return y;
}

// ConvTranspose1d run unpadded, then cropped to PyTorch's output window.
// `w` is (K, OC, IC) f32, `x` is (T, IC) f32; returns (L_torch, OC).
ggml_tensor * conv_transpose1d_(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                ggml_tensor * x, int stride, int pad, int out_pad) {
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, /*p0=*/0, /*d0=*/1);
    const int64_t k       = w->ne[0];
    const int64_t oc      = w->ne[1];
    const int64_t l_ggml  = (x->ne[0] - 1) * stride + k;
    const int64_t l_torch = l_ggml - pad - (pad - out_pad);
    if (l_torch <= 0) {
        throw std::runtime_error("DacDecoder: transposed conv chunk is shorter than its kernel");
    }
    y = ggml_cont(ctx, ggml_view_2d(ctx, y, l_torch, oc, y->nb[1],
                                    size_t(pad) * y->nb[0]));
    if (b) y = ggml_add(ctx, y, per_channel(ctx, f32_(ctx, b)));
    return y;
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// DacDecoder::Impl
// ───────────────────────────────────────────────────────────────────────────

struct DacDecoder::Impl {
    explicit Impl(Model & owner_) : owner(owner_) {}

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (w_buf)  ggml_backend_buffer_free(w_buf);
        if (w_ctx)  ggml_free(w_ctx);
    }

    Model & owner;

    struct Res {
        ggml_tensor *s1_a, *s1_i, *c1_w, *c1_b;
        ggml_tensor *s2_a, *s2_i, *c2_w, *c2_b;
        int dilation = 1;
    };
    struct Block {
        ggml_tensor *snake_a, *snake_i;
        ggml_tensor *up_w, *up_b;
        int stride = 1, pad = 0, out_pad = 0;
        Res res[3];
    };

    ggml_tensor *pq_w = nullptr, *pq_b = nullptr;
    ggml_tensor *in_w = nullptr, *in_b = nullptr;
    std::vector<Block> blocks;
    ggml_tensor *out_a = nullptr, *out_i = nullptr;
    ggml_tensor *out_w = nullptr, *out_b = nullptr;

    int32_t latent_dim   = 0;
    int64_t hop          = 1;
    int32_t chunk_frames = VAE_CHUNK_FRAMES;

    // f32 copies of every convolution kernel, materialised once. Needed twice
    // over: Vulkan's conv_transpose_1d is f32-only, and an f32 im2col makes
    // ggml_mul_mat demand an f32 right-hand side.
    ggml_context        * w_ctx = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    ggml_gallocr_t        galloc = nullptr;

    ggml_tensor * get(const std::string & n) const {
        auto * aux = owner.aux();
        auto it = aux->tensors.find(n);
        if (it == aux->tensors.end()) {
            throw std::runtime_error("DacDecoder: missing tensor " + n);
        }
        return it->second;
    }

    // Where the tapped tensors landed, so decode() can read them back. A tap
    // that was not requested stays null and is never read: only tensors marked
    // as graph outputs survive the allocator's buffer reuse.
    struct Tapped {
        ggml_tensor *post_quant = nullptr, *dec_in = nullptr;
        ggml_tensor *blk_up = nullptr, *blk_res0 = nullptr, *blk_out = nullptr;
    };

    // Build the (T, C) → waveform subgraph for one chunk.
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * z,
                        const DacTaps & taps, Tapped & out) const {
        auto tap = [](ggml_tensor * t, float * want, ggml_tensor ** slot) {
            if (want) { ggml_set_output(t); *slot = t; }
        };

        ggml_tensor * x = conv1d_(ctx, pq_w, pq_b, z, 1, 0, 1);
        tap(x, taps.post_quant, &out.post_quant);
        x = conv1d_(ctx, in_w, in_b, x, 1, /*pad=*/int(in_w->ne[0] / 2), 1);
        tap(x, taps.dec_in, &out.dec_in);

        for (size_t bi = 0; bi < blocks.size(); ++bi) {
            const Block & B = blocks[bi];
            const bool here = int(bi) == taps.blk_index;

            x = snake_(ctx, x, B.snake_a, B.snake_i);
            x = conv_transpose1d_(ctx, B.up_w, B.up_b, x, B.stride, B.pad, B.out_pad);
            if (here) tap(x, taps.blk_up, &out.blk_up);

            for (int r = 0; r < 3; ++r) {
                const Res & R = B.res[r];
                ggml_tensor * y = snake_(ctx, x, R.s1_a, R.s1_i);
                y = conv1d_(ctx, R.c1_w, R.c1_b, y, 1,
                            /*pad=*/3 * R.dilation, R.dilation);
                y = snake_(ctx, y, R.s2_a, R.s2_i);
                y = conv1d_(ctx, R.c2_w, R.c2_b, y, 1, 0, 1);
                x = ggml_add(ctx, x, y);
                if (here && r == 0) tap(x, taps.blk_res0, &out.blk_res0);
            }
            if (here) tap(x, taps.blk_out, &out.blk_out);
        }

        x = snake_(ctx, x, out_a, out_i);
        x = conv1d_(ctx, out_w, out_b, x, 1, /*pad=*/int(out_w->ne[0] / 2), 1);
        return ggml_tanh(ctx, x);
    }
};

// ───────────────────────────────────────────────────────────────────────────
// Construction
// ───────────────────────────────────────────────────────────────────────────

DacDecoder::DacDecoder(Model & owner) : m_impl(std::make_unique<Impl>(owner)) {
    Impl & I = *m_impl;
    const auto & d = owner.dims();

    if (d.arch != Arch::SoundEffect) {
        throw std::runtime_error("DacDecoder: model is not moss_soundeffect");
    }
    auto * aux = owner.aux();
    if (!aux || !aux->backend) {
        throw std::runtime_error("DacDecoder: aux backend not initialised");
    }
    if (d.vae_decoder_rates.empty()) {
        throw std::runtime_error("DacDecoder: the sidecar carries no moss.vae.decoder_rates");
    }
    I.latent_dim = d.vae_latent_dim > 0 ? d.vae_latent_dim : d.dit_out_dim;

    I.pq_w = I.get("moss.vae.post_quant.weight");
    I.pq_b = I.get("moss.vae.post_quant.bias");
    I.in_w = I.get("moss.vae.dec.in.weight");
    I.in_b = I.get("moss.vae.dec.in.bias");

    I.hop = 1;
    I.blocks.resize(d.vae_decoder_rates.size());
    for (size_t i = 0; i < I.blocks.size(); ++i) {
        const std::string b = "moss.vae.dec." + std::to_string(i + 1) + ".";
        Impl::Block & B = I.blocks[i];
        B.snake_a = I.get(b + "snake.alpha");
        B.snake_i = I.get(b + "snake.inv");
        B.up_w    = I.get(b + "up.weight");
        B.up_b    = I.get(b + "up.bias");

        // Upstream: kernel = 2*stride, padding = ceil(stride/2),
        // output_padding = stride % 2.
        B.stride  = d.vae_decoder_rates[i];
        B.pad     = (B.stride + 1) / 2;
        B.out_pad = B.stride % 2;
        if (B.up_w->ne[0] != 2 * B.stride) {
            throw std::runtime_error(
                "DacDecoder: " + b + "up.weight has kernel " + std::to_string(B.up_w->ne[0])
                + " but moss.vae.decoder_rates says stride " + std::to_string(B.stride)
                + " (expected kernel " + std::to_string(2 * B.stride) + ")");
        }
        I.hop *= B.stride;

        static const int DILATIONS[3] = { 1, 3, 9 };
        for (int r = 0; r < 3; ++r) {
            const std::string rb = b + "res." + std::to_string(r) + ".";
            Impl::Res & R = B.res[r];
            R.dilation = DILATIONS[r];
            R.s1_a = I.get(rb + "snake1.alpha");
            R.s1_i = I.get(rb + "snake1.inv");
            R.c1_w = I.get(rb + "conv1.weight");
            R.c1_b = I.get(rb + "conv1.bias");
            R.s2_a = I.get(rb + "snake2.alpha");
            R.s2_i = I.get(rb + "snake2.inv");
            R.c2_w = I.get(rb + "conv2.weight");
            R.c2_b = I.get(rb + "conv2.bias");
        }
    }
    if (d.vae_hop > 0 && I.hop != d.vae_hop) {
        throw std::runtime_error("DacDecoder: decoder_rates multiply to "
                                 + std::to_string(I.hop) + " but moss.vae.hop says "
                                 + std::to_string(d.vae_hop));
    }

    I.out_a = I.get("moss.vae.dec.out.snake.alpha");
    I.out_i = I.get("moss.vae.dec.out.snake.inv");
    I.out_w = I.get("moss.vae.dec.out.weight");
    I.out_b = I.get("moss.vae.dec.out.bias");

    // ── Materialise f32 kernels ────────────────────────────────────────────
    //
    // Done on the backend with a cast+copy graph rather than on the host, so no
    // f16 bit-twiddling and no round trip through system memory.
    {
        std::vector<ggml_tensor **> kernels = { &I.pq_w, &I.in_w, &I.out_w };
        for (auto & B : I.blocks) {
            kernels.push_back(&B.up_w);
            for (auto & R : B.res) { kernels.push_back(&R.c1_w); kernels.push_back(&R.c2_w); }
        }

        std::vector<std::pair<ggml_tensor **, ggml_tensor *>> todo;   // (slot, source)
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (kernels.size() + 8);
        ip.no_alloc = true;
        I.w_ctx = ggml_init(ip);
        if (!I.w_ctx) throw std::runtime_error("DacDecoder: ggml_init for f32 kernels failed");

        for (ggml_tensor ** slot : kernels) {
            if ((*slot)->type == GGML_TYPE_F32) continue;   // already f32 sidecar
            ggml_tensor * src = *slot;
            ggml_tensor * dst = ggml_new_tensor_4d(I.w_ctx, GGML_TYPE_F32,
                                                   src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
            ggml_set_name(dst, src->name);
            todo.emplace_back(slot, src);
            *slot = dst;
        }

        if (!todo.empty()) {
            I.w_buf = ggml_backend_alloc_ctx_tensors(I.w_ctx, aux->backend);
            if (!I.w_buf) throw std::runtime_error("DacDecoder: f32 kernel buffer alloc failed");

            ggml_init_params gp{};
            gp.mem_size = ggml_tensor_overhead() * (4 * todo.size() + 16)
                        + ggml_graph_overhead_custom(VAE_GRAPH_NODES, false);
            gp.no_alloc = true;
            ggml_context * gctx = ggml_init(gp);
            if (!gctx) throw std::runtime_error("DacDecoder: cast graph ggml_init failed");

            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
            ggml_cgraph * g = ggml_new_graph_custom(gctx, VAE_GRAPH_NODES, false);
            for (auto & [slot, src] : todo) {
                ggml_build_forward_expand(g, ggml_cpy(gctx, src, *slot));
            }
            const bool ok = ga && ggml_gallocr_alloc_graph(ga, g)
                         && ggml_backend_graph_compute(aux->backend, g) == GGML_STATUS_SUCCESS;
            if (ga) ggml_gallocr_free(ga);
            ggml_free(gctx);
            if (!ok) throw std::runtime_error("DacDecoder: f32 kernel materialisation failed");
        }
    }

    I.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
    if (!I.galloc) throw std::runtime_error("DacDecoder: gallocr_new failed");
}

DacDecoder::~DacDecoder() = default;

void DacDecoder::set_chunk_frames(int32_t n) {
    if (n <= 0) throw std::runtime_error("DacDecoder::set_chunk_frames: must be > 0");
    m_impl->chunk_frames = n;
}

// ───────────────────────────────────────────────────────────────────────────
// decode
// ───────────────────────────────────────────────────────────────────────────

std::vector<float> DacDecoder::decode(const float * latent, int64_t n_frames,
                                      const DacTaps & taps) {
    Impl & I = *m_impl;
    auto * aux = I.owner.aux();
    if (n_frames <= 0) return {};

    const bool tapped = taps.post_quant || taps.dec_in
                     || taps.blk_up || taps.blk_res0 || taps.blk_out;
    if (tapped && n_frames > I.chunk_frames) {
        // Chunk boundaries make intermediates ambiguous: each chunk's overlap is
        // trimmed at waveform rate, not at the rate of whatever seam is tapped.
        throw std::runtime_error(
            "DacDecoder::decode: taps require a single chunk; raise set_chunk_frames "
            "to at least " + std::to_string(n_frames));
    }

    const int64_t C   = I.latent_dim;
    const int64_t hop = I.hop;
    std::vector<float> wav(size_t(n_frames * hop));

    for (int64_t start = 0; start < n_frames; start += I.chunk_frames) {
        const int64_t end = std::min<int64_t>(start + I.chunk_frames, n_frames);
        // Widen by the overlap, clamped: at the true start and end of the
        // sequence the edge *is* the boundary the reference pads against, so
        // trimming there would be wrong.
        const int64_t ext0 = std::max<int64_t>(0, start - VAE_OVERLAP_FRAMES);
        const int64_t ext1 = std::min<int64_t>(n_frames, end + VAE_OVERLAP_FRAMES);
        const int64_t ext_len = ext1 - ext0;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 4096
                    + ggml_graph_overhead_custom(VAE_GRAPH_NODES, false);
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        if (!gctx) throw std::runtime_error("DacDecoder::decode: ggml_init failed");

        // Time-innermost, transposed from the DiT's channel-innermost latent.
        ggml_tensor * z = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ext_len, C);
        ggml_set_name(z, "latent");
        ggml_set_input(z);

        Impl::Tapped tapped_at;
        ggml_tensor * out = I.build(gctx, z, taps, tapped_at);
        ggml_set_name(out, "audio");
        ggml_set_output(out);

        ggml_cgraph * graph = ggml_new_graph_custom(gctx, VAE_GRAPH_NODES, false);
        ggml_build_forward_expand(graph, out);

        // A tapped graph needs its own allocator: ggml_gallocr_needs_realloc
        // ignores GGML_TENSOR_FLAG_OUTPUT, so it would happily reuse an
        // untapped run's plan and the taps would read overwritten buffers.
        ggml_gallocr_t galloc = I.galloc;
        ggml_gallocr_t tap_galloc = nullptr;
        if (tapped) {
            tap_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
            galloc = tap_galloc;
        }
        struct GallocGuard {
            ggml_gallocr_t g;
            ~GallocGuard() { if (g) ggml_gallocr_free(g); }
        } tap_guard{tap_galloc};

        if (!galloc || !ggml_gallocr_alloc_graph(galloc, graph)) {
            ggml_free(gctx);
            throw std::runtime_error("DacDecoder::decode: gallocr_alloc_graph failed");
        }

        {
            std::vector<float> zt(size_t(ext_len * C), 0.0f);
            for (int64_t t = 0; t < ext_len; ++t)
                for (int64_t c = 0; c < C; ++c)
                    zt[size_t(c * ext_len + t)] = latent[size_t((ext0 + t) * C + c)];
            ggml_backend_tensor_set(z, zt.data(), 0, zt.size() * sizeof(float));
        }

        if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_free(gctx);
            throw std::runtime_error("DacDecoder::decode: graph_compute failed");
        }

        if (out->ne[0] != ext_len * hop) {
            const int64_t got = out->ne[0];
            ggml_free(gctx);
            throw std::runtime_error("DacDecoder::decode: chunk produced "
                                     + std::to_string(got) + " samples, expected "
                                     + std::to_string(ext_len * hop));
        }
        ggml_backend_tensor_get(out, wav.data() + size_t(start * hop),
                                size_t((start - ext0) * hop) * sizeof(float),
                                size_t((end - start) * hop) * sizeof(float));

        auto fetch = [](ggml_tensor * src, float * dst) {
            if (src && dst) ggml_backend_tensor_get(src, dst, 0, ggml_nbytes(src));
        };
        fetch(tapped_at.post_quant, taps.post_quant);
        fetch(tapped_at.dec_in,     taps.dec_in);
        fetch(tapped_at.blk_up,     taps.blk_up);
        fetch(tapped_at.blk_res0,   taps.blk_res0);
        fetch(tapped_at.blk_out,    taps.blk_out);

        ggml_free(gctx);
    }

    return wav;
}

} // namespace openmoss
