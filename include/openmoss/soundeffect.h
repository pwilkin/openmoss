// SPDX-License-Identifier: Apache-2.0
//
// MOSS-SoundEffect-v2.0: text → sound effect, by flow matching.
//
// Unlike the two TTS families this model is not autoregressive and emits no
// codes. A Wan-style Diffusion Transformer predicts a velocity field in the
// latent space of a continuous DAC VAE; a Euler solver integrates it from pure
// noise down to a clean latent, which the VAE decoder turns into a waveform.
//
//   prompt ──► Qwen3-1.7B ──► context (512, 2048)
//                                 │
//   noise ──► ┌──────────────── DiT ◄─── timestep ───┐
//             │  30 blocks, dim 1536, 12 heads       │  × num_inference_steps
//             └──► velocity ──► Euler step ──────────┘
//                                 │
//                            latent (128, f) ──► DAC decoder ──► waveform
//
// Text conditioning is fixed-width: the prompt is right-padded to 512 slots and
// the rows past its real length are zeroed. Cross-attention attends to all 512
// slots unmasked, and the context never changes across solver steps — so every
// block's cross-attention K/V is computed once, up front, and reused.
//
// Duration is *textual*: the prompt carries a " duration: <X>s" suffix and the
// DiT always denoises a full `max_seconds` of latent regardless. The waveform is
// cropped afterwards.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace openmoss {

class Model;

// ───────────────────────────────────────────────────────────────────────────
// DiT
// ───────────────────────────────────────────────────────────────────────────

// Per-block cross-attention K/V for one fixed text conditioning tensor, held on
// the aux backend. Building one costs a 512-token pass over 30 blocks; reusing
// it saves that on every solver step.
class DiTContext;

// DiTContext's layout is private to dit.cpp because it is made of raw GGML
// handles. An out-of-line deleter is what lets callers still hold one by value.
struct DiTContextDeleter { void operator()(DiTContext *) const; };
using DiTContextPtr = std::unique_ptr<DiTContext, DiTContextDeleter>;

// Intermediate tensors to copy back to the host, for probes and tests. A null
// pointer means "don't tap"; a non-null one must have room for the shape noted.
// Tapping forces the tensor to be a graph output, which can inhibit reuse of its
// buffer, so leave these null on the hot path.
struct DiTTaps {
    float * t         = nullptr;   // (dim,)             time embedding
    float * t_mod     = nullptr;   // (dim, 6)           6-way modulation
    float * x_patched = nullptr;   // (dim, n_frames)    after patchify
    // All three of these describe block `blk_index`: the self-attention branch's
    // output before its gate, the cross-attention branch's output, and the
    // block's result. Having the branches separately is what tells a bad RoPE
    // from a bad modulation slot.
    float * blk_sa    = nullptr;   // (dim, n_frames)
    float * blk_ca    = nullptr;   // (dim, n_frames)
    float * blk_out   = nullptr;   // (dim, n_frames)
    int     blk_index = -1;
};

// The DiT compute graphs. Constructed lazily off Model; one per Model.
class DiTGraph {
public:
    explicit DiTGraph(Model & owner);
    ~DiTGraph();

    DiTGraph(const DiTGraph &)             = delete;
    DiTGraph & operator=(const DiTGraph &) = delete;

    // Encode a text-encoder hidden state into per-block cross-attention K/V.
    //   context: (text_dim, n_text) float32, *channel-innermost* — i.e. the
    //            transpose of PyTorch's [n_text, text_dim] memory order.
    //   ctx_emb_tap: optional (dim, n_text) copy of the projected context.
    DiTContextPtr encode_context(const float * context,
                                 int32_t       n_text,
                                 float *       ctx_emb_tap = nullptr);

    // One velocity prediction.
    //   latent:   (in_dim, n_frames) float32, channel-innermost
    //   timestep: the scheduler's timestep, in [0, sched_train_steps]
    //   out:      (out_dim, n_frames) float32, channel-innermost
    void forward(const float *      latent,
                 int64_t            n_frames,
                 float              timestep,
                 const DiTContext & ctx,
                 float *            out,
                 const DiTTaps &    taps = {});

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace openmoss
