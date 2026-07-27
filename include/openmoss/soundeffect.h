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
#include <functional>
#include <memory>
#include <string>
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

// ───────────────────────────────────────────────────────────────────────────
// DAC VAE decoder
// ───────────────────────────────────────────────────────────────────────────

// latent → waveform. `continuous: true` upstream, so there is no quantizer and
// no codebooks: a kernel-1 post-quant conv, an input conv, five upsampling
// blocks (Snake → transposed conv → three dilated residual units) and a final
// Snake + conv + tanh. Total upsample 960, i.e. 50 latent frames per second at
// 48 kHz, mono.
//
// The decoder is one long convolution stack, so it is chunked along time with
// an overlap wider than its receptive field: the last block alone would
// otherwise need hundreds of megabytes per temporary at full length.
// Intermediates to copy back, for probes and tests. Every buffer is
// (time, channels) with time innermost — which is also how PyTorch lays out
// [batch, channels, time], so reference dumps compare byte-for-byte with no
// transpose. Taps require the whole decode to fit in one chunk.
struct DacTaps {
    float * post_quant = nullptr;   // (T, latent_dim)
    float * dec_in     = nullptr;   // (T, channels)
    // All three describe block `blk_index` (0-based): after its Snake and
    // transposed convolution, after its first residual unit, and its result.
    // The first isolates the transposed conv's crop convention, which is the
    // most error-prone part of the port.
    float * blk_up     = nullptr;
    float * blk_res0   = nullptr;
    float * blk_out    = nullptr;
    int     blk_index  = -1;
};

class DacDecoder {
public:
    explicit DacDecoder(Model & owner);
    ~DacDecoder();

    DacDecoder(const DacDecoder &)             = delete;
    DacDecoder & operator=(const DacDecoder &) = delete;

    // latent: (latent_dim, n_frames) float32, channel-innermost — the same
    //         layout DiTGraph::forward emits.
    // returns n_frames * hop mono float32 samples.
    std::vector<float> decode(const float * latent, int64_t n_frames,
                              const DacTaps & taps = {});

    // Latent frames decoded per graph. Larger is fewer graph builds and more
    // peak memory; the default is tuned for roughly half a gigabyte of
    // temporaries. Overlap is added on top and trimmed away.
    void set_chunk_frames(int32_t n);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// ───────────────────────────────────────────────────────────────────────────
// End-to-end generation
// ───────────────────────────────────────────────────────────────────────────

struct SoundEffectRequest {
    std::string prompt;

    // Requested duration. Purely textual: it is appended to the prompt and the
    // DiT always denoises `moss.max_seconds` worth of latent regardless; the
    // waveform is cropped to this afterwards. Rounded to one decimal, because
    // the suffix the model was trained on carries exactly one.
    float seconds = 10.0f;

    int   num_inference_steps = 100;
    float cfg_scale           = 4.0f;   // 1.0 disables the unconditional branch
    float sigma_shift         = 5.0f;

    // 0 is a valid seed here (the upstream default), so there is no
    // "nondeterministic" sentinel — pass a random value for that.
    uint64_t seed = 0;

    // Upstream only ever uses the empty string, which tokenizes to nothing and
    // is then hard-zeroed, making the unconditional context exactly zeros. A
    // non-empty value runs the text encoder for that branch too.
    std::string negative_prompt;

    bool append_duration_suffix = true;

    // Verification hook: when set, this replaces the seeded noise. Layout is
    // (in_dim, n_latents), channel-innermost. Lets a caller feed the reference
    // implementation's initial latent and compare the result directly, without
    // having to reproduce PyTorch's RNG.
    std::vector<float> initial_latent;
};

struct SoundEffectResult {
    std::vector<float> waveform;        // mono float32 at `sampling_rate`
    int32_t            sampling_rate = 0;
    std::string        prompt;          // after formatting and cleaning
    int32_t            n_prompt_tokens = 0;
    int32_t            n_latents       = 0;

    // The final latent behind `waveform`, (out_dim, n_latents) channel-innermost.
    // Kept for the same reason the TTS pipeline keeps its codes: it is the seam
    // to diff against a reference without comparing audio.
    std::vector<float> latent;

    // The text conditioning, (text_dim, text_max_len) channel-innermost, right
    // padded with zeros. Worth keeping separately because the latent is a *weak*
    // test of this path: the guidance signal decays to under 1% of the velocity
    // within a few solver steps, so a broken text encoder still lands close.
    std::vector<float> context;

    double text_seconds = 0.0, sample_seconds = 0.0, decode_seconds = 0.0;
};

// Invoked after each solver step, with the 1-based step index and the total.
using SoundEffectProgress = std::function<void(int step, int total)>;

SoundEffectResult generate_sound_effect(Model & model,
                                        const SoundEffectRequest & req,
                                        SoundEffectProgress cb = {});

} // namespace openmoss
