// SPDX-License-Identifier: Apache-2.0
//
// MOSS-SoundEffect-v2.0 end to end: prompt → waveform, by flow matching.
//
//   1. Format the prompt and run the Qwen3-1.7B text encoder over it, giving a
//      512-slot conditioning tensor.
//   2. Precompute every DiT block's cross-attention K/V for that conditioning,
//      and again for the unconditional branch.
//   3. Integrate the velocity field from pure noise down to a clean latent with
//      an Euler solver, applying classifier-free guidance at each step.
//   4. Decode the latent with the DAC VAE and crop to the requested duration.
//
// Two shortcuts here are worth stating outright, because both look like
// oversights otherwise:
//
//   * The unconditional context is exactly zeros. The negative prompt is the
//     empty string, which tokenizes to no real tokens, and the text encoder then
//     zeroes every row past the real length — so there is nothing for Qwen3 to
//     do. It is skipped entirely.
//
//   * Duration is textual. The prompt carries a " duration: <X>s" suffix and the
//     DiT always denoises the full `moss.max_seconds`; the waveform is cropped
//     afterwards. Generating a shorter latent would be off-distribution.
//
// Both branches' cross-attention K/V are built once for the whole run, since the
// conditioning does not change across steps. At 100 steps that saves ~14 TFLOP.

#include "openmoss/soundeffect.h"
#include "openmoss/model.h"
#include "openmoss/tokenizer.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace openmoss {

namespace {

using clock_t_   = std::chrono::steady_clock;
using seconds_t  = std::chrono::duration<double>;

// The upstream prompter cleans with ftfy.fix_text, two HTML unescapes and a
// whitespace collapse. The first two are identity transforms for any prompt
// that is already valid, unescaped text, so only the collapse is reproduced:
// runs of whitespace become a single space, and the ends are trimmed.
std::string whitespace_clean(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    bool pending_space = false;
    for (char c : in) {
        const bool ws = (c == ' ' || c == '\t' || c == '\n' ||
                         c == '\r' || c == '\f' || c == '\v');
        if (ws) {
            pending_space = !out.empty();
        } else {
            if (pending_space) out.push_back(' ');
            pending_space = false;
            out.push_back(c);
        }
    }
    return out;
}

std::string trim(const std::string & in) {
    size_t b = in.find_first_not_of(" \t\n\r\f\v");
    if (b == std::string::npos) return {};
    size_t e = in.find_last_not_of(" \t\n\r\f\v");
    return in.substr(b, e - b + 1);
}

// `f"{p.strip()} duration: {seconds:.1f}s"`, then the whitespace collapse. The
// order matters: the strip happens before the suffix is appended, so the
// collapse only ever touches the caller's text.
std::string format_prompt(const std::string & prompt, float seconds, bool append) {
    std::string p = trim(prompt);
    if (append) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " duration: %.1fs", double(seconds));
        p += buf;
    }
    return whitespace_clean(p);
}

// Box-Muller on a fixed 64-bit Mersenne Twister. Deliberately explicit rather
// than std::normal_distribution, whose output is implementation-defined — a
// seed has to mean the same thing on every platform. It does *not* reproduce
// PyTorch's noise, so seeds are not comparable with the reference; feed
// SoundEffectRequest::initial_latent for that.
std::vector<float> seeded_noise(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n);
    const double scale = 1.0 / 9007199254740992.0;   // 2^-53
    for (size_t i = 0; i < n; i += 2) {
        double u1, u2;
        do { u1 = double(rng() >> 11) * scale; } while (u1 <= 0.0);
        u2 = double(rng() >> 11) * scale;
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 2.0 * 3.14159265358979323846 * u2;
        out[i] = float(r * std::cos(th));
        if (i + 1 < n) out[i + 1] = float(r * std::sin(th));
    }
    return out;
}

// FlowMatchScheduler with extra_one_step=True and sigma_min=0:
//   sigmas = linspace(1, 0, N+1)[:-1]
//   sigmas = shift*s / (1 + (shift-1)*s)
// The Euler step then walks sigma[i] → sigma[i+1], with the last step landing
// exactly on 0.
std::vector<double> make_sigmas(int n_steps, double shift) {
    std::vector<double> sigmas(size_t(n_steps) + 1, 0.0);
    for (int i = 0; i < n_steps; ++i) {
        const double s = 1.0 - double(i) / double(n_steps);
        sigmas[size_t(i)] = shift * s / (1.0 + (shift - 1.0) * s);
    }
    sigmas[size_t(n_steps)] = 0.0;   // the terminal sigma the last step jumps to
    return sigmas;
}

// Run the Qwen3 text encoder and lay its last hidden state out as
// (text_dim, n_text) channel-innermost, right-padded with zeros.
//
// The padding never needs to reach the backbone. Upstream right-pads to 512 and
// passes an attention mask, but the encoder is causal, so a real token at
// position i attends only to 0..i — all real — and every row past the real
// length is explicitly zeroed afterwards. Running just the real tokens is
// therefore exact, and skips up to 511 wasted token forwards.
std::vector<float> encode_text(Model & model, const std::string & text,
                               int32_t & n_tokens_out) {
    const auto & d = model.dims();
    const int32_t n_text = d.text_max_len;
    const int32_t hidden = d.hidden_size;

    std::vector<int32_t> ids = model.tokenizer()->encode(text, /*add_special=*/false);
    if (int32_t(ids.size()) > n_text) ids.resize(size_t(n_text));
    n_tokens_out = int32_t(ids.size());

    std::vector<float> ctx(size_t(hidden) * size_t(n_text), 0.0f);
    if (ids.empty()) return ctx;   // the empty prompt really is all zeros

    llama_context * lctx = model.backbone_ctx();
    llama_memory_clear(llama_get_memory(lctx), /*data=*/true);

    llama_batch batch = llama_batch_init(int32_t(ids.size()), /*embd=*/0, /*n_seq_max=*/1);
    batch.n_tokens = int32_t(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = llama_pos(i);
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;      // every position's hidden state is wanted
    }
    const int32_t rc = llama_decode(lctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("encode_text: llama_decode returned " + std::to_string(rc));
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        const float * h = llama_get_embeddings_ith(lctx, int32_t(i));
        if (!h) {
            throw std::runtime_error("encode_text: no hidden state at position "
                                     + std::to_string(i));
        }
        std::memcpy(ctx.data() + i * size_t(hidden), h, size_t(hidden) * sizeof(float));
    }
    return ctx;
}

} // namespace

SoundEffectResult generate_sound_effect(Model & model,
                                        const SoundEffectRequest & req,
                                        SoundEffectProgress cb) {
    const auto & d = model.dims();
    if (d.arch != Arch::SoundEffect) {
        throw std::runtime_error("generate_sound_effect: model is not moss_soundeffect");
    }
    if (req.num_inference_steps < 1) {
        throw std::runtime_error("generate_sound_effect: num_inference_steps must be >= 1");
    }

    const float max_seconds = float(d.max_seconds);
    const float seconds = std::round(req.seconds * 10.0f) / 10.0f;
    if (seconds <= 0.0f) {
        throw std::runtime_error("generate_sound_effect: seconds must be > 0");
    }
    if (seconds > max_seconds) {
        throw std::runtime_error(
            "generate_sound_effect: seconds " + std::to_string(seconds) +
            " exceeds this model's maximum of " + std::to_string(int(max_seconds)));
    }

    SoundEffectResult res;
    res.sampling_rate = d.sampling_rate;
    res.prompt = format_prompt(req.prompt, seconds, req.append_duration_suffix);

    // The latent is always the full length; only the waveform gets cropped.
    const int64_t n_latents = int64_t(d.sampling_rate) * int64_t(d.max_seconds) / d.vae_hop;
    const int64_t C         = d.dit_in_dim;
    res.n_latents = int32_t(n_latents);

    DiTGraph dit(model);

    // ── 1. Text conditioning ───────────────────────────────────────────────
    auto t0 = clock_t_::now();
    DiTContextPtr ctx_pos, ctx_neg;
    {
        res.context = encode_text(model, res.prompt, res.n_prompt_tokens);
        ctx_pos = dit.encode_context(res.context.data(), d.text_max_len);
    }
    if (req.cfg_scale != 1.0f) {
        // The empty negative prompt gives exactly zeros, with no text encoder
        // pass at all. A non-empty one is encoded normally.
        if (trim(req.negative_prompt).empty()) {
            const std::vector<float> zeros(size_t(d.dit_text_dim) * size_t(d.text_max_len), 0.0f);
            ctx_neg = dit.encode_context(zeros.data(), d.text_max_len);
        } else {
            int32_t unused = 0;
            const auto ctx = encode_text(model, whitespace_clean(trim(req.negative_prompt)), unused);
            ctx_neg = dit.encode_context(ctx.data(), d.text_max_len);
        }
    }
    res.text_seconds = seconds_t(clock_t_::now() - t0).count();

    // ── 2. Euler solve ─────────────────────────────────────────────────────
    t0 = clock_t_::now();
    const auto sigmas = make_sigmas(req.num_inference_steps, double(req.sigma_shift));

    std::vector<float> latent;
    if (!req.initial_latent.empty()) {
        if (req.initial_latent.size() != size_t(C * n_latents)) {
            throw std::runtime_error(
                "generate_sound_effect: initial_latent has " +
                std::to_string(req.initial_latent.size()) + " values, expected " +
                std::to_string(C * n_latents));
        }
        latent = req.initial_latent;
    } else {
        // sigma_0 is exactly 1, so the starting point is plain unit-variance
        // noise with no scaling.
        latent = seeded_noise(req.seed, size_t(C * n_latents));
    }

    std::vector<float> v_pos(size_t(C * n_latents));
    std::vector<float> v_neg(req.cfg_scale != 1.0f ? size_t(C * n_latents) : 0);

    for (int step = 0; step < req.num_inference_steps; ++step) {
        const float timestep = float(sigmas[size_t(step)] * double(d.sched_train_steps));
        dit.forward(latent.data(), n_latents, timestep, *ctx_pos, v_pos.data());

        const double dt = sigmas[size_t(step) + 1] - sigmas[size_t(step)];
        if (req.cfg_scale != 1.0f) {
            dit.forward(latent.data(), n_latents, timestep, *ctx_neg, v_neg.data());
            for (size_t i = 0; i < latent.size(); ++i) {
                const double v = double(v_neg[i])
                               + double(req.cfg_scale) * (double(v_pos[i]) - double(v_neg[i]));
                latent[i] = float(double(latent[i]) + v * dt);
            }
        } else {
            for (size_t i = 0; i < latent.size(); ++i) {
                latent[i] = float(double(latent[i]) + double(v_pos[i]) * dt);
            }
        }
        if (cb) cb(step + 1, req.num_inference_steps);
    }
    res.sample_seconds = seconds_t(clock_t_::now() - t0).count();
    res.latent = latent;

    // ── 3. Decode and crop ─────────────────────────────────────────────────
    t0 = clock_t_::now();
    DacDecoder vae(model);
    res.waveform = vae.decode(latent.data(), n_latents);
    res.decode_seconds = seconds_t(clock_t_::now() - t0).count();

    const size_t want = size_t(double(d.sampling_rate) * double(seconds));
    if (res.waveform.size() > want) res.waveform.resize(want);

    return res;
}

} // namespace openmoss
