// SPDX-License-Identifier: Apache-2.0
//
// Delay-pattern state machine + sampling for MOSS-TTS-Delay.
//
// Faithful port of `MossTTSDelayModel.generate` in
// `modeling_moss_tts.py` (and the reference NumPy implementation in
// `moss_tts_delay/llama_cpp/delay_state.py`), specialised to batch-size 1.
//
// At each generation step the model emits 1 + n_vq tokens:
//   - column 0 is a *text* token, picked from the Qwen3 vocab
//   - columns 1..n_vq are *audio* codebook indices
//
// Three state variables drive the scheduling:
//
//   audio_length     count of audio frames emitted so far in the current segment
//                    (reset to 0 on AUDIO_END)
//   delayed_length   −1 sentinel before the delay window starts; once the model
//                    emits an `audio_delay_slot` text token, ticks 0..n_vq;
//                    while < n_vq, audio columns 0..delayed_length are still
//                    valid; ≥ n_vq means flushing has finished
//   is_audio         true iff currently emitting an audio segment
//   is_stopping      true once the model emitted im_end (terminate)
//
// Per-step decision (column 0):
//   - if delayed_length < n_vq → emit audio_delay_slot (no sampling)
//   - if delayed_length == n_vq → emit audio_end (close segment)
//   - else → sample text from the masked text-vocab distribution
//
// Per-step decision (columns 1..n_vq):
//   - codebook i is "real" only if audio_length > i  AND  delayed_length-1 < i
//     (i.e. we are far enough into the segment for it to start, but not yet
//      flushing past it). Otherwise emit audio_pad_code.
//
// Sampling (text or audio):
//   logits → repetition penalty (audio only) → /temperature → top-k → top-p
//          → softmax → multinomial draw

#include "openmoss/delay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace openmoss {

// ──────────────────────────────────────────────────────────────────────────
// Sampling helpers (CPU side; logits are tiny — vocab ≤ 1025 audio, ~155k text)
// ──────────────────────────────────────────────────────────────────────────

// Declared in delay.h (forward-declared there, defined here) so DelayState can
// own one per instance.
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) {
        if (seed == 0) {
            std::random_device rd;
            seed = (uint64_t(rd()) << 32) | rd();
        }
        g.seed(seed);
    }
    float uniform01() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(g);
    }
};

namespace {

void apply_repetition_penalty(float * logits, int vocab,
                              const std::vector<int32_t> & history,
                              float penalty) {
    if (penalty == 1.0f) return;
    for (int32_t id : history) {
        if (id < 0 || id >= vocab) continue;
        if (logits[id] > 0) logits[id] /= penalty;
        else                logits[id] *= penalty;
    }
}

// Softmax in place over a contiguous logits buffer (with -inf entries as masks).
void softmax_inplace(float * x, int n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (x[i] > mx) mx = x[i];
    if (!std::isfinite(mx)) {
        // All-masked; keep zeros for caller to handle.
        std::fill(x, x + n, 0.f);
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.f) { std::fill(x, x + n, 0.f); return; }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

int32_t sample_one(float * logits, int vocab, float temperature,
                   float top_p, int top_k, bool do_sample, Rng & rng) {
    if (!do_sample) {
        // Argmax
        int best = 0;
        for (int i = 1; i < vocab; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }

    if (temperature > 0.f && temperature != 1.f) {
        const float inv = 1.f / temperature;
        for (int i = 0; i < vocab; ++i) logits[i] *= inv;
    }

    // top-k: build a list of (id, logit) pairs and pick the K largest.
    if (top_k > 0 && top_k < vocab) {
        std::vector<std::pair<float,int32_t>> v;
        v.reserve(vocab);
        for (int i = 0; i < vocab; ++i) {
            if (std::isfinite(logits[i])) v.emplace_back(logits[i], i);
        }
        if (int(v.size()) > top_k) {
            std::nth_element(v.begin(), v.begin() + top_k, v.end(),
                             [](auto & a, auto & b){ return a.first > b.first; });
            v.resize(top_k);
        }
        // Mask everything except the top-k by zeroing logits.
        std::vector<bool> keep(vocab, false);
        for (auto & p : v) keep[p.second] = true;
        for (int i = 0; i < vocab; ++i)
            if (!keep[i]) logits[i] = -std::numeric_limits<float>::infinity();
    }

    softmax_inplace(logits, vocab);

    // top-p: sort, accumulate, mask the tail.
    if (top_p > 0.f && top_p < 1.f) {
        std::vector<int32_t> order(vocab);
        for (int i = 0; i < vocab; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](int32_t a, int32_t b){ return logits[a] > logits[b]; });
        float acc = 0.f;
        size_t cut = order.size();
        for (size_t i = 0; i < order.size(); ++i) {
            acc += logits[order[i]];
            if (acc >= top_p) { cut = i + 1; break; }
        }
        // Renormalise the kept set.
        float sum = 0.f;
        for (size_t i = 0; i < cut;            ++i) sum += logits[order[i]];
        for (size_t i = cut; i < order.size(); ++i) logits[order[i]] = 0.f;
        if (sum > 0.f) {
            const float inv = 1.f / sum;
            for (size_t i = 0; i < cut; ++i) logits[order[i]] *= inv;
        }
    }

    // Multinomial draw via inverse CDF.
    const float u = rng.uniform01();
    float acc = 0.f;
    for (int i = 0; i < vocab; ++i) {
        acc += logits[i];
        if (acc >= u) return i;
    }
    // Fallback for floating-point sum < 1.0
    for (int i = vocab - 1; i >= 0; --i) if (logits[i] > 0.f) return i;
    return 0;
}

} // namespace

// ──────────────────────────────────────────────────────────────────────────
// DelayState
// ──────────────────────────────────────────────────────────────────────────

DelayState::~DelayState() = default;

DelayState::DelayState(const ModelDims & dims,
                       const std::vector<std::vector<int32_t>> & prompt_ids)
    : m_dims(dims), m_history(prompt_ids) {
    if (!prompt_ids.empty() && int32_t(prompt_ids.front().size()) != 1 + dims.n_vq) {
        throw std::invalid_argument("DelayState: prompt_ids width must equal 1 + n_vq");
    }

    // Mirror the upstream `is_continuation` logic: if the prompt ends on an
    // audio_start token, we're in continuation mode and the audio segment is
    // already open.
    if (!prompt_ids.empty()) {
        const int32_t last_text = prompt_ids.back().front();
        if (last_text == dims.audio_start_token_id ||
            last_text == dims.audio_assistant_gen_slot_token_id) {
            m_is_audio = true;
            // Search backwards for the most recent audio_start to derive an
            // initial audio_length.
            for (int64_t i = int64_t(prompt_ids.size()) - 1; i >= 0; --i) {
                if (prompt_ids[i].front() == dims.audio_start_token_id) {
                    m_audio_length = int64_t(prompt_ids.size()) - i;
                    break;
                }
            }
        }
    }
}

DelayStep DelayState::step(const float * text_logits,
                           const float * audio_logits,
                           const SamplingConfig & sc) {
    const int32_t n_vq         = m_dims.n_vq;
    const int32_t aud_v        = m_dims.audio_vocab_size; // 1024
    const int32_t aud_v_full   = aud_v + 1;                // 1025
    const int32_t pad_code     = m_dims.audio_pad_code;

    // RNG is owned per DelayState (per request) and seeded lazily on first use,
    // so each generation honours its own sc.seed and no state leaks across the
    // server's requests.
    if (!m_rng) m_rng = std::make_unique<Rng>(sc.seed);
    Rng & rng = *m_rng;

    DelayStep result;
    result.ids.assign(1 + n_vq, pad_code);

    // ── Column 0: text token ──────────────────────────────────────────────
    int32_t next_text = m_dims.pad_token_id;

    if (m_is_stopping) {
        result.ids[0] = m_dims.pad_token_id;
        result.stop = true;
        return result;
    }

    if (m_delayed_length >= 0 && m_delayed_length < n_vq) {
        // We're still in the initial delay window — model must emit the slot token.
        next_text = m_dims.audio_assistant_delay_slot_token_id;
    } else if (m_delayed_length == n_vq) {
        // Time to close the audio segment with audio_end.
        next_text = m_dims.audio_end_token_id;
        m_is_audio = false;
    } else if (m_is_audio && m_delayed_length < 0 && sc.max_audio_frames > 0 &&
               m_audio_length >= int64_t(sc.max_audio_frames)) {
        // Hit the length cap — force the end-of-segment flush so the model can't
        // ramble far past the requested length.
        next_text = m_dims.audio_assistant_delay_slot_token_id;
    } else {
        // Sample text from the (caller-provided, copied) logits buffer.
        // We need a mutable copy to apply masking; the caller owns the original.
        // Vocab size comes from the loaded backbone (token_embd rows), not a
        // hard-coded constant, so v1.5 backbones with a different vocab work.
        const int text_vocab_size = m_dims.text_vocab_size;
        std::vector<float> tmp(text_logits, text_logits + text_vocab_size);

        // Mask special tokens that must not appear here.
        auto mask = [&](int id) {
            if (id >= 0 && id < text_vocab_size) tmp[id] = -std::numeric_limits<float>::infinity();
        };
        if (!m_is_audio) {
            // Outside an audio segment: forbid pad / generation slot / delay slot / audio_end.
            mask(m_dims.pad_token_id);
            mask(m_dims.audio_assistant_gen_slot_token_id);
            mask(m_dims.audio_assistant_delay_slot_token_id);
            mask(m_dims.audio_end_token_id);
        } else {
            // Inside an audio segment: only the slots are valid for column 0.
            for (int i = 0; i < text_vocab_size; ++i) {
                if (i != m_dims.audio_assistant_gen_slot_token_id &&
                    i != m_dims.audio_assistant_delay_slot_token_id) {
                    tmp[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        if (m_step_idx == 0) mask(m_dims.audio_assistant_delay_slot_token_id);
        if (m_step_idx <= n_vq) mask(m_dims.im_end_token_id);

        // Until we've generated a minimum number of frames, forbid the delay slot
        // (which begins the end-of-segment flush). Without this the model can pick
        // it on the first audio frame, collapsing the segment to T≈n_vq — which
        // extract_audio_codes discards as empty (the degenerate immediate EOS).
        if (m_is_audio && m_delayed_length < 0 &&
            m_audio_length < int64_t(sc.min_audio_frames)) {
            mask(m_dims.audio_assistant_delay_slot_token_id);
        }

        next_text = sample_one(tmp.data(), text_vocab_size,
                               sc.text_temperature, sc.text_top_p, sc.text_top_k,
                               sc.text_temperature > 0.f, rng);

        if (next_text == m_dims.audio_start_token_id)  m_is_audio = true;
        if (next_text == m_dims.im_end_token_id)        m_is_stopping = true;
    }
    result.ids[0] = next_text;

    // ── Columns 1..n_vq: audio codebook tokens ────────────────────────────
    // Pre/post-audio mask per upstream code:
    //   sampling_audio_mask[i] = (audio_length > i) AND (i >= delayed_length)
    // The upstream uses INT64_MAX as the "no delay yet" sentinel and then
    // explicitly forces post = True for that case. We do the same with an
    // explicit sentinel branch, since INT64_MAX-1 < i is never true for
    // finite i and would silently skip every audio codebook during warm-up.
    const bool delayed_in_sentinel = (m_delayed_length < 0);

    // Sampling buffer for one audio head at a time.
    std::vector<float> abuf(aud_v_full);

    // Build a flat history of column-i tokens for repetition penalty.
    std::vector<int32_t> hist_col(m_history.size());

    for (int32_t i = 0; i < n_vq; ++i) {
        const bool pre  = m_audio_length > int64_t(i);
        const bool post = delayed_in_sentinel
                            ? true
                            : int64_t(i) >= m_delayed_length;
        if (!(pre && post)) {
            result.ids[1 + i] = pad_code;
            continue;
        }

        // Copy this head's logits and mask the pad code.
        std::memcpy(abuf.data(), audio_logits + i * aud_v_full,
                    aud_v_full * sizeof(float));
        abuf[pad_code] = -std::numeric_limits<float>::infinity();

        if (sc.audio_repetition_penalty != 1.f) {
            for (size_t h = 0; h < m_history.size(); ++h) hist_col[h] = m_history[h][1 + i];
            apply_repetition_penalty(abuf.data(), aud_v_full, hist_col,
                                      sc.audio_repetition_penalty);
        }
        result.ids[1 + i] = sample_one(abuf.data(), aud_v_full,
                                       sc.audio_temperature, sc.audio_top_p, sc.audio_top_k,
                                       sc.audio_temperature > 0.f, rng);
    }

    // ── Update state ──────────────────────────────────────────────────────
    if (next_text == m_dims.audio_start_token_id ||
        next_text == m_dims.audio_assistant_gen_slot_token_id ||
        next_text == m_dims.audio_assistant_delay_slot_token_id) {
        m_audio_length++;
    }
    if (next_text == m_dims.audio_end_token_id) {
        m_audio_length = 0;
    }

    // delayed_length convention here differs from the reference by one (the
    // reference jumps MAX→0→1 in a single step; here it stays at 0 on the
    // transition step). This is deliberate and matched by extract_audio_codes,
    // which uses `T - n_vq` (not `T - n_vq + 1`): the extra flush frame the
    // off-by-one produces is exactly the one that convention drops, so the
    // emitted frame set is identical to the reference. Do NOT "fix" one without
    // the other — see extract_audio_codes.
    if (m_delayed_length < 0 && next_text == m_dims.audio_assistant_delay_slot_token_id) {
        m_delayed_length = 0;
    } else if (m_delayed_length >= 0) {
        m_delayed_length++;
        if (m_delayed_length > n_vq) m_delayed_length = -1;  // back to sentinel after flush
    }

    m_history.push_back(result.ids);
    m_step_idx++;
    return result;
}

std::vector<int32_t> DelayState::extract_audio_codes(int32_t & n_vq_out, int32_t & t_audio) const {
    // Walk the history, find the last opened audio segment, and unstride the
    // per-codebook delay so each row contains a contiguous block of valid codes.
    n_vq_out = m_dims.n_vq;
    t_audio = 0;

    // Find the first row of the active audio segment. The assistant prompt
    // always ends with `<audio_start>`, so the *last* `audio_start` in the
    // history bounds the generation segment. (Earlier `<audio_start>` rows
    // belong to user-supplied reference audio and must be skipped.) Note we
    // explicitly do NOT match gen_slot here — earlier code searched for
    // `audio_start OR gen_slot` and ended up landing on the *last* generated
    // gen_slot row, leaving T ≈ n_vq and T_audio ≈ 0.
    int64_t start = -1;
    for (int64_t i = int64_t(m_history.size()) - 1; i >= 0; --i) {
        if (m_history[size_t(i)].front() == m_dims.audio_start_token_id) {
            start = i + 1; // first audio frame is immediately after the marker
            break;
        }
    }
    if (start < 0) return {};

    // Find the matching audio_end (or end-of-history).
    int64_t end = int64_t(m_history.size());
    for (int64_t i = start; i < end; ++i) {
        if (m_history[size_t(i)].front() == m_dims.audio_end_token_id) {
            end = i; break;
        }
    }
    const int64_t T = end - start;
    if (T <= int64_t(m_dims.n_vq)) return {}; // not enough frames to cover the delay window

    // Strip the tail-pad rows (last n_vq-1 rows are partially padded) and
    // un-shift each codebook by its index so the output is rectangular.
    const int64_t T_audio = T - int64_t(m_dims.n_vq);
    t_audio = int32_t(T_audio);
    std::vector<int32_t> out(size_t(m_dims.n_vq) * size_t(T_audio));
    for (int32_t cb = 0; cb < m_dims.n_vq; ++cb) {
        for (int64_t t = 0; t < T_audio; ++t) {
            // Codebook cb is delayed by `cb` steps relative to t=0.
            out[size_t(cb) * size_t(T_audio) + size_t(t)] =
                m_history[size_t(start + t + cb)][1 + cb];
        }
    }
    return out;
}

} // namespace openmoss
