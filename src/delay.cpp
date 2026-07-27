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
#include "openmoss/frame_decoder.h"

#include "sampling_internal.h"

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

// Rng and the sampling primitives live in sampling_internal.h — they are
// shared verbatim with the local-transformer decoder.
using sampling::apply_repetition_penalty;
using sampling::softmax_inplace;
using sampling::sample_one;


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

// ───────────────────────────────────────────────────────────────────────────
// IFrameDecoder adapter
//
// The delay pattern projects the backbone hidden state through all n_vq heads
// up front, so this is a thin wrapper: run the heads, then hand the dense
// (n_vq, Vfull) logits block to DelayState exactly as before.
// ───────────────────────────────────────────────────────────────────────────

namespace {

class DelayFrameDecoder final : public IFrameDecoder {
public:
    DelayFrameDecoder(Model & model,
                      const std::vector<std::vector<int32_t>> & prompt_rows)
        : m_model(model), m_state(model.dims(), prompt_rows),
          m_row_width(1 + model.dims().n_vq) {}

    FrameStep step(const float * text_logits,
                   const float * hidden,
                   const SamplingConfig & sc) override {
        const std::vector<float> audio_logits = m_model.compute_audio_logits(hidden);
        DelayStep s = m_state.step(text_logits, audio_logits.data(), sc);
        FrameStep out;
        out.ids  = std::move(s.ids);
        out.stop = s.stop;
        return out;
    }

    std::vector<int32_t> extract_audio_codes(int32_t & n_cb_out,
                                             int32_t & t_audio_out) const override {
        return m_state.extract_audio_codes(n_cb_out, t_audio_out);
    }

    int32_t row_width() const override { return m_row_width; }

private:
    Model &    m_model;
    DelayState m_state;
    int32_t    m_row_width;
};

} // namespace

std::unique_ptr<IFrameDecoder> make_delay_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows) {
    return std::make_unique<DelayFrameDecoder>(model, prompt_rows);
}

} // namespace openmoss
