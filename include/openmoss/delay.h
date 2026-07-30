// SPDX-License-Identifier: Apache-2.0
//
// Delay-pattern state machine for multi-codebook RVQ generation, mirroring
// `MossTTSDelayModel.generate` in the upstream PyTorch code.
//
// The model emits 1 + n_vq tokens per step. Codebook i is "delayed" by i
// steps so that early tokens pad-fill before the model starts predicting them
// and there's a flush window of n_vq pad tokens at end-of-utterance.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openmoss/model.h"

namespace openmoss {

// Defined in delay.cpp; held per-DelayState so each generation seeds its own
// stream (a single static RNG would ignore per-request seeds and leak state
// across requests on the persistent server).
struct Rng;

// SamplingConfig now lives in model.h — it is shared by every family.

// Per-step result: (1 + n_vq) ids for the *next* position.
struct DelayStep {
    std::vector<int32_t> ids; // size = 1 + n_vq
    bool stop = false;
};

class DelayState {
public:
    DelayState(const ModelDims & dims, const std::vector<std::vector<int32_t>> & prompt_ids);
    ~DelayState();

    // Advance one step given:
    //   text_logits:  (vocab,)
    //   audio_logits: (n_vq, audio_vocab_size + 1) row-major
    DelayStep step(const float * text_logits,
                   const float * audio_logits,
                   const SamplingConfig & sc);

    // Extract just the audio codebook ids, with pad/delay rows stripped, for
    // codec decoding. Shape returned: (n_vq, T_a) row-major.
    std::vector<int32_t> extract_audio_codes(int32_t & n_vq_out, int32_t & t_audio) const;

private:
    const ModelDims m_dims;
    int             m_step_idx       = 0;
    bool            m_is_audio       = false;
    bool            m_is_stopping    = false;
    int64_t         m_audio_length   = 0;
    // Audio frames already present in the prompt when this segment opened — the
    // assistant-side continuation prefix. min/max_audio_frames bound what the
    // model *generates*, so they are measured against m_audio_length minus this.
    // Reset to 0 whenever a segment closes, since later segments have no prefix.
    int64_t         m_audio_prefix   = 0;
    int64_t         m_delayed_length = -1;     // -1 sentinel ≈ INT64_MAX
    std::vector<std::vector<int32_t>> m_history; // (T, 1+n_vq)
    std::unique_ptr<Rng> m_rng;                  // seeded lazily on first step()
};

} // namespace openmoss
