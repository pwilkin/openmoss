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

struct SamplingConfig {
    float text_temperature  = 1.5f;
    float text_top_p        = 1.0f;
    int   text_top_k        = 50;

    float audio_temperature       = 1.7f;
    float audio_top_p             = 0.8f;
    int   audio_top_k             = 25;
    float audio_repetition_penalty = 1.0f;

    // Forbid the end-of-segment delay slot until at least this many audio frames
    // have been generated. Prevents the model from collapsing the segment on the
    // first frame (degenerate immediate end-of-speech). 0 = no floor.
    int min_audio_frames = 0;

    // Force the end-of-segment delay slot once this many audio frames exist.
    // Bounds the segment so the model can't ramble far past the requested length.
    // 0 = no cap.
    int max_audio_frames = 0;

    uint64_t seed = 0; // 0 = nondeterministic
};

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
    int64_t         m_delayed_length = -1;     // -1 sentinel ≈ INT64_MAX
    std::vector<std::vector<int32_t>> m_history; // (T, 1+n_vq)
    std::unique_ptr<Rng> m_rng;                  // seeded lazily on first step()
};

} // namespace openmoss
