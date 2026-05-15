// SPDX-License-Identifier: Apache-2.0
//
// High-level end-to-end TTS pipeline: text (+ optional reference audio) →
// audio codes → waveform.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "openmoss/delay.h"
#include "openmoss/model.h"

namespace openmoss {

struct GenerateRequest {
    std::string text;
    std::optional<std::vector<float>> reference_wav;   // mono f32 @ 24 kHz
    std::optional<std::string>        instruction;     // e.g. voice description
    std::optional<std::string>        language;        // "en", "zh", ...
    std::optional<std::string>        quality;         // upstream "quality" hint
    std::optional<int>                tokens;          // duration hint (1s ≈ 12.5 tokens)
    int                               max_new_tokens = 4096;
    SamplingConfig                    sampling;
};

struct GenerateResult {
    std::vector<float> waveform;       // mono f32 @ 24 kHz
    int32_t            n_audio_frames; // before upsampling
    double             prefill_seconds = 0.0;
    double             generate_seconds = 0.0;
    double             decode_seconds   = 0.0;
};

// Optional callback invoked once the codec produces a chunk of waveform; lets
// callers stream output. The callback's argument is owned by the pipeline and
// only valid for the duration of the call.
using StreamCallback = std::function<void(const float * pcm, int64_t n_samples)>;

GenerateResult generate(Model & model,
                        const GenerateRequest & req,
                        StreamCallback cb = {});

} // namespace openmoss
