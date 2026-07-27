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
    // Channel-interleaved f32 at the model's sampling rate / channel count.
    std::optional<std::vector<float>> reference_wav;
    std::optional<std::string>        instruction;     // e.g. voice description
    std::optional<std::string>        language;        // "en", "zh", ...
    std::optional<std::string>        quality;         // upstream "quality" hint
    std::optional<int>                tokens;          // duration hint (1s ≈ 12.5 tokens)
    int                               max_new_tokens = 4096;
    SamplingConfig                    sampling;
};

struct GenerateResult {
    // Channel-interleaved f32 at the model's sampling rate. Length is
    // n_audio_frames * downsample_rate * n_channels.
    std::vector<float> waveform;
    int32_t            n_channels = 1;
    int32_t            n_audio_frames; // before upsampling

    // The raw codec codes behind `waveform`, (n_codebooks, n_audio_frames)
    // row-major. Kept so callers can diff against a reference implementation
    // without having to compare audio.
    std::vector<int32_t> audio_codes;
    int32_t              n_codebooks = 0;
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
