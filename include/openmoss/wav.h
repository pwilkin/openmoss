// SPDX-License-Identifier: Apache-2.0
//
// Minimal RIFF/WAV I/O — mono float32 → 16-bit PCM and back. Avoids pulling in
// libsndfile so that the shipped binary stays a single self-contained executable.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openmoss {

// Read a mono WAV file. Resamples via simple linear interpolation if the file
// is not at `target_sr`. Returns f32 PCM in [-1, 1].
std::vector<float> read_wav_mono(const std::string & path, int32_t target_sr);

// Write mono f32 PCM as 16-bit little-endian WAV at `sample_rate`.
void write_wav_mono(const std::string & path,
                    const float *       pcm,
                    int64_t             n_samples,
                    int32_t             sample_rate);

// Same encoding, but returns the WAV bytes in memory (RIFF header + PCM).
// Useful for serving WAVs over HTTP without touching disk.
std::vector<uint8_t> encode_wav_mono(const float * pcm,
                                      int64_t       n_samples,
                                      int32_t       sample_rate);

// Decode a WAV byte buffer (in-memory equivalent of read_wav_mono). Resamples
// linearly to `target_sr` if the embedded sample rate differs.
std::vector<float> decode_wav_mono(const uint8_t * data,
                                    size_t          n_bytes,
                                    int32_t         target_sr);

} // namespace openmoss
