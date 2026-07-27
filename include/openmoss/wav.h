// SPDX-License-Identifier: Apache-2.0
//
// Minimal RIFF/WAV I/O — float32 ↔ 16-bit PCM. Avoids pulling in libsndfile so
// that the shipped binary stays a single self-contained executable.
//
// Multi-channel buffers are **interleaved** (L R L R …), matching both the WAV
// on-disk layout and the codec's channel-interleaved waveform representation.
// `n_samples` always counts individual samples, not frames: a 1 s stereo buffer
// at 48 kHz is 96000 samples.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openmoss {

// Read a WAV file as `want_channels` interleaved channels, resampling linearly
// if the file is not at `target_sr`. Channel count is adapted as needed:
// downmix by averaging to mono, take the leading channels when the file has
// more than requested, and duplicate when it has fewer.
std::vector<float> read_wav(const std::string & path, int32_t target_sr,
                            int32_t want_channels);

// Write interleaved f32 PCM as 16-bit little-endian WAV.
void write_wav(const std::string & path,
               const float *       pcm,
               int64_t             n_samples,
               int32_t             sample_rate,
               int32_t             n_channels);

// Same encoding, but returns the WAV bytes in memory (RIFF header + PCM).
// Useful for serving WAVs over HTTP without touching disk.
std::vector<uint8_t> encode_wav(const float * pcm,
                                int64_t       n_samples,
                                int32_t       sample_rate,
                                int32_t       n_channels);

// Decode a WAV byte buffer (in-memory equivalent of read_wav).
std::vector<float> decode_wav(const uint8_t * data,
                              size_t          n_bytes,
                              int32_t         target_sr,
                              int32_t         want_channels);

// ── Mono convenience wrappers (want_channels / n_channels = 1) ──────────────
std::vector<float> read_wav_mono(const std::string & path, int32_t target_sr);
void write_wav_mono(const std::string & path, const float * pcm,
                    int64_t n_samples, int32_t sample_rate);
std::vector<uint8_t> encode_wav_mono(const float * pcm, int64_t n_samples,
                                     int32_t sample_rate);
std::vector<float> decode_wav_mono(const uint8_t * data, size_t n_bytes,
                                   int32_t target_sr);

} // namespace openmoss
