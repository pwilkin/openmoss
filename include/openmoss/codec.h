// SPDX-License-Identifier: Apache-2.0
//
// MOSS Audio Tokenizer (RVQ codec) wrapped in GGML compute graphs.
//
// Architecture (from upstream config.json):
//
//   Encoder: 4 stages of Causal Transformer + PatchedPretransform downsample
//     stage 0  patch=240 (waveform → 240-dim @ 100 Hz frames)
//     stage 1  12 layers, d=768
//     stage 2  patch=2 (downsample 2×)
//     stage 3  12 layers, d=768
//     stage 4  patch=2
//     stage 5  12 layers, d=768
//     stage 6  patch=2
//     stage 7  32 layers, d=1280   ← biggest stage
//     output   768-dim codebook input
//
//   Quantizer: RLFQ, 32 codebooks, codebook_size=1024, codebook_dim=8,
//              rvq_dim=512, internal projections (input_dim 768 ↔ rvq_dim 512).
//
//   Decoder: mirror of encoder (one variant uses time-reversed stages).
//
// Frame rate: 12.5 Hz (downsample 1920 = 240 × 2 × 2 × 2).

#pragma once

#include <cstdint>
#include <vector>

namespace openmoss {

class Model;

// Encoder: waveform → 32 codebook indices.
//   waveform: float32 mono @ 24 kHz, length T_wav (will be padded to multiple of 1920)
//   returns: (n_vq, T_audio) row-major, where T_audio = T_wav / 1920
std::vector<int32_t> codec_encode(Model & model,
                                  const float * waveform,
                                  int64_t       n_samples,
                                  int32_t &     n_vq_out,
                                  int32_t &     t_audio_out);

// Decoder: 32 codebook indices → waveform.
//   codes: (n_vq, T_audio) row-major
//   returns: float32 mono @ 24 kHz, length T_audio * 1920
std::vector<float> codec_decode(Model & model,
                                const int32_t * codes,
                                int32_t         n_vq,
                                int32_t         t_audio);

// Internal owner of the codec compute graphs; held inside Model.
class CodecGraphs;

} // namespace openmoss
