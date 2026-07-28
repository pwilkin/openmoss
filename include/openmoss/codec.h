// SPDX-License-Identifier: Apache-2.0
//
// MOSS Audio Tokenizer (RVQ codec) wrapped in GGML compute graphs.
//
// Both generations share one design: a stack of causal Transformer stages
// separated by PatchedPretransform reshapes that trade channels for time. The
// encoder patches the waveform down to the frame rate, an RLFQ quantizer turns
// the frame latent into codes, and the decoder mirrors the encoder back up.
// Which generation a GGUF holds is recorded in `moss.codec.version`.
//
//   v1 (MOSS-Audio-Tokenizer)     24 kHz mono,   hop 1920, 4 stages/side
//                                 patches 240,2,2,2; a single 10 s attention
//                                 window at every stage.
//
//   v2 (MOSS-Audio-Tokenizer-v2)  48 kHz stereo, hop 3840/channel, 6 stages/side
//                                 patches 240,2,2,2,2,2; per-stage attention
//                                 windows of {10,10,8,4,2,1} s outwards from the
//                                 frame end. Every stage materialises learned
//                                 input/output projections, including square
//                                 ones (v1 used nn.Identity there).
//
//   Quantizer (identical in both): RLFQ, 32 codebooks, codebook_size=1024,
//   codebook_dim=8, rvq_dim=512, projections 768 ↔ 512.
//
// Frame rate is 12.5 Hz in both. Stereo in v2 is *not* a second channel through
// the network: L/R are time-interleaved into a single stream before the encoder
// and de-interleaved after the decoder, so the network stays 1-channel and the
// waveform buffers below are simply interleaved.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace openmoss {

class Model;

// Encoder: waveform → codebook indices.
//   waveform:  float32, channel-interleaved at the model's sampling rate.
//              Padded internally to a whole number of frames.
//   n_vq_out:  set to the number of codebooks the model consumes (32 for
//              MOSS-TTS-Delay, 12 for MOSS-TTS-Local). RVQ is greedy, so a
//              prefix of the codebooks is exact, not an approximation.
//   returns:   (n_vq_out, T_audio) row-major.
std::vector<int32_t> codec_encode(Model & model,
                                  const float * waveform,
                                  int64_t       n_samples,
                                  int32_t &     n_vq_out,
                                  int32_t &     t_audio_out);

// Decoder: codebook indices → waveform.
//   codes:   (n_vq, T_audio) row-major
//   returns: float32, channel-interleaved, length
//            T_audio * downsample_rate * n_channels
std::vector<float> codec_decode(Model & model,
                                const int32_t * codes,
                                int32_t         n_vq,
                                int32_t         t_audio);

// Incremental decoder: the same waveform as codec_decode(), emitted a chunk at
// a time as codes arrive.
//
// Chunking the decoder statelessly does not work. Its six stages are banded at
// 10+10+8+4+2+1 s and those windows *add*, so a chunk needs 35 s of history —
// measured, 128 frames of context leaves a relative error of 1.144, larger than
// the signal. Paying that history back on every chunk costs 36x the work at 1 s
// granularity, which the codec's ~30x real time cannot sustain.
//
// So this keeps a per-stage K/V cache instead: each push's graph is handed the
// prefix its queries can still reach, and nothing is recomputed. Cost per frame
// matches the batch decode's, and granularity drops to a single frame.
//
// The win is *latency*, not throughput — for MOSS-TTS-Local the LM is the
// constraint at 1.74x real time, and the codec was never the bottleneck.
class CodecStreamDecoder {
public:
    // `n_vq` is fixed for the stream — the codec sums residuals coarse-to-fine,
    // so a prefix of the codebooks is valid, but it may not vary mid-utterance.
    CodecStreamDecoder(Model & model, int32_t n_vq);
    ~CodecStreamDecoder();

    CodecStreamDecoder(const CodecStreamDecoder &)             = delete;
    CodecStreamDecoder & operator=(const CodecStreamDecoder &) = delete;

    // Feed `t_audio` more frames — (n_vq, t_audio) row-major, same layout as
    // codec_decode(). Returns exactly the samples those frames produce:
    // t_audio * samples_per_frame() channel-interleaved f32, picking up where
    // the previous push left off.
    //
    // Any chunk size works, but the graph is rebuilt per push, so a constant one
    // lets the allocator settle. Larger chunks amortise the per-push overhead;
    // smaller ones cut latency.
    std::vector<float> push(const int32_t * codes, int32_t t_audio);

    // Forget all cached state. The next push starts a fresh utterance at
    // position 0.
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Internal owner of the codec compute graphs; held inside Model.
class CodecGraphs;

} // namespace openmoss
