// SPDX-License-Identifier: Apache-2.0
//
// Family-agnostic seam between the backbone loop and the audio-code sampler.
//
// The two MOSS families differ in *when* the audio heads run:
//
//   MOSS-TTS-Delay  projects the backbone hidden state through all n_vq heads
//                   in parallel, then samples n_vq codes independently, with
//                   codebook i staggered i steps behind codebook 0.
//
//   MOSS-TTS-Local  runs a small "local" (depth) transformer for n_vq sequential
//                   micro-steps per frame; code k must be sampled before the
//                   logits for code k+1 can be computed at all.
//
// That is why the interface takes the raw `hidden` vector rather than
// pre-computed audio logits: head invocation has to belong to the decoder.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openmoss/model.h"

namespace openmoss {

// One decoded frame: the (1 + n_codebooks) ids forming the next grid row.
struct FrameStep {
    std::vector<int32_t> ids;
    bool stop = false;
};

class IFrameDecoder {
public:
    virtual ~IFrameDecoder() = default;

    // Advance one frame.
    //   text_logits: (text_vocab_size,) from the backbone. Families that decide
    //                continuation with a dedicated head ignore this.
    //   hidden:      (hidden_size,) backbone last hidden state.
    // When the returned FrameStep has `stop` set, `ids` is not part of the
    // output — the frame is discarded.
    virtual FrameStep step(const float * text_logits,
                           const float * hidden,
                           const SamplingConfig & sc) = 0;

    // Audio codes accumulated so far, with pad/delay rows stripped.
    // Shape: (n_cb_out, t_audio_out) row-major.
    virtual std::vector<int32_t> extract_audio_codes(int32_t & n_cb_out,
                                                     int32_t & t_audio_out) const = 0;

    // Width of one grid row: 1 text column + n_codebooks audio columns.
    virtual int32_t row_width() const = 0;
};

// Per-family constructors. `prompt_rows` is the (S, row_width) prompt grid,
// which seeds any continuation state the family keeps.
std::unique_ptr<IFrameDecoder> make_delay_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows);

std::unique_ptr<IFrameDecoder> make_local_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows);

// Dispatch on model.dims().arch.
std::unique_ptr<IFrameDecoder> make_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows);

} // namespace openmoss
