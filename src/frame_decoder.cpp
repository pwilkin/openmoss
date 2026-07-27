// SPDX-License-Identifier: Apache-2.0
//
// Frame-decoder dispatch. The per-family implementations live next to the
// machinery they wrap: DelayFrameDecoder in delay.cpp, LocalFrameDecoder in
// local_transformer.cpp.

#include "openmoss/frame_decoder.h"

#include <stdexcept>
#include <string>

namespace openmoss {

std::unique_ptr<IFrameDecoder> make_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows) {
    switch (model.dims().arch) {
        case Arch::TTSDelay: return make_delay_frame_decoder(model, prompt_rows);
        case Arch::TTSLocal: return make_local_frame_decoder(model, prompt_rows);
    }
    throw std::runtime_error(std::string("make_frame_decoder: unhandled architecture ")
                             + arch_name(model.dims().arch));
}

} // namespace openmoss
