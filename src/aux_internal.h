// SPDX-License-Identifier: Apache-2.0
//
// Internal definition of Model::Aux. Shared between model.cpp (which owns it)
// and codec.cpp (which reaches in for the backend handle and tensor map).

#pragma once

#include "openmoss/model.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace openmoss {

struct Model::Aux {
    ggml_backend_t        backend = nullptr;
    ggml_context        * ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;

    // Tensors owned by us:
    //   - moss.audio_embed.{i}.weight     i in [0, n_vq)
    //   - moss.audio_head.{i}.weight      i in [0, n_vq)
    //   - moss.codec.*                    (lazy/optional)
    //   - "_text_embed"                   the Qwen3 input embedding table
    std::unordered_map<std::string, ggml_tensor *> tensors;
    ggml_tensor * text_embed = nullptr;

    bool codec_present = false;
    int32_t hidden_size      = 0;
    int32_t n_vq             = 0;
    int32_t audio_vocab_full = 0;
    int32_t text_vocab_size  = 0;

    ggml_gallocr_t galloc = nullptr;

    ~Aux() {
        if (galloc)  ggml_gallocr_free(galloc);
        if (buffer)  ggml_backend_buffer_free(buffer);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

} // namespace openmoss
