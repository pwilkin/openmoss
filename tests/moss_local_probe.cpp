// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic: drive MOSS-TTS-Local's local ("depth") transformer from a fixed
// backbone hidden state and print the 12 codes it produces greedily.
//
// Feeding *the same* hidden state to this and to the reference PyTorch model
// isolates the depth transformer from any backbone precision difference, so a
// mismatch points squarely at the local block — the RoPE flavour (interleaved
// vs NeoX), LayerNorm-vs-RMSNorm, the un-gated SiLU MLP, or the tied heads.
//
// usage: moss-local-probe <model.gguf> <hidden.f32>
//   hidden.f32 is a raw little-endian float32 vector of length hidden_size.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "openmoss/frame_decoder.h"
#include "openmoss/model.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-local-probe <model.gguf> <hidden.f32>\n");
        return 2;
    }
    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;      // the backbone is unused here
    opts.skip_codec   = true;   // and so is the codec
    auto model = openmoss::Model::load(argv[1], opts);
    const auto & d = model->dims();

    std::vector<float> hidden(size_t(d.hidden_size));
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        f.read(reinterpret_cast<char *>(hidden.data()),
               std::streamsize(hidden.size() * sizeof(float)));
        if (size_t(f.gcount()) != hidden.size() * sizeof(float)) {
            std::fprintf(stderr, "expected %zu floats in %s\n", hidden.size(), argv[2]);
            return 1;
        }
    }

    auto dec = openmoss::make_frame_decoder(*model, {});

    openmoss::SamplingConfig sc;
    sc.text_temperature  = 0.0f;   // greedy
    sc.audio_temperature = 0.0f;   // greedy
    sc.audio_repetition_penalty = 1.0f;
    // Force the frame to be emitted even when the stop head says end-of-speech:
    // we want the codes for comparison. The stop head shares micro-step 0's
    // output with codebook 0, so matching codes implies a matching u.
    sc.min_audio_frames = 1;

    auto step = dec->step(/*text_logits=*/nullptr, hidden.data(), sc);
    if (step.stop) { std::printf("STOP\n"); return 0; }

    for (size_t i = 1; i < step.ids.size(); ++i) {
        std::printf("%d%s", step.ids[i], i + 1 == step.ids.size() ? "\n" : " ");
    }
    return 0;
}
