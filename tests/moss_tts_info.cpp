// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic tool: load a GGUF and print what was found.
// Useful for validating the converter end-to-end without running full generation.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "openmoss/model.h"

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: moss-tts-info <gguf>\n");
        return 2;
    }
    openmoss::LoadOptions lo;
    lo.n_gpu_layers = 0;   // pure CPU — keeps the test cheap
    lo.flash_attn   = false;

    auto m = openmoss::Model::load(argv[1], lo);
    const auto & d = m->dims();
    std::printf("\n--- model loaded ---\n");
    std::printf("  hidden_size              %d\n", d.hidden_size);
    std::printf("  n_vq                     %d\n", d.n_vq);
    std::printf("  audio_vocab_size         %d\n", d.audio_vocab_size);
    std::printf("  audio_pad_code           %d\n", d.audio_pad_code);
    std::printf("  sampling_rate            %d Hz\n", d.sampling_rate);
    std::printf("  downsample_rate          %d\n", d.downsample_rate);
    std::printf("  audio_start_token        %d\n", d.audio_start_token_id);
    std::printf("  audio_end_token          %d\n", d.audio_end_token_id);
    std::printf("  audio_user_slot_token    %d\n", d.audio_user_slot_token_id);
    std::printf("  audio_gen_slot_token     %d\n", d.audio_assistant_gen_slot_token_id);
    std::printf("  audio_delay_slot_token   %d\n", d.audio_assistant_delay_slot_token_id);
    std::printf("  im_start / im_end        %d / %d\n", d.im_start_token_id, d.im_end_token_id);
    std::printf("  audio embeddings loaded  %d (expected %d)\n",
                m->n_audio_embed_loaded(), d.n_vq);
    std::printf("  audio heads loaded       %d (expected %d)\n",
                m->n_audio_head_loaded(), d.n_vq);
    std::printf("  codec present            %s\n", m->codec_present() ? "yes" : "no");
    std::printf("---\n");
    return 0;
}
