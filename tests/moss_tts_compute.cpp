// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for the GGML compute graphs: input embeddings + audio LM heads.
// Exercises the data path end-to-end against a real GGUF without running
// generation. Useful for catching regressions in the graph builder, the
// gallocr lifecycle, and the host↔device tensor transfers.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "openmoss/model.h"

static void check_finite(const std::vector<float> & v, const char * label) {
    int n_nan = 0, n_inf = 0;
    double s = 0.0;
    float mn = +1e9f, mx = -1e9f;
    for (float x : v) {
        if (std::isnan(x)) ++n_nan;
        else if (std::isinf(x)) ++n_inf;
        else {
            s += x;
            if (x < mn) mn = x;
            if (x > mx) mx = x;
        }
    }
    std::printf("  %s: size=%zu mean=%.6g min=%.6g max=%.6g  nan=%d inf=%d\n",
                label, v.size(), s / std::max<size_t>(v.size(), 1), mn, mx, n_nan, n_inf);
    if (n_nan || n_inf) {
        throw std::runtime_error(std::string(label) + ": non-finite values");
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: moss-tts-compute-test <gguf>\n");
        return 2;
    }
    openmoss::LoadOptions lo;
    lo.n_gpu_layers = 0;     // backbone on CPU; aux still on CUDA
    lo.flash_attn   = false;
    lo.n_ctx        = 256;   // tiny — we don't actually run the backbone

    auto m = openmoss::Model::load(argv[1], lo);
    const auto & d = m->dims();
    const int32_t n_vq    = d.n_vq;
    const int32_t hidden  = d.hidden_size;
    const int32_t Vfull   = d.audio_vocab_size + 1;
    const int32_t pad     = d.audio_pad_code;

    std::printf("\n=== compute_input_embeddings ===\n");
    // 4-position prompt grid: 1 col text + 32 cols audio (pad).
    const int32_t S = 4;
    std::vector<int32_t> grid(size_t(S) * size_t(1 + n_vq), pad);
    grid[0 * (1 + n_vq) + 0] = d.im_start_token_id;
    grid[1 * (1 + n_vq) + 0] = d.im_end_token_id;
    grid[2 * (1 + n_vq) + 0] = d.audio_start_token_id;
    grid[3 * (1 + n_vq) + 0] = d.audio_end_token_id;

    auto embs = m->compute_input_embeddings(grid.data(), S);
    check_finite(embs, "input_embeds");
    if (int(embs.size()) != S * hidden) {
        std::fprintf(stderr, "  unexpected size %zu (want %d)\n", embs.size(), S * hidden);
        return 1;
    }

    std::printf("\n=== compute_audio_logits ===\n");
    // Use the first prompt position's embedding as a synthetic hidden state
    // (it's a real hidden-sized vector, just not from the backbone — fine for
    // a graph plumbing test).
    auto logits = m->compute_audio_logits(embs.data() + 0 * hidden);
    check_finite(logits, "audio_logits");
    if (int(logits.size()) != n_vq * Vfull) {
        std::fprintf(stderr, "  unexpected size %zu (want %d)\n", logits.size(), n_vq * Vfull);
        return 1;
    }

    std::printf("\nAll graphs computed without errors.\n");
    return 0;
}
