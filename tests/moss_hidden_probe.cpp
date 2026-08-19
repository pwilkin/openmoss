// SPDX-License-Identifier: Apache-2.0
//
// In-house diagnostic: prefill the delay-family prompt grid for a text-only
// request and dump (a) the last-position hidden state libllama hands back and
// (b) the audio-head logits our aux graph computes from it. Comparing both
// against the PyTorch reference on the same grid splits a backbone divergence
// from an aux-graph divergence in one run.
//
// usage: moss-hidden-probe <model.gguf> <text>
// output (stdout):
//   HIDDEN <h0> <h1> ... <h{hidden-1}>
//   TEXTTOP <id:logit> x5
//   CH<i> top-10 as <id:logit> pairs, one line per codebook

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "llama.h"
#include "openmoss/model.h"
#include "openmoss/pipeline.h"

static void top_line(const char * label, const float * v, int n, int k) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    std::printf("%s", label);
    for (int i = 0; i < k; ++i) std::printf(" %d:%.4f", idx[i], v[idx[i]]);
    std::printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-hidden-probe <model.gguf> <text> [fa=0|1]\n");
        return 2;
    }
    openmoss::LoadOptions lo;
    lo.n_gpu_layers = 0;      // CPU backbone: this is a numeric probe, not a perf test
    lo.aux_cpu      = true;
    lo.skip_codec   = true;
    // Optional third arg "fa=0|1" toggles flash attention (default on, matching
    // the CLI) so its numeric effect can be isolated.
    for (int ai = 3; ai < argc; ++ai) {
        if (std::strncmp(argv[ai], "fa=", 3) == 0) lo.flash_attn = (argv[ai][3] == '1');
        if (std::strcmp(argv[ai], "kv=f32") == 0)  lo.kv_f32 = true;
        // gpu=N: run the backbone on Vulkan device N instead of CPU, so the
        // Vulkan numeric path (f16 vs f32 accumulation) can be measured.
        if (std::strncmp(argv[ai], "gpu=", 4) == 0) {
            lo.n_gpu_layers = -1;
            lo.main_gpu     = std::atoi(argv[ai] + 4);
        }
        if (std::strcmp(argv[ai], "vkf32") == 0) setenv("GGML_VK_DISABLE_F16", "1", 0);
    }
    std::fprintf(stderr, "[hidden-probe] flash_attn=%d\n", lo.flash_attn ? 1 : 0);
    auto model = openmoss::Model::load(argv[1], lo);
    const auto & d = model->dims();

    openmoss::GenerateRequest req;
    req.text = argv[2];       // template default = VoiceGen, matching the probe/parity setup

    int32_t n_pos = 0;
    auto grid = openmoss::debug_build_prompt_grid(*model, req, nullptr, 0, n_pos);
    std::fprintf(stderr, "[hidden-probe] grid: %d rows\n", n_pos);

    // Optional "rows=N" argument: prefill only the first N grid rows. rows=1 is
    // the sharpest bisect — a single token has no cross-position attention, so
    // a divergence there cannot be rope/mask/attention mechanics.
    for (int ai = 3; ai < argc; ++ai) {
        if (std::strncmp(argv[ai], "rows=", 5) == 0) {
            int32_t want = std::atoi(argv[ai] + 5);
            if (want > 0 && want < n_pos) n_pos = want;
        }
    }
    std::fprintf(stderr, "[hidden-probe] using %d rows\n", n_pos);

    const int32_t hidden = d.hidden_size;
    auto embds = model->compute_input_embeddings(grid.data(), n_pos);

    // Dump the last row's input embedding (the audio_start row): norm + head,
    // to compare against the reference's summed text+audio embedding.
    {
        const float * e = embds.data() + size_t(n_pos - 1) * size_t(hidden);
        double nrm = 0.0;
        for (int i = 0; i < hidden; ++i) nrm += double(e[i]) * double(e[i]);
        std::printf("EMBLAST norm=%.6f head", std::sqrt(nrm));
        for (int i = 0; i < 8; ++i) std::printf(" %.6g", e[i]);
        std::printf("\n");
    }

    // Prefill exactly like generate() does: one batch, only the last row
    // produces logits/embeddings.
    llama_context * ctx = model->backbone_ctx();
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_pos, hidden, 1);
    batch.n_tokens = n_pos;
    std::memcpy(batch.embd, embds.data(), size_t(n_pos) * size_t(hidden) * sizeof(float));
    for (int32_t i = 0; i < n_pos; ++i) {
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_pos - 1) ? 1 : 0;
    }
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed\n");
        return 1;
    }
    llama_batch_free(batch);

    const float * text_logits = llama_get_logits_ith(ctx, -1);
    const float * hvec        = llama_get_embeddings_ith(ctx, -1);
    if (!text_logits || !hvec) {
        std::fprintf(stderr, "no logits/embeddings\n");
        return 1;
    }

    double nrm = 0.0;
    for (int i = 0; i < hidden; ++i) nrm += double(hvec[i]) * double(hvec[i]);
    std::fprintf(stderr, "[hidden-probe] hidden norm = %.4f\n", std::sqrt(nrm));

    std::printf("HIDDEN");
    for (int i = 0; i < hidden; ++i) std::printf(" %.6g", hvec[i]);
    std::printf("\n");

    top_line("TEXTTOP", text_logits, d.text_vocab_size, 5);

    auto alog = model->compute_audio_logits(hvec);
    const int32_t Vfull = d.audio_vocab_size + 1;
    for (int32_t i = 0; i < d.n_vq; ++i) {
        char label[16];
        std::snprintf(label, sizeof(label), "CH%d", i);
        // Mask the pad code the way generation does before ranking.
        std::vector<float> row(alog.begin() + size_t(i) * Vfull,
                               alog.begin() + size_t(i + 1) * Vfull);
        row[d.audio_pad_code] = -1e30f;
        top_line(label, row.data(), Vfull, 10);
    }
    return 0;
}
