// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic: dump the prompt grid for a request, so it can be diffed against
// the reference processor's own output.
//
// Prompt construction is the one part of the pipeline with no numerical
// tolerance — an id is right or it is not — and it is also the easiest to get
// subtly wrong, because the reference tokenizes segment by segment and one-pass
// encoding silently merges tokens across seams. Comparing ids needs no model
// weights, which makes this the cheapest check in the repo.
//
// usage: moss-prompt-probe <model.gguf> <text> [ref_codes.txt] [ref_text]
//   ref_codes.txt is the (n_vq, T) whitespace-separated matrix that
//   --dump-codes writes. Supplying ref_text as well selects continuation mode.
//
// Prints one line per grid row: "<pos> <text_id> <code0> <code1> …".

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "openmoss/model.h"
#include "openmoss/pipeline.h"

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: moss-prompt-probe <model.gguf> <text> [ref_codes.txt] [ref_text]\n");
        return 2;
    }
    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;     // no generation happens here
    opts.skip_codec   = true;
    auto model = openmoss::Model::load(argv[1], opts);

    openmoss::GenerateRequest req;
    req.text = argv[2];

    // The codec is skipped, so reference *codes* are read directly rather than
    // encoded from a WAV — that keeps this probe independent of the codec.
    std::vector<int32_t> ref_codes;
    int32_t T_ref = 0;
    if (argc > 3 && argv[3][0]) {
        std::ifstream f(argv[3]);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
        std::vector<std::vector<int32_t>> rows;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::vector<int32_t> row;
            int32_t v;
            while (is >> v) row.push_back(v);
            if (!row.empty()) rows.push_back(std::move(row));
        }
        if (rows.empty()) { std::fprintf(stderr, "%s is empty\n", argv[3]); return 1; }
        T_ref = int32_t(rows[0].size());
        for (const auto & r : rows) {
            if (int32_t(r.size()) != T_ref) {
                std::fprintf(stderr, "ragged code matrix in %s\n", argv[3]);
                return 1;
            }
            ref_codes.insert(ref_codes.end(), r.begin(), r.end());
        }
        std::fprintf(stderr, "[probe] reference: %zu codebooks x %d frames\n",
                     rows.size(), T_ref);
    }
    if (argc > 4 && argv[4][0]) req.ref_text = argv[4];

    int32_t n_pos = 0;
    auto grid = openmoss::debug_build_prompt_grid(
        *model, req, T_ref > 0 ? &ref_codes : nullptr, T_ref, n_pos);

    const int32_t cols = 1 + model->dims().n_vq;
    std::fprintf(stderr, "[probe] %d rows x %d cols%s\n", n_pos, cols,
                 req.ref_text ? " (continuation)" : "");
    for (int32_t r = 0; r < n_pos; ++r) {
        std::printf("%d", grid[size_t(r) * size_t(cols)]);
        for (int32_t c = 1; c < cols; ++c) {
            std::printf(" %d", grid[size_t(r) * size_t(cols) + size_t(c)]);
        }
        std::printf("\n");
    }
    return 0;
}
