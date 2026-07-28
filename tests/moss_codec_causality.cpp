// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic: is the codec decoder exactly causal, and how much left context
// does a chunk actually need?
//
// Both questions decide whether streaming is possible at all. The decoder's
// attention is banded — `(delta >= 0) & (delta < context)` — so in principle
// output frame t depends only on inputs at or before t. This checks that
// empirically, because "in principle" has been wrong before.
//
// Test 1 (causality): decode a prefix of the codes and compare against the same
//   prefix of a full decode. Exact agreement means the decoder never looks
//   forward, so a stream can emit audio as codes arrive.
//
// Test 2 (left context): decode frames [N-W, N) with W frames of history and
//   compare the tail against the full decode. The smallest W that agrees is the
//   history a stateless chunked decoder must carry. The per-stage windows sum
//   to 35 s, so the expectation is that W is large — which is what makes a
//   stateless decoder impractical and a KV cache necessary.
//
// usage: moss-codec-causality <model.gguf> <codes.txt>
//   codes.txt is the (n_vq, T) matrix that moss-tts-cli --dump-codes writes.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "openmoss/codec.h"
#include "openmoss/model.h"

namespace {

// Relative error over a sample range, plus the largest single deviation.
void compare(const char * what, const std::vector<float> & a, size_t a_off,
             const std::vector<float> & b, size_t b_off, size_t n) {
    double se = 0.0, ref = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = double(a[a_off + i]) - double(b[b_off + i]);
        se += d * d; ref += double(b[b_off + i]) * double(b[b_off + i]);
        if (std::fabs(d) > mx) mx = std::fabs(d);
    }
    std::printf("  %-28s rel %.3e  max|d| %.3e  %s\n", what,
                ref > 0 ? std::sqrt(se / ref) : 0.0, mx,
                mx == 0.0 ? "(bit-identical)" : "");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-codec-causality <model.gguf> <codes.txt>\n");
        return 2;
    }
    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;
    auto model = openmoss::Model::load(argv[1], opts);
    const auto & d = model->dims();

    std::vector<std::vector<int32_t>> rows;
    {
        std::ifstream f(argv[2]);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::vector<int32_t> r; int32_t v;
            while (is >> v) r.push_back(v);
            if (!r.empty()) rows.push_back(std::move(r));
        }
    }
    if (rows.empty()) { std::fprintf(stderr, "no codes\n"); return 1; }
    const int32_t n_vq = int32_t(rows.size());
    const int32_t T    = int32_t(rows[0].size());
    const int64_t hop  = d.samples_per_frame();

    // Flatten to (n_vq, T) row-major, and provide a prefix slicer.
    auto slice = [&](int32_t t0, int32_t t1) {
        std::vector<int32_t> out(size_t(n_vq) * size_t(t1 - t0));
        for (int32_t c = 0; c < n_vq; ++c)
            for (int32_t t = t0; t < t1; ++t)
                out[size_t(c) * size_t(t1 - t0) + size_t(t - t0)] = rows[size_t(c)][size_t(t)];
        return out;
    };

    std::printf("codes: %d codebooks x %d frames (%.1fs), hop %lld samples/frame\n",
                n_vq, T, double(T) * double(d.downsample_rate) / double(d.sampling_rate),
                (long long)hop);

    const auto full_codes = slice(0, T);
    const auto full = openmoss::codec_decode(*model, full_codes.data(), n_vq, T);
    std::printf("full decode: %zu samples\n\n", full.size());

    // ── Test 1: is a prefix decode the prefix of a full decode? ────────────
    std::printf("causality — decode(codes[:M]) vs decode(codes)[:M*hop]:\n");
    for (int32_t M : { T / 4, T / 2, (3 * T) / 4 }) {
        if (M <= 0) continue;
        const auto pc = slice(0, M);
        const auto p  = openmoss::codec_decode(*model, pc.data(), n_vq, M);
        char label[64];
        std::snprintf(label, sizeof(label), "M = %d of %d", M, T);
        compare(label, p, 0, full, 0, size_t(int64_t(M) * hop));
    }

    // ── Test 2: how much history does the tail need? ───────────────────────
    std::printf("\nleft context — decode(codes[N-W:N]) tail vs full decode tail:\n");
    const int32_t N = T;
    for (int32_t W : { 8, 16, 32, 64, 128, 256, 512 }) {
        if (W >= N) break;
        const auto wc = slice(N - W, N);
        const auto w  = openmoss::codec_decode(*model, wc.data(), n_vq, W);
        // Compare only the last 8 frames, which is the audio a chunk of this
        // width would actually emit.
        const int32_t emit = 8;
        char label[64];
        std::snprintf(label, sizeof(label), "W = %d frames (%.1fs)", W,
                      double(W) * double(d.downsample_rate) / double(d.sampling_rate));
        compare(label,
                w,    size_t(int64_t(W - emit) * hop),
                full, size_t(int64_t(N - emit) * hop),
                size_t(int64_t(emit) * hop));
    }
    return 0;
}
