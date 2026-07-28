// SPDX-License-Identifier: Apache-2.0
//
// Does the incremental codec decoder produce the same waveform as the batch one?
//
// This is the only check that matters for streaming. The per-stage K/V cache is
// supposed to make chunked decoding *exact* rather than approximate: every
// query sees the same keys, at the same RoPE positions, that a full decode would
// have given it. So the bar here is not "close enough to sound right" — it is
// the f16 floor.
//
// For scale, from moss-codec-causality on the same codes: a stateless chunk with
// 128 frames of history lands at a relative error of 1.144, larger than the
// signal itself, and needs 512 frames to reach 8.7e-3. Anything in that range
// here means the cache is not doing its job.
//
// The residual that does remain comes from attention *blocking*: the batch path
// splits its queries at ATTN_CHUNK boundaries, the stream splits them at chunk
// boundaries, and flash attention sums a different partition of the same terms.
// That is the same effect moss-codec-causality already measures at ~1e-3 for a
// prefix decode, and it does not grow with length.
//
// usage: moss-codec-stream <model.gguf> <codes.txt> [chunk ...]
//   codes.txt is the (n_vq, T) matrix that moss-tts-cli --dump-codes writes.

#include <algorithm>
#include <chrono>
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

using clock_t_  = std::chrono::steady_clock;
using seconds_t = std::chrono::duration<double>;

struct Diff {
    double rel = 0.0;
    double max = 0.0;
};

Diff compare(const std::vector<float> & a, size_t a_off,
             const std::vector<float> & b, size_t b_off, size_t n) {
    double se = 0.0, ref = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = double(a[a_off + i]) - double(b[b_off + i]);
        se += d * d;
        ref += double(b[b_off + i]) * double(b[b_off + i]);
        if (std::fabs(d) > mx) mx = std::fabs(d);
    }
    Diff out;
    out.rel = ref > 0 ? std::sqrt(se / ref) : 0.0;
    out.max = mx;
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: moss-codec-stream <model.gguf> <codes.txt> [chunk ...]\n");
        return 2;
    }
    openmoss::LoadOptions opts;
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

    std::vector<int32_t> codes(size_t(n_vq) * size_t(T));
    for (int32_t c = 0; c < n_vq; ++c) {
        if (int32_t(rows[size_t(c)].size()) != T) {
            std::fprintf(stderr, "ragged codes: row %d has %zu entries, expected %d\n",
                         c, rows[size_t(c)].size(), T);
            return 1;
        }
        for (int32_t t = 0; t < T; ++t) {
            codes[size_t(c) * size_t(T) + size_t(t)] = rows[size_t(c)][size_t(t)];
        }
    }

    const double secs = double(T) * double(d.downsample_rate) / double(d.sampling_rate);
    std::printf("codes: %d codebooks x %d frames (%.1fs), hop %lld samples/frame\n",
                n_vq, T, secs, (long long)hop);

    // ── Reference: one batch decode ────────────────────────────────────────
    auto t0 = clock_t_::now();
    const auto full = openmoss::codec_decode(*model, codes.data(), n_vq, T);
    const double batch_s = seconds_t(clock_t_::now() - t0).count();
    std::printf("batch decode: %zu samples in %.2fs (%.1fx real time)\n\n",
                full.size(), batch_s, secs / batch_s);

    std::vector<int32_t> chunks;
    for (int i = 3; i < argc; ++i) chunks.push_back(std::atoi(argv[i]));
    if (chunks.empty()) chunks = { 1, 4, 8, 32 };
    // The whole utterance in one push exercises the streaming graph with an
    // empty cache at every stage, so it must reproduce the batch decode
    // *exactly*. That separates the two things that could be wrong: if this
    // control drifts, the streaming graph is not the batch graph; if only the
    // chunked runs drift, the K/V cache is at fault.
    chunks.insert(chunks.begin(), T);

    int failures = 0;
    std::vector<double> push_s;

    for (int32_t chunk : chunks) {
        if (chunk <= 0) continue;
        const bool control = (chunk >= T);

        openmoss::CodecStreamDecoder stream(*model, n_vq);

        // Push one chunk at a time. Each push gets its own contiguous
        // (n_vq, chunk) slice, exactly as a caller draining an LM would build it.
        std::vector<float> out;
        out.reserve(full.size());
        std::vector<int32_t> slice;
        push_s.clear();
        auto t_stream = clock_t_::now();
        for (int32_t t0f = 0; t0f < T; t0f += chunk) {
            const int32_t n = std::min<int32_t>(chunk, T - t0f);
            slice.resize(size_t(n_vq) * size_t(n));
            for (int32_t c = 0; c < n_vq; ++c) {
                for (int32_t t = 0; t < n; ++t) {
                    slice[size_t(c) * size_t(n) + size_t(t)] =
                        codes[size_t(c) * size_t(T) + size_t(t0f + t)];
                }
            }
            auto t_push = clock_t_::now();
            auto part = stream.push(slice.data(), n);
            push_s.push_back(seconds_t(clock_t_::now() - t_push).count());
            out.insert(out.end(), part.begin(), part.end());
        }
        const double stream_s = seconds_t(clock_t_::now() - t_stream).count();

        if (out.size() != full.size()) {
            std::printf("chunk %-4d SAMPLE COUNT MISMATCH: %zu vs %zu\n",
                        chunk, out.size(), full.size());
            ++failures;
            continue;
        }

        // The first push pays for shader compilation and the allocator's first
        // reservation, so it says nothing about steady-state latency. Report it
        // separately from the median of the rest.
        std::vector<double> rest(push_s.begin() + (push_s.size() > 1 ? 1 : 0),
                                 push_s.end());
        std::sort(rest.begin(), rest.end());
        const double median = rest.empty() ? push_s.front() : rest[rest.size() / 2];
        const double chunk_secs = double(chunk) * double(d.downsample_rate)
                                / double(d.sampling_rate);

        const Diff all = compare(out, 0, full, 0, full.size());
        if (control) {
            std::printf("single push (%d frames) — control: rel %.3e  max|d| %.3e  %s\n\n",
                        T, all.rel, all.max,
                        all.max == 0.0 ? "(bit-identical to batch)" : "*** NOT EXACT ***");
            std::printf("%-8s %-9s %-9s %-11s %-11s %-8s %s\n",
                        "chunk", "1st push", "median", "rel", "max|d|", "total", "vs real time");
            std::printf("%-8s %-9s %-9s %-11s %-11s %-8s %s\n",
                        "-----", "--------", "------", "---", "------", "-----", "------------");
            // The control must be exact, not merely close.
            if (all.max != 0.0) ++failures;
            continue;
        }

        std::printf("%-8d %-9.3f %-9.3f %-11.3e %-11.3e %-8.2f %.1fx  (chunk = %.2fs audio)\n",
                    chunk, push_s.front(), median, all.rel, all.max, stream_s,
                    secs / stream_s, chunk_secs);
        // A working cache lands near the f16 floor. 1e-2 is far above that and
        // far below the 1.144 a stateless chunk of this size would give, so it
        // separates "exact" from "broken" without being sensitive to blocking.
        if (!(all.rel < 1e-2)) ++failures;
    }

    // ── Where the error sits along the utterance ───────────────────────────
    //
    // A cache that is too short does not fail uniformly: the first chunk is
    // always right (nothing to remember yet) and the error appears once the
    // window fills. Reporting per-chunk error separates that from a flat
    // numerical offset.
    if (T > 16) {
        const int32_t chunk = 8;
        openmoss::CodecStreamDecoder stream(*model, n_vq);
        std::printf("\nper-chunk error at chunk=%d (frame range → rel):\n", chunk);
        std::vector<int32_t> slice;
        int32_t reported = 0;
        for (int32_t t0f = 0; t0f < T; t0f += chunk) {
            const int32_t n = std::min<int32_t>(chunk, T - t0f);
            slice.resize(size_t(n_vq) * size_t(n));
            for (int32_t c = 0; c < n_vq; ++c) {
                for (int32_t t = 0; t < n; ++t) {
                    slice[size_t(c) * size_t(n) + size_t(t)] =
                        codes[size_t(c) * size_t(T) + size_t(t0f + t)];
                }
            }
            const auto part = stream.push(slice.data(), n);
            const Diff dd = compare(part, 0, full, size_t(int64_t(t0f) * hop),
                                    part.size());
            // Print the first few and then every 16th chunk — enough to see a
            // trend without a wall of output.
            if (reported < 6 || (reported % 16) == 0) {
                std::printf("  frames %5d..%-5d  rel %.3e  max|d| %.3e\n",
                            t0f, t0f + n, dd.rel, dd.max);
            }
            ++reported;
        }
    }

    if (failures) {
        std::printf("\nFAILED: %d configuration(s) exceeded the tolerance\n", failures);
        return 1;
    }
    std::printf("\nOK\n");
    return 0;
}
