// SPDX-License-Identifier: Apache-2.0
//
// Codec round-trip smoke test: reads a WAV, encodes via codec_encode, then
// decodes via codec_decode, then writes the resulting WAV plus prints code
// statistics. Useful for catching regressions in either direction.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "openmoss/codec.h"
#include "openmoss/model.h"
#include "openmoss/wav.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: moss-codec-roundtrip <model.gguf> <input.wav> <output.wav>\n");
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string in_wav     = argv[2];
    const std::string out_wav    = argv[3];

    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;        // we don't actually use the LM here, save VRAM
    auto model = openmoss::Model::load(model_path, opts);

    const auto & d  = model->dims();
    const int32_t sr = d.sampling_rate;
    const int32_t ch = d.n_channels;

    auto wav = openmoss::read_wav(in_wav, sr, ch);
    std::fprintf(stderr, "loaded %zu samples (%.2fs, %d ch @ %d Hz) from %s\n",
                 wav.size(), double(wav.size()) / double(sr) / double(ch),
                 ch, sr, in_wav.c_str());

    int32_t n_vq = 0, T_audio = 0;
    auto t_enc0 = std::chrono::steady_clock::now();
    auto codes = openmoss::codec_encode(*model, wav.data(), int64_t(wav.size()),
                                          n_vq, T_audio);
    auto t_enc1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> enc_s = t_enc1 - t_enc0;
    std::fprintf(stderr, "encoded: n_vq=%d T_audio=%d (%.2fs of audio) in %.2fs\n",
                 n_vq, T_audio,
                 double(T_audio) * double(d.downsample_rate) / double(sr),
                 enc_s.count());

    // Quick sanity stats: per-codebook unique-count + min/max.
    for (int i = 0; i < n_vq; i += 8) {
        std::set<int32_t> uniq;
        int32_t mn = 1 << 30, mx = -1;
        for (int t = 0; t < T_audio; ++t) {
            int32_t v = codes[size_t(i) * size_t(T_audio) + size_t(t)];
            uniq.insert(v);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        std::fprintf(stderr, "  codebook %2d: %zu unique values  range=[%d, %d]\n",
                     i, uniq.size(), mn, mx);
    }

    auto t_dec0 = std::chrono::steady_clock::now();
    auto wav_out = openmoss::codec_decode(*model, codes.data(), n_vq, T_audio);
    auto t_dec1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dec_s = t_dec1 - t_dec0;
    std::fprintf(stderr, "decoded back %zu samples in %.2fs\n",
                 wav_out.size(), dec_s.count());

    openmoss::write_wav(out_wav, wav_out.data(), int64_t(wav_out.size()), sr, ch);
    std::fprintf(stderr, "wrote %s\n", out_wav.c_str());

    // Envelope correlation over the overlapping span, per channel. This is what
    // caught the missing sliding-window mask in issue #7: a broken mask degrades
    // gradually, so the correlation only falls apart past the context length.
    const size_t n_cmp = std::min(wav.size(), wav_out.size()) / size_t(ch);
    const size_t win   = size_t(sr) / 100;   // 10 ms envelope window
    for (int32_t c = 0; c < ch; ++c) {
        std::vector<double> ea, eb;
        for (size_t i = 0; i + win <= n_cmp; i += win) {
            double sa = 0, sb = 0;
            for (size_t j = 0; j < win; ++j) {
                sa += std::fabs(double(wav    [(i + j) * size_t(ch) + size_t(c)]));
                sb += std::fabs(double(wav_out[(i + j) * size_t(ch) + size_t(c)]));
            }
            ea.push_back(sa / double(win));
            eb.push_back(sb / double(win));
        }
        auto corr = [](const std::vector<double> & a, const std::vector<double> & b,
                       size_t lo, size_t hi) {
            if (hi <= lo) return 0.0;
            double ma = 0, mb = 0;
            for (size_t i = lo; i < hi; ++i) { ma += a[i]; mb += b[i]; }
            ma /= double(hi - lo); mb /= double(hi - lo);
            double num = 0, da = 0, db = 0;
            for (size_t i = lo; i < hi; ++i) {
                const double x = a[i] - ma, y = b[i] - mb;
                num += x * y; da += x * x; db += y * y;
            }
            return (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
        };
        const size_t half = ea.size() / 2;
        std::fprintf(stderr,
                     "  ch%d envelope correlation: overall %.4f | first half %.4f | second half %.4f\n",
                     c, corr(ea, eb, 0, ea.size()), corr(ea, eb, 0, half),
                     corr(ea, eb, half, ea.size()));
    }
    return 0;
}
