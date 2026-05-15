// SPDX-License-Identifier: Apache-2.0
//
// Codec round-trip smoke test: reads a WAV, encodes via codec_encode, then
// decodes via codec_decode, then writes the resulting WAV plus prints code
// statistics. Useful for catching regressions in either direction.

#include <chrono>
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

    auto wav = openmoss::read_wav_mono(in_wav, 24000);
    std::fprintf(stderr, "loaded %zu samples (%.2fs) from %s\n",
                 wav.size(), wav.size() / 24000.0, in_wav.c_str());

    int32_t n_vq = 0, T_audio = 0;
    auto t_enc0 = std::chrono::steady_clock::now();
    auto codes = openmoss::codec_encode(*model, wav.data(), int64_t(wav.size()),
                                          n_vq, T_audio);
    auto t_enc1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> enc_s = t_enc1 - t_enc0;
    std::fprintf(stderr, "encoded: n_vq=%d T_audio=%d (%.2fs of audio) in %.2fs\n",
                 n_vq, T_audio, T_audio * 1920.0 / 24000.0, enc_s.count());

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

    openmoss::write_wav_mono(out_wav, wav_out.data(), int64_t(wav_out.size()), 24000);
    std::fprintf(stderr, "wrote %s\n", out_wav.c_str());
    return 0;
}
