// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic: run the DAC VAE decoder on a fixed latent and diff every seam
// against a PyTorch reference dump.
//
//   post_quant  after the kernel-1 post-quant conv   (T, latent_dim)
//   dec_in      after the k=7 input conv             (T, 2048)
//   blk1_up     after block 1's Snake + transpose    (8T, 1024)
//   blk1_res0   after block 1's first residual unit  (8T, 1024)
//   blk1..5_out after each upsampling block
//   audio       the waveform, after tanh             (960T,)
//
// `blk1_up` is the one that matters most: ggml's conv_transpose_1d refuses
// padding, so the port runs it unpadded and crops afterwards, and an off-by-one
// there shifts everything downstream by a sample without changing its shape.
//
// The reference decodes in fp32, so the only expected difference is f16 weight
// storage — the sidecar's convolution kernels — which is worth a few times 1e-4.
// Snake's alphas and their reciprocals are f32 on both sides by construction.
//
// usage: moss-sfx-vae-probe <model.gguf> <ref_vae_dir> [suffix]
//   MOSS_AUX_CPU=1 reruns on the CPU backend, which separates a graph error
//   from a backend numerics difference.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "openmoss/model.h"
#include "openmoss/soundeffect.h"

namespace {

std::vector<float> read_f32(const std::string & dir, const std::string & name,
                            const std::string & suffix, size_t expect) {
    static const char * exts[] = { ".bin", ".f32", "" };
    for (const char * ext : exts) {
        const std::string path = dir + "/" + name + suffix + ext;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) continue;
        const std::streamoff bytes = f.tellg();
        if (size_t(bytes) != expect * sizeof(float)) {
            std::fprintf(stderr, "%s: expected %zu floats, file holds %lld\n",
                         path.c_str(), expect, (long long)(bytes / 4));
            std::exit(1);
        }
        f.seekg(0);
        std::vector<float> out(expect);
        f.read(reinterpret_cast<char *>(out.data()), std::streamsize(expect * sizeof(float)));
        return out;
    }
    std::fprintf(stderr, "cannot open %s/%s%s{.bin,.f32,}\n",
                 dir.c_str(), name.c_str(), suffix.c_str());
    std::exit(1);
}

double compare(const char * what, const std::vector<float> & got,
               const std::vector<float> & ref) {
    double max_abs = 0.0, se = 0.0, ref_se = 0.0;
    size_t argmax = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = double(got[i]) - double(ref[i]);
        if (std::fabs(d) > max_abs) { max_abs = std::fabs(d); argmax = i; }
        se     += d * d;
        ref_se += double(ref[i]) * double(ref[i]);
    }
    const double rel = ref_se > 0.0 ? std::sqrt(se / ref_se) : 0.0;
    std::printf("  %-11s rel %.3e   max|d| %.3e at [%zu] (got %+.6f, ref %+.6f)\n",
                what, rel, max_abs, argmax, got[argmax], ref[argmax]);
    return rel;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-sfx-vae-probe <model.gguf> <ref_vae_dir> [suffix]\n");
        return 2;
    }
    const std::string ref_dir = argv[2];
    const std::string suffix  = argc > 3 ? argv[3] : "";

    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;      // the text encoder is not exercised here
    opts.skip_codec   = true;
    opts.aux_cpu      = std::getenv("MOSS_AUX_CPU") != nullptr;
    auto model = openmoss::Model::load(argv[1], opts);
    const auto & d = model->dims();
    if (d.arch != openmoss::Arch::SoundEffect) {
        std::fprintf(stderr, "not a moss_soundeffect model\n");
        return 1;
    }

    const size_t C   = size_t(d.vae_latent_dim);
    const size_t hop = size_t(d.vae_hop);
    size_t T = 0;
    {
        static const char * exts[] = { ".bin", ".f32", "" };
        for (const char * ext : exts) {
            std::ifstream f(ref_dir + "/z_in" + suffix + ext, std::ios::binary | std::ios::ate);
            if (f) { T = size_t(f.tellg()) / sizeof(float) / C; break; }
        }
        if (!T) { std::fprintf(stderr, "cannot size z_in\n"); return 1; }
    }

    // Channel widths per block, halving from `channels` down to 64.
    std::vector<size_t> blk_ch, blk_len;
    {
        size_t ch = 2048, len = T;
        for (size_t i = 0; i < d.vae_decoder_rates.size(); ++i) {
            ch /= 2;
            len *= size_t(d.vae_decoder_rates[i]);
            blk_ch.push_back(ch);
            blk_len.push_back(len);
        }
    }
    std::printf("vae: latent=%zu hop=%zu frames=%zu -> %zu samples\n", C, hop, T, T * hop);

    // The reference stores [1, C, T] with time innermost — which is exactly the
    // decoder's internal layout, so seams compare with no transpose. The public
    // decode() entry point takes the DiT's channel-innermost latent, though, so
    // the input alone has to be flipped.
    const auto z_ref = read_f32(ref_dir, "z_in", suffix, C * T);
    std::vector<float> z_in(C * T);
    for (size_t c = 0; c < C; ++c)
        for (size_t t = 0; t < T; ++t)
            z_in[t * C + c] = z_ref[c * T + t];

    openmoss::DacDecoder vae(*model);
    vae.set_chunk_frames(int32_t(T));   // taps need the whole decode in one chunk

    std::vector<float> post_quant(C * T), dec_in(2048 * T);
    std::vector<float> blk_up(blk_ch[0] * blk_len[0]);
    std::vector<float> blk_res0(blk_ch[0] * blk_len[0]);
    std::vector<float> blk_out(blk_ch[0] * blk_len[0]);

    openmoss::DacTaps taps;
    taps.post_quant = post_quant.data();
    taps.dec_in     = dec_in.data();
    taps.blk_up     = blk_up.data();
    taps.blk_res0   = blk_res0.data();
    taps.blk_out    = blk_out.data();
    taps.blk_index  = 0;                // block 1 upstream
    auto audio = vae.decode(z_in.data(), int64_t(T), taps);

    // Only one block can be tapped per decode, so the remaining four each cost
    // their own pass. Worth it: which block first goes out of budget says
    // whether a transposed conv's crop or a residual unit's dilation is wrong.
    std::vector<std::vector<float>> later_out(blk_ch.size());
    for (size_t b = 1; b < blk_ch.size(); ++b) {
        later_out[b].resize(blk_ch[b] * blk_len[b]);
        openmoss::DacTaps t2;
        t2.blk_out   = later_out[b].data();
        t2.blk_index = int(b);
        vae.decode(z_in.data(), int64_t(T), t2);
    }

    // Storing the conv kernels as f16 is the only expected difference; a few
    // times 1e-4 covers it, with room for the backends' own intermediates.
    constexpr double TOL = 2e-3;
    int failures = 0;
    auto check = [&](const char * what, const std::vector<float> & got,
                     const std::vector<float> & ref) {
        const double rel = compare(what, got, ref);
        if (!(rel < TOL)) {
            std::printf("    FAIL: %s is %.3e, over its %.0e budget\n", what, rel, TOL);
            ++failures;
        }
    };

    check("post_quant", post_quant, read_f32(ref_dir, "post_quant", suffix, C * T));
    check("dec_in",     dec_in,     read_f32(ref_dir, "dec_in",     suffix, 2048 * T));
    check("blk1_up",    blk_up,     read_f32(ref_dir, "blk1_up",    suffix,
                                             blk_ch[0] * blk_len[0]));
    check("blk1_res0",  blk_res0,   read_f32(ref_dir, "blk1_res0",  suffix,
                                             blk_ch[0] * blk_len[0]));
    check("blk1_out",   blk_out,    read_f32(ref_dir, "blk1_out",   suffix,
                                             blk_ch[0] * blk_len[0]));
    for (size_t b = 1; b < blk_ch.size(); ++b) {
        const std::string name = "blk" + std::to_string(b + 1) + "_out";
        check(name.c_str(), later_out[b],
              read_f32(ref_dir, name, suffix, blk_ch[b] * blk_len[b]));
    }
    check("audio",      audio,      read_f32(ref_dir, "audio",      suffix, T * hop));

    // Chunking is only exercised when the decode spans more than one chunk, and
    // a too-narrow overlap shows up as a seam every chunk_frames*hop samples.
    // Re-decode chunked and diff against the single-shot result: they should
    // agree to f32 round-off, since the two differ only in graph partitioning.
    if (T > 8) {
        vae.set_chunk_frames(int32_t(T / 4));
        const auto chunked = vae.decode(z_in.data(), int64_t(T));
        const double rel = compare("chunked", chunked, audio);
        if (!(rel < 1e-5)) {
            std::printf("    FAIL: chunk overlap is too narrow (%.3e)\n", rel);
            ++failures;
        }
    }

    std::printf("%s\n", failures ? "FAILED" : "ok — every seam within its f16 budget");
    return failures ? 1 : 0;
}
