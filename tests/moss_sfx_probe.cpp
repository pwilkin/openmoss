// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic: run MOSS-SoundEffect's DiT on a fixed latent / context / timestep
// and diff every seam against a PyTorch reference dump.
//
// Generated audio is useless for validating a diffusion model — 100 solver steps
// amplify any numerical difference — so this compares at seams instead, feeding
// the identical input bytes to both sides:
//
//   t          time embedding                 (dim,)
//   t_mod      6-way modulation               (dim, 6)
//   ctx_emb    projected text conditioning    (dim, n_text)
//   x_patched  latent after patchify          (dim, n_frames)
//   blk0_out   after the first block          (dim, n_frames)
//   blk29_out  after the last block           (dim, n_frames)
//   dit_out    velocity prediction            (out_dim, n_frames)
//
// A structural error (wrong RoPE flavour, swapped modulation slots, gated
// cross-attention, erf-GELU) shows up as an O(1) relative error. Correct wiring
// lands around 1e-3, which is the f16 weight storage plus ggml's f16-table GELU
// on the CPU backend — the reference itself runs in bf16 in production.
//
// usage: moss-sfx-probe <model.gguf> <ref_dir> [suffix]
//   Reads raw little-endian float32 dumps named <ref_dir>/<seam><suffix>.bin.
//   `suffix` selects an alternate case (e.g. "_t16" for the 16-frame one).
//
//   MOSS_AUX_CPU=1   rerun on the CPU backend, isolating a graph error from a
//                    backend numerics difference.
//   MOSS_SEAM_TOL=F  raise the per-seam budget, which is calibrated for an f16
//                    sidecar. A Q8_0 one sits around 5e-3 at every seam while
//                    leaving dit_out-dc unchanged.

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

// PyTorch keeps latents as [B, C, T] (time innermost); the DiT graph works
// channel-innermost. Both directions of that swap.
std::vector<float> transpose(const std::vector<float> & in, size_t rows, size_t cols) {
    std::vector<float> out(in.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            out[c * rows + r] = in[r * cols + c];
    return out;
}

// Subtract each channel's mean over frames. `v` is (n_ch, n_frames) with the
// channel innermost.
//
// The DiT's output is dominated by a per-channel DC offset of ±150: residual
// channel 421 grows to ~50x its neighbours by the last block, so the head's
// non-affine LayerNorm collapses every frame onto nearly the same unit
// direction. The per-frame signal riding on top is only ~2.5 std. A plain
// relative error against `dit_out` therefore reads ~1e-3 even if the actual
// frame-to-frame structure is destroyed, so it is checked demeaned as well.
std::vector<float> demean(const std::vector<float> & v, size_t n_ch, size_t n_frames) {
    std::vector<float> out = v;
    for (size_t c = 0; c < n_ch; ++c) {
        double sum = 0.0;
        for (size_t t = 0; t < n_frames; ++t) sum += double(v[t * n_ch + c]);
        const float mean = float(sum / double(n_frames));
        for (size_t t = 0; t < n_frames; ++t) out[t * n_ch + c] -= mean;
    }
    return out;
}

// Prints max |a-b| alongside the scale-free ‖a-b‖/‖b‖, and returns the latter.
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
    std::printf("  %-10s rel %.3e   max|d| %.3e at [%zu] (got %+.6f, ref %+.6f)\n",
                what, rel, max_abs, argmax, got[argmax], ref[argmax]);
    return rel;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: moss-sfx-probe <model.gguf> <ref_dir> [suffix]\n");
        return 2;
    }
    const std::string ref_dir = argv[2];
    const std::string suffix  = argc > 3 ? argv[3] : "";

    openmoss::LoadOptions opts;
    opts.n_gpu_layers = 0;      // the text encoder is not exercised here
    opts.skip_codec   = true;
    // MOSS_AUX_CPU=1 reruns the same graphs on the CPU backend. Comparing the
    // two isolates a graph error from a backend numerics difference.
    opts.aux_cpu      = std::getenv("MOSS_AUX_CPU") != nullptr;
    auto model = openmoss::Model::load(argv[1], opts);
    const auto & d = model->dims();
    if (d.arch != openmoss::Arch::SoundEffect) {
        std::fprintf(stderr, "not a moss_soundeffect model\n");
        return 1;
    }

    // The frame count is whatever the reference dumped; infer it from x_in.
    const size_t n_text = size_t(d.text_max_len);
    size_t n_frames = 0;
    {
        static const char * exts[] = { ".bin", ".f32", "" };
        for (const char * ext : exts) {
            std::ifstream f(ref_dir + "/x_in" + suffix + ext, std::ios::binary | std::ios::ate);
            if (f) { n_frames = size_t(f.tellg()) / sizeof(float) / size_t(d.dit_in_dim); break; }
        }
        if (!n_frames) { std::fprintf(stderr, "cannot size x_in\n"); return 1; }
    }
    std::printf("dit: dim=%d layers=%d heads=%d frames=%zu text=%zu\n",
                d.dit_dim, d.dit_n_layers, d.dit_n_heads, n_frames, n_text);

    const size_t dim = size_t(d.dit_dim);
    const size_t oc  = size_t(d.dit_out_dim);
    const size_t ic  = size_t(d.dit_in_dim);

    // [1, C, T] → channel-innermost.
    const auto x_ref   = read_f32(ref_dir, "x_in", suffix, ic * n_frames);
    const auto x_in    = transpose(x_ref, ic, n_frames);
    // [1, n_text, text_dim] is already channel-innermost — no transpose.
    const auto ctx_in  = read_f32(ref_dir, "context_in", suffix,
                                  size_t(d.dit_text_dim) * n_text);
    const auto ts      = read_f32(ref_dir, "timestep_in", suffix, 1);

    openmoss::DiTGraph dit(*model);

    std::vector<float> ctx_emb(dim * n_text);
    auto ctx = dit.encode_context(ctx_in.data(), int32_t(n_text), ctx_emb.data());

    std::vector<float> t(dim), t_mod(dim * 6), x_patched(dim * n_frames);
    std::vector<float> blk0(dim * n_frames), blk_last(dim * n_frames);
    std::vector<float> blk0_sa(dim * n_frames), blk0_ca(dim * n_frames);
    std::vector<float> velocity(oc * n_frames);

    openmoss::DiTTaps taps;
    taps.t         = t.data();
    taps.t_mod     = t_mod.data();
    taps.x_patched = x_patched.data();
    taps.blk_sa    = blk0_sa.data();
    taps.blk_ca    = blk0_ca.data();
    taps.blk_out   = blk0.data();
    taps.blk_index = 0;
    dit.forward(x_in.data(), int64_t(n_frames), ts[0], *ctx, velocity.data(), taps);

    // The last block needs a second pass: only one block can be tapped at a time.
    openmoss::DiTTaps taps_last;
    taps_last.blk_out   = blk_last.data();
    taps_last.blk_index = d.dit_n_layers - 1;
    std::vector<float> velocity2(oc * n_frames);
    dit.forward(x_in.data(), int64_t(n_frames), ts[0], *ctx, velocity2.data(), taps_last);

    std::printf("timestep %.6f\n", double(ts[0]));

    // Storing the weights as f16 costs ~2e-4 relative at every seam — measured by
    // rerunning the fp32 reference with its weights round-tripped through f16.
    // The error does not accumulate over the 30 blocks, so one budget covers all
    // of them; 1e-3 leaves room for the backends' own f16 intermediates.
    // A quantised sidecar legitimately exceeds this — Q8_0 lands around 5e-3 at
    // every seam — so raise it with MOSS_SEAM_TOL rather than reading a failure
    // as a porting bug. Note that the seams are where quantisation shows up and
    // `dit_out-dc` is where it does not: weight-rounding error is almost
    // entirely frame-constant, and the head passes that at ~0.8x while
    // amplifying frame-varying error ~50x.
    double SEAM_TOL = 1e-3;
    if (const char * env = std::getenv("MOSS_SEAM_TOL")) SEAM_TOL = std::atof(env);
    // The head is a different story. Its LayerNorm collapses every frame onto
    // nearly one direction, so whatever error arrives is amplified ~50x in the
    // demeaned output. 5e-2 is what the f16 budget turns into after that gain.
    const double HEAD_TOL = 5e-2;

    int failures = 0;
    auto check = [&](const char * what, double tol, const std::vector<float> & got,
                     const std::vector<float> & ref) {
        const double rel = compare(what, got, ref);
        if (!(rel < tol)) {
            std::printf("    FAIL: %s is %.3e, over its %.0e budget\n", what, rel, tol);
            ++failures;
        }
    };

    check("t",         SEAM_TOL, t,         read_f32(ref_dir, "t",         suffix, dim));
    check("t_mod",     SEAM_TOL, t_mod,     read_f32(ref_dir, "t_mod",     suffix, dim * 6));
    check("ctx_emb",   SEAM_TOL, ctx_emb,   read_f32(ref_dir, "ctx_emb",   suffix, dim * n_text));
    check("x_patched", SEAM_TOL, x_patched, read_f32(ref_dir, "x_patched", suffix, dim * n_frames));
    check("blk0_sa",   SEAM_TOL, blk0_sa,   read_f32(ref_dir, "blk0_sa_out", suffix, dim * n_frames));
    check("blk0_ca",   SEAM_TOL, blk0_ca,   read_f32(ref_dir, "blk0_ca_out", suffix, dim * n_frames));
    check("blk0_out",  SEAM_TOL, blk0,      read_f32(ref_dir, "blk0_out",  suffix, dim * n_frames));
    check("blk_last",  SEAM_TOL, blk_last,
          read_f32(ref_dir, "blk" + std::to_string(d.dit_n_layers - 1) + "_out",
                   suffix, dim * n_frames));
    {
        const auto ref = transpose(read_f32(ref_dir, "dit_out", suffix, oc * n_frames),
                                   oc, n_frames);
        compare("dit_out", velocity, ref);   // dominated by DC; reported, not gated
        check("dit_out-dc", HEAD_TOL, demean(velocity, oc, n_frames),
              demean(ref, oc, n_frames));
    }

    std::printf("%s\n", failures ? "FAILED" : "ok — every seam within its f16 budget");
    return failures ? 1 : 0;
}
