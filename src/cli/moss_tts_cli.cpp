// SPDX-License-Identifier: Apache-2.0
//
// One-shot TTS CLI. Loads a GGUF, synthesizes one utterance, writes a WAV.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#include "openmoss/model.h"
#include "openmoss/pipeline.h"
#include "openmoss/wav.h"

namespace {

struct Args {
    std::string model_path;
    std::string text;
    std::string output_path = "out.wav";
    std::optional<std::string> reference;
    std::optional<std::string> instruction;
    std::optional<std::string> language;
    std::optional<int> tokens;
    int  n_gpu_layers = -1;
    int  main_gpu = -1;
    int  max_new_tokens = 4096;
    bool flash_attn = true;
    bool skip_codec = false;
    bool aux_cpu    = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(stderr,
        "Usage: moss-tts-cli --model <gguf> --text <text> [options]\n"
        "\n"
        "Required:\n"
        "  --model PATH           GGUF file produced by scripts/convert_hf_to_gguf.py\n"
        "  --text  STRING         Text to synthesize\n"
        "\n"
        "Optional:\n"
        "  --output PATH          Output WAV (default: out.wav)\n"
        "  --reference PATH       Reference WAV for voice cloning\n"
        "  --instruction STRING   Voice/style description (voice generation mode)\n"
        "  --language CODE        Language code hint (en/zh/...)\n"
        "  --tokens N             Approximate audio token count (1s ≈ 12.5 tokens)\n"
        "  --max-new-tokens N     Generation cap (default: 4096)\n"
        "  --n-gpu-layers N       Backbone GPU offload (default: all)\n"
        "  --main-gpu N           GPU device index to pin model to\n"
        "                         (default: auto, picks GPU with most free VRAM)\n"
        "  --no-flash-attn        Disable flash attention\n"
        "  --skip-codec           Don't load codec tensors (saves ~3.4 GB VRAM,\n"
        "                         disables waveform synthesis)\n"
        "  --aux-cpu              Force audio embeds + codec onto CPU (workaround\n"
        "                         for backends missing ops, e.g. Metal DIAG_MASK_INF).\n"
        "                         Backbone still uses the GPU.\n"
    );
    std::exit(code);
}

std::string require_str(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) usage(2);
    return argv[++i];
}
int require_int(int & i, int argc, char ** argv) {
    return std::atoi(require_str(i, argc, argv).c_str());
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--model")           a.model_path  = require_str(i, argc, argv);
        else if (k == "--text")            a.text        = require_str(i, argc, argv);
        else if (k == "--output")          a.output_path = require_str(i, argc, argv);
        else if (k == "--reference")       a.reference   = require_str(i, argc, argv);
        else if (k == "--instruction")     a.instruction = require_str(i, argc, argv);
        else if (k == "--language")        a.language    = require_str(i, argc, argv);
        else if (k == "--tokens")          a.tokens      = require_int(i, argc, argv);
        else if (k == "--max-new-tokens")  a.max_new_tokens = require_int(i, argc, argv);
        else if (k == "--n-gpu-layers")    a.n_gpu_layers = require_int(i, argc, argv);
        else if (k == "--main-gpu")        a.main_gpu     = require_int(i, argc, argv);
        else if (k == "--no-flash-attn")   a.flash_attn = false;
        else if (k == "--skip-codec")      a.skip_codec = true;
        else if (k == "--aux-cpu")         a.aux_cpu    = true;
        else if (k == "--help" || k == "-h") usage(0);
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(2); }
    }
    if (a.model_path.empty() || a.text.empty()) usage(2);

    openmoss::LoadOptions lo;
    lo.n_gpu_layers = a.n_gpu_layers;
    lo.main_gpu     = a.main_gpu;
    lo.flash_attn   = a.flash_attn;
    lo.skip_codec   = a.skip_codec;
    lo.aux_cpu      = a.aux_cpu;
    auto model = openmoss::Model::load(a.model_path, lo);

    openmoss::GenerateRequest req;
    req.text          = a.text;
    req.instruction   = a.instruction;
    req.language      = a.language;
    req.tokens        = a.tokens;
    req.max_new_tokens = a.max_new_tokens;
    if (a.reference) {
        req.reference_wav = openmoss::read_wav_mono(*a.reference, 24000);
    }

    auto result = openmoss::generate(*model, req);
    openmoss::write_wav_mono(a.output_path,
                             result.waveform.data(),
                             int64_t(result.waveform.size()),
                             24000);
    std::fprintf(stderr, "wrote %lld samples (%.2fs) to %s\n",
                 (long long)result.waveform.size(),
                 result.waveform.size() / 24000.0,
                 a.output_path.c_str());
    return 0;
}
