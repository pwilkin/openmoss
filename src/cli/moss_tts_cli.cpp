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
    int  n_batch  = -1;
    int  n_ctx    = -1;
    int  max_new_tokens = 4096;
    // Sampling. Defaults are filled in from the model's family after load, so
    // "unset" has to be distinguishable from an explicit value.
    std::optional<float> temperature, top_p, text_temperature, text_top_p;
    std::optional<float> audio_temperature, audio_top_p, audio_repetition_penalty;
    std::optional<int>   top_k, text_top_k, audio_top_k;
    std::optional<uint64_t> seed;
    std::optional<std::string> dump_codes;
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
        "\n"
        "Sampling (defaults come from the model's family; 0 temperature = greedy):\n"
        "  --temperature F        Sets BOTH channels\n"
        "  --top-p F              Sets BOTH channels\n"
        "  --top-k N              Sets BOTH channels\n"
        "  --text-temperature F   Text/stop channel only (takes precedence)\n"
        "  --text-top-p F\n"
        "  --text-top-k N\n"
        "  --audio-temperature F  Audio codebook channel only (takes precedence)\n"
        "  --audio-top-p F\n"
        "  --audio-top-k N\n"
        "  --audio-repetition-penalty F\n"
        "  --seed N               Non-negative; omit for a random per-run seed\n"
        "  --dump-codes PATH      Write the generated (n_vq, T) code matrix as text\n"
        "  --n-batch N            libllama batch size (default: 512). Raise if a\n"
        "                         long prompt exceeds this size.\n"
        "  --n-ctx N              libllama context size (default: 8192)\n"
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
float require_float(int & i, int argc, char ** argv) {
    return float(std::atof(require_str(i, argc, argv).c_str()));
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
        else if (k == "--temperature")       a.temperature       = require_float(i, argc, argv);
        else if (k == "--top-p")             a.top_p             = require_float(i, argc, argv);
        else if (k == "--top-k")             a.top_k             = require_int(i, argc, argv);
        else if (k == "--text-temperature")  a.text_temperature  = require_float(i, argc, argv);
        else if (k == "--text-top-p")        a.text_top_p        = require_float(i, argc, argv);
        else if (k == "--text-top-k")        a.text_top_k        = require_int(i, argc, argv);
        else if (k == "--audio-temperature") a.audio_temperature = require_float(i, argc, argv);
        else if (k == "--audio-top-p")       a.audio_top_p       = require_float(i, argc, argv);
        else if (k == "--audio-top-k")       a.audio_top_k       = require_int(i, argc, argv);
        else if (k == "--audio-repetition-penalty")
                                             a.audio_repetition_penalty = require_float(i, argc, argv);
        else if (k == "--seed")              a.seed              = uint64_t(std::strtoull(require_str(i, argc, argv).c_str(), nullptr, 10));
        else if (k == "--dump-codes")        a.dump_codes        = require_str(i, argc, argv);
        else if (k == "--n-gpu-layers")    a.n_gpu_layers = require_int(i, argc, argv);
        else if (k == "--main-gpu")        a.main_gpu     = require_int(i, argc, argv);
        else if (k == "--n-batch")         a.n_batch      = require_int(i, argc, argv);
        else if (k == "--n-ctx")           a.n_ctx        = require_int(i, argc, argv);
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
    if (a.n_batch > 0) lo.n_batch = a.n_batch;
    if (a.n_ctx   > 0) lo.n_ctx   = a.n_ctx;
    lo.flash_attn   = a.flash_attn;
    lo.skip_codec   = a.skip_codec;
    lo.aux_cpu      = a.aux_cpu;
    auto model = openmoss::Model::load(a.model_path, lo);

    openmoss::GenerateRequest req;
    // Family defaults first, then the broad --temperature/--top-p/--top-k
    // aliases, then the per-channel flags, which win.
    req.sampling = openmoss::default_sampling(model->dims().arch);
    if (a.temperature) { req.sampling.text_temperature = *a.temperature;
                         req.sampling.audio_temperature = *a.temperature; }
    if (a.top_p)       { req.sampling.text_top_p = *a.top_p;
                         req.sampling.audio_top_p = *a.top_p; }
    if (a.top_k)       { req.sampling.text_top_k = *a.top_k;
                         req.sampling.audio_top_k = *a.top_k; }
    if (a.text_temperature)  req.sampling.text_temperature  = *a.text_temperature;
    if (a.text_top_p)        req.sampling.text_top_p        = *a.text_top_p;
    if (a.text_top_k)        req.sampling.text_top_k        = *a.text_top_k;
    if (a.audio_temperature) req.sampling.audio_temperature = *a.audio_temperature;
    if (a.audio_top_p)       req.sampling.audio_top_p       = *a.audio_top_p;
    if (a.audio_top_k)       req.sampling.audio_top_k       = *a.audio_top_k;
    if (a.audio_repetition_penalty)
        req.sampling.audio_repetition_penalty = *a.audio_repetition_penalty;
    if (a.seed) req.sampling.seed = *a.seed;

    req.text          = a.text;
    req.instruction   = a.instruction;
    req.language      = a.language;
    req.tokens        = a.tokens;
    req.max_new_tokens = a.max_new_tokens;
    // Sample rate and channel count come from the model, not a constant: the
    // delay family is 24 kHz mono, MOSS-TTS-Local is 48 kHz stereo.
    const auto & dims = model->dims();
    if (a.reference) {
        req.reference_wav = openmoss::read_wav(*a.reference, dims.sampling_rate,
                                               dims.n_channels);
    }

    auto result = openmoss::generate(*model, req);

    if (a.dump_codes) {
        FILE * f = std::fopen(a.dump_codes->c_str(), "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.dump_codes->c_str()); return 1; }
        for (int32_t cb = 0; cb < result.n_codebooks; ++cb) {
            for (int32_t t = 0; t < result.n_audio_frames; ++t) {
                std::fprintf(f, "%d%s",
                             result.audio_codes[size_t(cb) * size_t(result.n_audio_frames) + size_t(t)],
                             t + 1 == result.n_audio_frames ? "\n" : " ");
            }
        }
        std::fclose(f);
        std::fprintf(stderr, "wrote codes (%d x %d) to %s\n",
                     result.n_codebooks, result.n_audio_frames, a.dump_codes->c_str());
    }

    openmoss::write_wav(a.output_path,
                        result.waveform.data(),
                        int64_t(result.waveform.size()),
                        dims.sampling_rate,
                        result.n_channels);
    std::fprintf(stderr, "wrote %lld samples (%.2fs, %d ch @ %d Hz) to %s\n",
                 (long long)result.waveform.size(),
                 double(result.waveform.size())
                     / double(dims.sampling_rate) / double(result.n_channels),
                 result.n_channels, dims.sampling_rate,
                 a.output_path.c_str());
    return 0;
}
