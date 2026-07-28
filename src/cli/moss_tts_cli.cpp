// SPDX-License-Identifier: Apache-2.0
//
// One-shot TTS CLI. Loads a GGUF, synthesizes one utterance, writes a WAV.

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <string>

#include "openmoss/model.h"
#include "openmoss/pipeline.h"
#include "openmoss/soundeffect.h"
#include "openmoss/wav.h"

#include <fstream>
#include <vector>

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
    std::optional<int>   min_audio_frames, max_audio_frames;
    std::optional<uint64_t> seed;
    std::optional<std::string> dump_codes;
    bool flash_attn = true;
    bool skip_codec = false;
    bool aux_cpu    = false;
    std::optional<int>         stream_chunk;   // frames per incremental decode
    std::optional<std::string> stream_out;     // raw s16le sink; "-" = stdout

    // MOSS-SoundEffect only. It is not autoregressive, so none of the sampling
    // flags above apply; these drive the flow-matching solver instead.
    std::optional<float> seconds, cfg_scale, sigma_shift;
    std::optional<int>   steps;
    std::optional<std::string> negative_prompt;
    std::optional<std::string> dump_latent, init_latent, dump_context;
    bool no_duration_suffix = false;
    bool vk_f32 = false;
};

[[noreturn]] void usage(int code) {
    std::fprintf(stderr,
        "Usage: moss-tts-cli --model <gguf> --text <text> [options]\n"
        "\n"
        "Required:\n"
        "  --model PATH           GGUF file produced by scripts/convert_hf_to_gguf.py\n"
        "  --version              Print the openmoss version and exit\n"
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
        "  --min-audio-frames N   Forbid end-of-speech before N frames (0 = no floor)\n"
        "  --max-audio-frames N   Force end-of-speech at N frames (0 = no cap). One\n"
        "                         frame is 80 ms. This is the bound to reach for when a\n"
        "                         model will not emit end-of-speech and generates until\n"
        "                         --max-new-tokens: it caps the damage without needing\n"
        "                         to know why. MOSS-VoiceGenerator is the usual case —\n"
        "                         it has no reference audio to anchor its length.\n"
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
        "\n"
        "Streaming (autoregressive families only; MOSS-SoundEffect refines the whole\n"
        "latent at once and has no prefix to emit):\n"
        "  --stream [N]           Decode incrementally, N frames at a time (default 8).\n"
        "                         A frame is 80 ms, so N is the output granularity.\n"
        "                         The decode runs inline with generation, so small N\n"
        "                         costs throughput: 8 gives 0.64s chunks at ~1.46x real\n"
        "                         time against 1.74x for one batch decode at the end.\n"
        "                         The WAV written at exit is identical either way.\n"
        "  --stream-out PATH      Also write each chunk as raw 16-bit little-endian PCM\n"
        "                         as it arrives; '-' means stdout. Implies --stream.\n"
        "                         Pipe it to a player to hear the latency directly:\n"
        "                           --stream-out - | aplay -f S16_LE -r 48000 -c 2\n"
        "                         (use -r 24000 -c 1 for the delay family)\n"
        "\n"
        "MOSS-SoundEffect only (the model is a diffusion transformer, not an\n"
        "autoregressive one, so the sampling flags above do not apply). --text\n"
        "carries the sound description:\n"
        "  --seconds F            Output duration (default: 10.0, capped by the model)\n"
        "  --steps N              Solver steps (default: 100). Each costs two DiT\n"
        "                         passes unless --cfg-scale is 1.\n"
        "  --cfg-scale F          Classifier-free guidance (default: 4.0; 1.0 skips\n"
        "                         the unconditional pass and halves the work)\n"
        "  --sigma-shift F        Flow-match schedule shift (default: 5.0)\n"
        "  --negative-prompt STR  Default is empty, which upstream treats as an\n"
        "                         all-zero conditioning and needs no encoder pass\n"
        "  --no-duration-suffix   Don't append ' duration: <X>s' to the prompt.\n"
        "                         The model was trained with it; expect drift.\n"
        "  --dump-latent PATH     Write the final latent as raw f32\n"
        "  --init-latent PATH     Read the starting noise from raw f32 instead of\n"
        "                         seeding it, for diffing against a reference\n"
        "  --dump-context PATH    Write the text conditioning as raw f32. Worth\n"
        "                         diffing on its own: guidance decays to under 1%%\n"
        "                         of the velocity within a few steps, so the final\n"
        "                         latent barely tests the text encoder\n"
        "  --vk-f32               Recommended on Vulkan. Sets GGML_VK_DISABLE_F16,\n"
        "                         which stops the backend accumulating libllama's\n"
        "                         matmuls in f16. That costs the text encoder ~3%%\n"
        "                         relative error, and the DiT cross-attends to it on\n"
        "                         every one of the ~200 forward passes: it moves the\n"
        "                         final latent from 1.7e-2 to 1.1e-3 against the fp32\n"
        "                         reference, with no measurable slowdown. The graphs\n"
        "                         in this repo already force f32 accumulation, so\n"
        "                         this only affects the backbone.\n"
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
        else if (k == "--min-audio-frames")  a.min_audio_frames  = require_int(i, argc, argv);
        else if (k == "--max-audio-frames")  a.max_audio_frames  = require_int(i, argc, argv);
        else if (k == "--seed")              a.seed              = uint64_t(std::strtoull(require_str(i, argc, argv).c_str(), nullptr, 10));
        else if (k == "--dump-codes")        a.dump_codes        = require_str(i, argc, argv);
        else if (k == "--n-gpu-layers")    a.n_gpu_layers = require_int(i, argc, argv);
        else if (k == "--main-gpu")        a.main_gpu     = require_int(i, argc, argv);
        else if (k == "--n-batch")         a.n_batch      = require_int(i, argc, argv);
        else if (k == "--n-ctx")           a.n_ctx        = require_int(i, argc, argv);
        else if (k == "--stream") {
            // Optional count: only swallow the next token when it is one.
            if (i + 1 < argc && std::isdigit(static_cast<unsigned char>(argv[i + 1][0]))) {
                a.stream_chunk = std::atoi(argv[++i]);
            } else {
                a.stream_chunk = 8;
            }
        }
        else if (k == "--stream-out")      a.stream_out = require_str(i, argc, argv);
        else if (k == "--no-flash-attn")   a.flash_attn = false;
        else if (k == "--skip-codec")      a.skip_codec = true;
        else if (k == "--aux-cpu")         a.aux_cpu    = true;
        else if (k == "--seconds")         a.seconds     = require_float(i, argc, argv);
        else if (k == "--steps")           a.steps       = require_int(i, argc, argv);
        else if (k == "--cfg-scale")       a.cfg_scale   = require_float(i, argc, argv);
        else if (k == "--sigma-shift")     a.sigma_shift = require_float(i, argc, argv);
        else if (k == "--negative-prompt") a.negative_prompt = require_str(i, argc, argv);
        else if (k == "--no-duration-suffix") a.no_duration_suffix = true;
        else if (k == "--dump-latent")     a.dump_latent = require_str(i, argc, argv);
        else if (k == "--init-latent")     a.init_latent = require_str(i, argc, argv);
        else if (k == "--dump-context")    a.dump_context = require_str(i, argc, argv);
        else if (k == "--vk-f32")          a.vk_f32 = true;
        else if (k == "--version") { std::printf("openmoss %s\n", OPENMOSS_VERSION); return 0; }
        else if (k == "--help" || k == "-h") usage(0);
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(2); }
    }
    if (a.model_path.empty() || a.text.empty()) usage(2);

    // Must precede Model::load: the Vulkan backend reads this when it
    // initialises the device, which happens during the load.
    if (a.vk_f32) setenv("GGML_VK_DISABLE_F16", "1", /*overwrite=*/0);

    openmoss::LoadOptions lo;
    lo.n_gpu_layers = a.n_gpu_layers;
    lo.main_gpu     = a.main_gpu;
    if (a.n_batch > 0) lo.n_batch = a.n_batch;
    if (a.n_ctx   > 0) lo.n_ctx   = a.n_ctx;
    lo.flash_attn   = a.flash_attn;
    lo.skip_codec   = a.skip_codec;
    lo.aux_cpu      = a.aux_cpu;
    auto model = openmoss::Model::load(a.model_path, lo);

    // MOSS-SoundEffect is not autoregressive: no prompt grid, no codes, no
    // sampling. It takes the flow-matching path instead, sharing only the model
    // loader and the WAV writer with the two TTS families.
    if (model->dims().arch == openmoss::Arch::SoundEffect) {
        openmoss::SoundEffectRequest sreq;
        sreq.prompt = a.text;
        if (a.seconds)     sreq.seconds             = *a.seconds;
        if (a.steps)       sreq.num_inference_steps = *a.steps;
        if (a.cfg_scale)   sreq.cfg_scale           = *a.cfg_scale;
        if (a.sigma_shift) sreq.sigma_shift         = *a.sigma_shift;
        if (a.negative_prompt) sreq.negative_prompt = *a.negative_prompt;
        if (a.seed)        sreq.seed                = *a.seed;
        sreq.append_duration_suffix = !a.no_duration_suffix;

        if (a.init_latent) {
            std::ifstream f(*a.init_latent, std::ios::binary | std::ios::ate);
            if (!f) { std::fprintf(stderr, "cannot open %s\n", a.init_latent->c_str()); return 1; }
            const std::streamoff bytes = f.tellg();
            f.seekg(0);
            sreq.initial_latent.resize(size_t(bytes) / sizeof(float));
            f.read(reinterpret_cast<char *>(sreq.initial_latent.data()), bytes);
        }

        auto result = openmoss::generate_sound_effect(
            *model, sreq, [](int step, int total) {
                std::fprintf(stderr, "\r[sfx] step %d/%d", step, total);
                if (step == total) std::fprintf(stderr, "\n");
                std::fflush(stderr);
            });

        std::fprintf(stderr,
                     "[sfx] prompt: \"%s\" (%d tokens, %d latents)\n"
                     "[sfx] text %.2fs, solve %.2fs, decode %.2fs\n",
                     result.prompt.c_str(), result.n_prompt_tokens, result.n_latents,
                     result.text_seconds, result.sample_seconds, result.decode_seconds);

        if (a.dump_context) {
            std::ofstream f(*a.dump_context, std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot open %s\n", a.dump_context->c_str()); return 1; }
            f.write(reinterpret_cast<const char *>(result.context.data()),
                    std::streamsize(result.context.size() * sizeof(float)));
            std::fprintf(stderr, "wrote context (%zu floats) to %s\n",
                         result.context.size(), a.dump_context->c_str());
        }

        if (a.dump_latent) {
            std::ofstream f(*a.dump_latent, std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot open %s\n", a.dump_latent->c_str()); return 1; }
            f.write(reinterpret_cast<const char *>(result.latent.data()),
                    std::streamsize(result.latent.size() * sizeof(float)));
            std::fprintf(stderr, "wrote latent (%zu floats) to %s\n",
                         result.latent.size(), a.dump_latent->c_str());
        }

        openmoss::write_wav(a.output_path, result.waveform.data(),
                            int64_t(result.waveform.size()),
                            result.sampling_rate, /*channels=*/1);
        std::fprintf(stderr, "wrote %lld samples (%.2fs, 1 ch @ %d Hz) to %s\n",
                     (long long)result.waveform.size(),
                     double(result.waveform.size()) / double(result.sampling_rate),
                     result.sampling_rate, a.output_path.c_str());
        return 0;
    }

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
    if (a.min_audio_frames) req.sampling.min_audio_frames = *a.min_audio_frames;
    if (a.max_audio_frames) req.sampling.max_audio_frames = *a.max_audio_frames;
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

    // Streaming: decode alongside generation and report when audio first exists.
    // That number is the whole point of the feature — the total is unchanged.
    openmoss::StreamCallback scb;
    FILE * stream_file = nullptr;
    // These outlive the block that builds the callback — it captures by
    // reference and is not invoked until generate() runs.
    auto    stream_t0    = std::chrono::steady_clock::now();
    int64_t stream_total = 0;
    bool    stream_first = true;
    if (a.stream_out && !a.stream_chunk) a.stream_chunk = 8;
    if (a.stream_chunk) {
        if (*a.stream_chunk < 1) {
            std::fprintf(stderr, "--stream needs a positive frame count\n");
            return 2;
        }
        req.stream_chunk_frames = *a.stream_chunk;
        if (a.stream_out) {
            if (*a.stream_out == "-") {
                stream_file = stdout;
            } else {
                stream_file = std::fopen(a.stream_out->c_str(), "wb");
                if (!stream_file) {
                    std::fprintf(stderr, "cannot open %s\n", a.stream_out->c_str());
                    return 1;
                }
            }
        }
        stream_t0 = std::chrono::steady_clock::now();
        scb = [&](const float * pcm, int64_t n_samples) {
            const double at = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - stream_t0).count();
            if (stream_first) {
                std::fprintf(stderr, "[stream] first audio after %.2fs\n", at);
                stream_first = false;
            }
            stream_total += n_samples;
            if (stream_file) {
                const auto bytes = openmoss::encode_pcm_s16le(pcm, n_samples);
                std::fwrite(bytes.data(), 1, bytes.size(), stream_file);
                std::fflush(stream_file);
            }
            // Audio produced so far, against the wall-clock spent producing it.
            const double have = double(stream_total)
                              / double(dims.sampling_rate) / double(dims.n_channels);
            std::fprintf(stderr, "\r[stream] %6.2fs of audio in %6.2fs (%.2fx real time)",
                         have, at, have / (at > 0 ? at : 1));
            std::fflush(stderr);
        };
    }

    auto result = openmoss::generate(*model, req, scb);
    if (a.stream_chunk) std::fprintf(stderr, "\n");
    if (stream_file && stream_file != stdout) std::fclose(stream_file);

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
