// SPDX-License-Identifier: Apache-2.0
//
// Persistent-model HTTP server. Loads the GGUF once and answers POST /tts
// requests in a serial fashion (libllama state is not reentrant, so a single
// mutex around generate() is required for now).
//
// API
// ---
//   GET  /health                       → "ok\n"  (200)
//   GET  /info                         → JSON with model dims + load options
//   GET  /  (and other static paths)   → WebUI (when --webui-dir is set)
//   POST /v1/audio/speech              → OpenAI-compatible TTS endpoint
//   POST /tts
//        Content-Type: application/json
//        Body: see TtsRequest below.
//        Response: 200 audio/wav (16-bit PCM @ 24 kHz mono)
//                   on error: 4xx / 5xx text/plain with the message.
//
// /tts endpoint request schema (all fields optional except `text`):
//   {
//     "text":              str,       // required
//     "instruction":       str,
//     "language":          str,       // "en" | "zh" | …
//     "tokens":            int,       // duration hint, 1s ≈ 12.5 tokens
//     "max_new_tokens":    int,       // default 4096
//     "reference_wav_b64": str,       // base64 of a WAV file for voice cloning
//     "sampling": {
//        "text_temperature": float, "text_top_p": float, "text_top_k": int,
//        "audio_temperature": float, "audio_top_p": float, "audio_top_k": int,
//        "audio_repetition_penalty": float, "seed": uint64
//     }
//   }
//
// /v1/audio/speech request schema:
//   {
//     "model":             str,       // ignored (model is pre-loaded)
//     "input":             str,       // required — text to synthesize
//     "voice":             str,       // optional — mapped to instruction or ignored
//     "response_format":   str,       // "wav" (default), only WAV is supported
//     "speed":             float      // optional — 0.25..4.0, maps to token count
//   }

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "openmoss/delay.h"
#include "openmoss/model.h"
#include "openmoss/pipeline.h"
#include "openmoss/wav.h"

using json = nlohmann::json;

namespace {

[[noreturn]] void usage(int code) {
    std::fprintf(stderr,
        "Usage: moss-tts-server --model <gguf> [options]\n"
        "  --host HOST            (default: 127.0.0.1)\n"
        "  --port PORT            (default: 8080)\n"
        "  --n-gpu-layers N       (default: -1 = all)\n"
        "  --main-gpu N           (default: -1 = auto, picks GPU with most free VRAM)\n"
        "  --n-batch N            libllama batch size (default: 512). Raise if a\n"
        "                         long prompt exceeds this size.\n"
        "  --n-ctx N              libllama context size (default: 8192)\n"
        "  --no-flash-attn\n"
        "  --skip-codec           (no waveform synthesis; codes only — debug)\n"
        "  --aux-cpu              force audio embeds + codec onto CPU\n"
        "                          (workaround for Metal DIAG_MASK_INF)\n"
        "  --webui-dir DIR        serve a static WebUI from DIR at /\n"
        "                          (default: auto-detect ./webui or <binary>/webui)\n"
        "  --no-webui             disable WebUI auto-detection\n"
    );
    std::exit(code);
}

// Resolve a usable WebUI directory. Honor an explicit override first, otherwise
// look in a couple of conventional spots: the CWD, the binary's directory, and
// one level up (so `build/moss-tts-server` finds `../webui`).
std::string find_webui_dir(const std::string & explicit_dir,
                            const std::string & argv0) {
    namespace fs = std::filesystem;
    auto check = [](const fs::path & p) -> std::string {
        std::error_code ec;
        if (fs::is_directory(p, ec) && fs::exists(p / "index.html", ec)) {
            return fs::absolute(p, ec).string();
        }
        return {};
    };

    if (!explicit_dir.empty()) {
        auto r = check(explicit_dir);
        if (!r.empty()) return r;
        std::fprintf(stderr,
            "[server] --webui-dir '%s' is not a directory containing index.html\n",
            explicit_dir.c_str());
        return {};
    }

    if (auto r = check(fs::path("webui")); !r.empty()) return r;

    std::error_code ec;
    fs::path exe(argv0);
    fs::path exe_dir = fs::absolute(exe, ec).parent_path();
    if (auto r = check(exe_dir / "webui"); !r.empty()) return r;
    if (auto r = check(exe_dir.parent_path() / "webui"); !r.empty()) return r;
    return {};
}

// ── tiny base64 decoder ───────────────────────────────────────────────────
std::vector<uint8_t> b64_decode(const std::string & s) {
    static int8_t tab[256];
    static bool inited = false;
    if (!inited) {
        for (int i = 0; i < 256; ++i) tab[i] = -1;
        const char * alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tab[uint8_t(alphabet[i])] = int8_t(i);
        inited = true;
    }

    std::vector<uint8_t> out;
    out.reserve(s.size() * 3 / 4);
    int val = 0, bits = -8;
    for (char c : s) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '=') continue;
        int8_t v = tab[uint8_t(c)];
        if (v < 0) throw std::runtime_error("invalid base64 character");
        val = (val << 6) | v;
        bits += 6;
        if (bits >= 0) {
            out.push_back(uint8_t((val >> bits) & 0xff));
            bits -= 8;
        }
    }
    return out;
}

// Optional with fallback helper.
template <typename T>
T jget(const json & j, const char * k, const T & dflt) {
    auto it = j.find(k);
    if (it == j.end() || it->is_null()) return dflt;
    return it->get<T>();
}

// Per-model default decoding. The struct defaults suit MOSS-TTS (n_vq=32). The
// reduced-codebook MOSS-VoiceGenerator (n_vq=16) is sensitive to decoding and
// degenerates (immediate end-of-speech) at those settings; use its documented
// recommended hyperparameters instead. Callers can still override per request.
openmoss::SamplingConfig default_sampling(int n_vq) {
    openmoss::SamplingConfig sc;
    if (n_vq < 32) {
        sc.audio_temperature        = 1.5f;
        sc.audio_top_p              = 0.6f;
        sc.audio_top_k              = 50;
        sc.audio_repetition_penalty = 1.1f;
    }
    return sc;
}

openmoss::SamplingConfig parse_sampling(const json & j, const openmoss::SamplingConfig & base) {
    openmoss::SamplingConfig sc = base;
    if (!j.is_object()) return sc;
    sc.text_temperature        = jget(j, "text_temperature",        sc.text_temperature);
    sc.text_top_p              = jget(j, "text_top_p",              sc.text_top_p);
    sc.text_top_k              = jget(j, "text_top_k",              sc.text_top_k);
    sc.audio_temperature       = jget(j, "audio_temperature",       sc.audio_temperature);
    sc.audio_top_p             = jget(j, "audio_top_p",             sc.audio_top_p);
    sc.audio_top_k             = jget(j, "audio_top_k",             sc.audio_top_k);
    sc.audio_repetition_penalty = jget(j, "audio_repetition_penalty", sc.audio_repetition_penalty);
    sc.seed                    = jget(j, "seed",                    sc.seed);
    return sc;
}

// MOSS-VoiceGenerator (n_vq<32) has no reference audio to anchor length: left
// free it either collapses the segment on the first frame (degenerate immediate
// end-of-speech) or rambles for minutes. Estimate a length from the text when the
// caller gave none, then bound the generated segment around it — floor at 3/4
// (render the text, no early collapse), cap at 3/2 (no ramble) — leaving a
// natural-ending window in between. No-op for full MOSS-TTS (n_vq>=32), which is
// anchored by its reference audio.
void finalize_voicegen_request(openmoss::GenerateRequest & req, const openmoss::ModelDims & dims) {
    if (dims.n_vq >= 32) return;
    if (!req.tokens && !req.text.empty()) {
        // ~12.5 audio tokens/s at ~2.5 words/s -> ~5 tokens/word.
        int words = 1;
        for (char c : req.text) if (c == ' ' || c == '\n' || c == '\t') ++words;
        req.tokens = std::max(40, std::min(1000, words * 5));
    }
    if (req.tokens) {
        if (req.sampling.min_audio_frames == 0)
            req.sampling.min_audio_frames = std::max(24, req.tokens.value() * 3 / 4);
        if (req.sampling.max_audio_frames == 0)
            req.sampling.max_audio_frames = std::max(48, req.tokens.value() * 3 / 2);
    }
}

void send_text_error(httplib::Response & rs, int status, const std::string & msg) {
    rs.status = status;
    rs.set_content(msg + "\n", "text/plain");
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    int  port         = 8080;
    int  n_gpu_layers = -1;
    int  main_gpu     = -1;
    int  n_batch      = -1;
    int  n_ctx        = -1;
    bool flash_attn   = true;
    bool skip_codec   = false;
    bool aux_cpu      = false;
    std::string webui_dir_arg;
    bool no_webui     = false;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(2);
            return argv[++i];
        };
        if      (k == "--model")          model_path   = next();
        else if (k == "--host")           host         = next();
        else if (k == "--port")           port         = std::atoi(next().c_str());
        else if (k == "--n-gpu-layers")   n_gpu_layers = std::atoi(next().c_str());
        else if (k == "--main-gpu")       main_gpu     = std::atoi(next().c_str());
        else if (k == "--n-batch")        n_batch      = std::atoi(next().c_str());
        else if (k == "--n-ctx")          n_ctx        = std::atoi(next().c_str());
        else if (k == "--no-flash-attn")  flash_attn   = false;
        else if (k == "--skip-codec")     skip_codec   = true;
        else if (k == "--aux-cpu")        aux_cpu      = true;
        else if (k == "--webui-dir")      webui_dir_arg = next();
        else if (k == "--no-webui")       no_webui     = true;
        else if (k == "--help" || k == "-h") usage(0);
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(2); }
    }
    if (model_path.empty()) usage(2);

    openmoss::LoadOptions lo;
    lo.n_gpu_layers = n_gpu_layers;
    lo.main_gpu     = main_gpu;
    if (n_batch > 0) lo.n_batch = n_batch;
    if (n_ctx   > 0) lo.n_ctx   = n_ctx;
    lo.flash_attn   = flash_attn;
    lo.skip_codec   = skip_codec;
    lo.aux_cpu      = aux_cpu;
    auto model = openmoss::Model::load(model_path, lo);
    std::fprintf(stderr,
                  "[server] model loaded; codec=%s\n",
                  model->codec_loaded() ? "on" : "off");

    httplib::Server svr;

    // Larger payload allowance: a 60 s reference WAV is ~5.7 MB raw + ~7.6 MB
    // base64. Round up to keep some slack for headers + JSON overhead.
    svr.set_payload_max_length(64 * 1024 * 1024);

    // Permissive CORS so the WebUI works whether served from this server or
    // from a different origin during development.
    svr.set_pre_routing_handler([](const httplib::Request & rq, httplib::Response & rs) {
        rs.set_header("Access-Control-Allow-Origin", "*");
        rs.set_header("Access-Control-Allow-Headers", "Content-Type");
        rs.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        if (rq.method == "OPTIONS") {
            rs.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::mutex gen_mu;
    std::atomic<uint64_t> n_requests{0};

    svr.Get("/health", [](const httplib::Request &, httplib::Response & rs) {
        rs.set_content("ok\n", "text/plain");
    });

    svr.Get("/info", [&](const httplib::Request &, httplib::Response & rs) {
        const auto & d = model->dims();
        json info = {
            {"sampling_rate",     d.sampling_rate},
            {"n_vq",              d.n_vq},
            {"audio_vocab_size",  d.audio_vocab_size},
            {"hidden_size",       d.hidden_size},
            {"frame_rate_hz",     double(d.sampling_rate) / double(d.downsample_rate)},
            {"codec_present",     model->codec_present()},
            {"codec_loaded",      model->codec_loaded()},
            {"requests_served",   uint64_t(n_requests.load())},
        };
        rs.set_content(info.dump(), "application/json");
    });

    svr.Post("/tts", [&](const httplib::Request & rq, httplib::Response & rs) {
        const uint64_t req_id = ++n_requests;
        const auto t0 = std::chrono::steady_clock::now();

        json body;
        try {
            body = json::parse(rq.body);
        } catch (const std::exception & e) {
            return send_text_error(rs, 400,
                std::string("invalid JSON body: ") + e.what());
        }
        if (!body.is_object()) return send_text_error(rs, 400, "JSON body must be an object");
        if (!body.contains("text") || !body["text"].is_string())
            return send_text_error(rs, 400, "missing required field 'text' (string)");

        openmoss::GenerateRequest req;
        req.text          = body["text"].get<std::string>();
        if (body.contains("instruction") && body["instruction"].is_string())
            req.instruction = body["instruction"].get<std::string>();
        if (body.contains("language") && body["language"].is_string())
            req.language    = body["language"].get<std::string>();
        if (body.contains("quality") && body["quality"].is_string())
            req.quality     = body["quality"].get<std::string>();
        if (body.contains("tokens") && body["tokens"].is_number_integer())
            req.tokens      = body["tokens"].get<int>();
        req.max_new_tokens  = jget(body, "max_new_tokens", req.max_new_tokens);
        req.sampling        = parse_sampling(body.value("sampling", json::object()),
                                             default_sampling(model->dims().n_vq));
        finalize_voicegen_request(req, model->dims());

        if (body.contains("reference_wav_b64") && body["reference_wav_b64"].is_string()) {
            try {
                auto wav_bytes = b64_decode(body["reference_wav_b64"].get<std::string>());
                req.reference_wav = openmoss::decode_wav_mono(
                    wav_bytes.data(), wav_bytes.size(),
                    model->dims().sampling_rate);
            } catch (const std::exception & e) {
                return send_text_error(rs, 400,
                    std::string("could not decode reference_wav_b64: ") + e.what());
            }
        }

        openmoss::GenerateResult result;
        try {
            std::lock_guard<std::mutex> g(gen_mu);
            std::fprintf(stderr,
                "[server] req#%llu text=%zu chars ref=%s max=%d\n",
                (unsigned long long)req_id,
                req.text.size(),
                req.reference_wav ? "yes" : "no",
                req.max_new_tokens);
            result = openmoss::generate(*model, req);
        } catch (const std::exception & e) {
            return send_text_error(rs, 500,
                std::string("generation failed: ") + e.what());
        }

        if (result.waveform.empty()) {
            return send_text_error(rs, 500,
                "generation produced no audio (codec missing or model emitted EOS too early)");
        }

        auto wav_bytes = openmoss::encode_wav_mono(
            result.waveform.data(), int64_t(result.waveform.size()),
            model->dims().sampling_rate);

        const std::chrono::duration<double> total_s =
            std::chrono::steady_clock::now() - t0;
        std::fprintf(stderr,
            "[server] req#%llu ok: %zu samples (%.2fs audio) — prefill %.2fs gen %.2fs decode %.2fs total %.2fs\n",
            (unsigned long long)req_id,
            result.waveform.size(),
            result.waveform.size() / double(model->dims().sampling_rate),
            result.prefill_seconds,
            result.generate_seconds,
            result.decode_seconds,
            total_s.count());

        rs.set_header("X-MOSS-Audio-Frames", std::to_string(result.n_audio_frames));
        rs.set_header("X-MOSS-Generate-Seconds",
                       std::to_string(result.generate_seconds));
        rs.set_header("X-MOSS-Decode-Seconds",
                       std::to_string(result.decode_seconds));
        rs.set_content(reinterpret_cast<const char *>(wav_bytes.data()),
                        wav_bytes.size(), "audio/wav");
    });

    svr.Post("/v1/audio/speech", [&](const httplib::Request & rq, httplib::Response & rs) {
        const uint64_t req_id = ++n_requests;
        const auto t0 = std::chrono::steady_clock::now();

        json body;
        try {
            body = json::parse(rq.body);
        } catch (const std::exception & e) {
            return send_text_error(rs, 400,
                std::string("invalid JSON body: ") + e.what());
        }
        if (!body.is_object()) return send_text_error(rs, 400, "JSON body must be an object");
        if (!body.contains("input") || !body["input"].is_string())
            return send_text_error(rs, 400, "missing required field 'input' (string)");

        // Check response_format — only "wav" is supported.
        std::string fmt = jget(body, "response_format", std::string("wav"));
        if (fmt != "wav") {
            return send_text_error(rs, 400,
                "unsupported response_format '" + fmt + "'; only 'wav' is supported");
        }

        // Translate OpenAI fields → native MOSS GenerateRequest.
        openmoss::GenerateRequest req;
        req.text = body["input"].get<std::string>();

        // "voice" → instruction hint (e.g. "alloy", "echo", …).
        if (body.contains("voice") && body["voice"].is_string()) {
            req.instruction = body["voice"].get<std::string>();
        }

        // "speed" scales the token budget.  speed=1.0 → default max_new_tokens.
        // We approximate: 1s of audio ≈ 12.5 tokens, and the default model output
        // is roughly 4096 tokens ≈ 328s.  speed just multiplies the budget.
        double speed = jget(body, "speed", 1.0);
        if (speed <= 0.0 || speed > 4.0) {
            return send_text_error(rs, 400,
                "speed must be between 0.25 and 4.0");
        }
        // Scale the default token budget by speed (lower speed = fewer tokens = shorter audio).
        req.max_new_tokens = std::max(1, int(4096 / speed));
        req.sampling       = default_sampling(model->dims().n_vq);
        finalize_voicegen_request(req, model->dims());

        openmoss::GenerateResult result;
        try {
            std::lock_guard<std::mutex> g(gen_mu);
            std::fprintf(stderr,
                "[server] req#%llu /v1/audio/speech input=%zu chars speed=%.2f max=%d\n",
                (unsigned long long)req_id,
                req.text.size(),
                speed,
                req.max_new_tokens);
            result = openmoss::generate(*model, req);
        } catch (const std::exception & e) {
            return send_text_error(rs, 500,
                std::string("generation failed: ") + e.what());
        }

        if (result.waveform.empty()) {
            return send_text_error(rs, 500,
                "generation produced no audio (codec missing or model emitted EOS too early)");
        }

        auto wav_bytes = openmoss::encode_wav_mono(
            result.waveform.data(), int64_t(result.waveform.size()),
            model->dims().sampling_rate);

        const std::chrono::duration<double> total_s =
            std::chrono::steady_clock::now() - t0;
        std::fprintf(stderr,
            "[server] req#%llu ok: %zu samples (%.2fs audio) — prefill %.2fs gen %.2fs decode %.2fs total %.2fs\n",
            (unsigned long long)req_id,
            result.waveform.size(),
            result.waveform.size() / double(model->dims().sampling_rate),
            result.prefill_seconds,
            result.generate_seconds,
            result.decode_seconds,
            total_s.count());

        rs.set_header("X-MOSS-Audio-Frames", std::to_string(result.n_audio_frames));
        rs.set_header("X-MOSS-Generate-Seconds",
                       std::to_string(result.generate_seconds));
        rs.set_header("X-MOSS-Decode-Seconds",
                       std::to_string(result.decode_seconds));
        rs.set_content(reinterpret_cast<const char *>(wav_bytes.data()),
                        wav_bytes.size(), "audio/wav");
    });

    svr.set_logger([](const httplib::Request & rq, const httplib::Response & rs) {
        std::fprintf(stderr, "[server] %s %s → %d\n",
                      rq.method.c_str(), rq.path.c_str(), rs.status);
    });

    // Static WebUI mount. The mount is added last so the JSON / audio
    // handlers above always take precedence on path conflicts.
    if (!no_webui) {
        std::string webui_dir = find_webui_dir(webui_dir_arg, argv[0]);
        if (!webui_dir.empty()) {
            svr.set_mount_point("/", webui_dir);
            std::fprintf(stderr, "[server] WebUI mounted at / from %s\n",
                          webui_dir.c_str());
        } else if (!webui_dir_arg.empty()) {
            std::fprintf(stderr, "[server] WebUI disabled (override path invalid)\n");
        } else {
            std::fprintf(stderr, "[server] WebUI not found; pass --webui-dir DIR to enable\n");
        }
    }

    std::fprintf(stderr, "[server] listening on http://%s:%d\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "[server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}