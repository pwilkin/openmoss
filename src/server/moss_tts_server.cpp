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
//   GET  /v1/models                    → OpenAI-compatible model list (one entry)
//   GET  /v1/audio/voices              → voices registry (see below)
//   GET  /  (and other static paths)   → WebUI (when --webui-dir is set)
//   POST /tts                          → native TTS, autoregressive families
//   POST /sfx                          → native MOSS-SoundEffect
//   POST /v1/audio/speech              → OpenAI-compatible; dispatches on the
//                                        loaded model's architecture
//
// One model is loaded at startup, so exactly one of /tts and /sfx applies; the
// other answers 400 with a pointer to the right one. Responses carry the model's
// own sample rate and channel count (24 kHz mono, 48 kHz stereo, or 48 kHz mono
// depending on the family) — `response_format` selects "wav" (RIFF) or "pcm"
// (raw 16-bit little-endian, no container).
//
// /tts request schema (all fields optional except `text`):
//   {
//     "text":              str,       // required. May start with "${token:N}"
//                                     //   as an inline duration hint.
//     "instruction":       str,
//     "language":          str,       // "en" | "zh" | …
//     "token_count":       int,       // duration hint, 1s ≈ 12.5 tokens.
//                                     //   Wins over the "${token:N}" prefix.
//     "tokens":            int,       // legacy alias for token_count
//     "max_new_tokens":    int,       // default 4096
//     "response_format":   str,       // "wav" (default) | "pcm"
//     "stream":            bool,      // default false. See "Streaming" below.
//     "stream_chunk_frames": int,     // default 8; a frame is 80 ms
//     "reference_wav_b64": str,       // base64 of a WAV file for voice cloning
//     "ref_text":          str,       // transcript of the reference. Selects
//                                     //   continuation mode (moss_tts_local
//                                     //   only): the reference goes in the
//                                     //   assistant channel and the model
//                                     //   carries on speaking from it.
//     "references": [ {"ref_audio": str, "ref_text": str} ]   // same, array form.
//                                     //   The delay family accepts SEVERAL
//                                     //   entries — one per speaker, in [S1],
//                                     //   [S2], … order. Their transcripts do
//                                     //   NOT go in ref_text (rejected) but in
//                                     //   `text`, as "[S1] <transcript1> [S2]
//                                     //   <transcript2> [S1] <turn> …", per the
//                                     //   MOSS-TTSD prompt format.
//     "continuation_prefix": bool,    // default: per checkpoint — on for
//                                     //   n_vq<32, OFF for the flagship MOSS-TTS
//                                     //   (n_vq>=32), where it stops the model
//                                     //   following its text. Delay family with
//                                     //   references only: splice the reference
//                                     //   audio behind the assistant's
//                                     //   audio_start so the model does not
//                                     //   re-speak the transcripts. Disable for
//                                     //   an identity A/B — or to stream, which
//                                     //   cannot trim the prefix.
//     "voice":             str,       // a REGISTERED voice id (--voice-dir;
//                                     //   see /v1/voices). ""/"default" = none.
//                                     //   Delay family: its stored transcript
//                                     //   is prepended to `text`.
//     "voices":            [str, …],  // delay family: one registered voice per
//                                     //   speaker, [S1]..[Sn] order; stored
//                                     //   transcripts are prepended as
//                                     //   "[S1] <t1> [S2] <t2> " so `text`
//                                     //   carries only the dialogue turns.
//                                     //   Reference codes are encoded once and
//                                     //   cached — the encode otherwise
//                                     //   dominates request latency.
//     "sampling": {
//        "text_temperature": float, "text_top_p": float, "text_top_k": int,
//        "audio_temperature": float, "audio_top_p": float, "audio_top_k": int,
//        "audio_repetition_penalty": float, "seed": uint64
//     }
//   }
//
// /sfx request schema (all fields optional except `prompt`):
//   {
//     "prompt":            str,       // required ("input"/"text" also accepted)
//     "seconds":           float,     // default 10.0, capped by the model
//     "steps":             int,       // default 100; each costs two DiT passes
//     "cfg_scale":         float,     // default 4.0; 1.0 halves the work
//     "sigma_shift":       float,     // default 5.0
//     "negative_prompt":   str,       // default empty, which is what it was trained on
//     "seed":              uint64,    // 0 is a valid seed, not a sentinel
//     "response_format":   str        // "wav" (default) | "pcm"
//   }
//
// /v1/audio/speech request schema:
//   {
//     "model":             str,       // ignored (the model is pre-loaded)
//     "input":             str,       // required — text, or the sound description
//     "voice":             str,       // TTS only — mapped to instruction
//     "response_format":   str,       // "wav" (default) | "pcm"
//     "stream":            bool,      // TTS only. See "Streaming" below.
//     "stream_chunk_frames": int,     // default 8; a frame is 80 ms
//     "speed":             float,     // TTS only — 0.25..4.0, scales token budget
//     …                               // sampling keys may also be passed flat
//                                     //   here, as the cookbooks send them; for
//                                     //   MOSS-SoundEffect the /sfx solver
//                                     //   fields are accepted instead
//   }
//
// Streaming
// ---------
// `"stream": true` on /tts or /v1/audio/speech answers with chunked transfer,
// writing each codec chunk as it is decoded rather than buffering the whole
// utterance. It cuts time-to-first-audio from the full generation time to
// roughly one chunk (~0.7 s for the default 8 frames on MOSS-TTS-Local),
// at the cost of about 8% total throughput, since the codec now runs inline
// with generation.
//
// It requires `response_format: "pcm"`. A RIFF header states its own data
// length in its first 44 bytes and that is unknown until generation stops, so
// rather than emit a header that lies the endpoint rejects the combination.
// X-MOSS-Sample-Rate and X-MOSS-Channels carry what a player needs.
//
// The status line is sent before generation starts, so a mid-stream failure
// truncates the response instead of becoming a 500 — the request is fully
// validated first, and failures are logged server-side.
//
// MOSS-SoundEffect cannot stream at all: flow matching refines every latent
// frame simultaneously, so no prefix of the audio exists until the last solver
// step finishes. /sfx rejects the flag.
//
// A note on voices: there is no upstream standard. Every MOSS cookbook passes
// voice="default", and cloning is driven by reference audio rather than a named
// voice, so /v1/audio/voices follows the vLLM-Omni shape and reports the single
// "default" entry (empty for MOSS-SoundEffect, which has no notion of a voice).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "httplib.h"
#include <nlohmann/json.hpp>

#include "openmoss/delay.h"
#include "openmoss/model.h"
#include "openmoss/pipeline.h"
#include "openmoss/soundeffect.h"
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
        "  --template NAME        Prompt template + sampling defaults for the delay\n"
        "                         family: auto (default), ttsd or voicegen. The two\n"
        "                         checkpoints ship different processors and decoding\n"
        "                         hyperparameters but are indistinguishable from the\n"
        "                         GGUF alone (both have n_vq=16), so `auto` keeps the\n"
        "                         historical n_vq<32 → voicegen guess; pass\n"
        "                         `--template ttsd` when serving MOSS-TTSD.\n"
        "  --no-flash-attn\n"
        "  --skip-codec           (no waveform synthesis; codes only — debug)\n"
        "  --aux-cpu              force audio embeds + codec onto CPU\n"
        "                          (workaround for Metal DIAG_MASK_INF)\n"
        "  --voice-dir DIR        enable the voice registry: persistent named\n"
        "                          references at DIR/{id}.wav + DIR/{id}.json,\n"
        "                          managed via /v1/voices and used by passing\n"
        "                          voice/voices on a speech request. Reference\n"
        "                          codes are encoded once and cached in memory.\n"
        "  --webui-dir DIR        serve a static WebUI from DIR at /\n"
        "                          (default: auto-detect ./webui or <binary>/webui)\n"
        "  --no-webui             disable WebUI auto-detection\n"
        "\n"
        "Routes: /health /info /v1/models /v1/audio/voices /v1/audio/speech\n"
        "        plus /tts (autoregressive families) or /sfx (MOSS-SoundEffect)\n"
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

// ── voice registry ────────────────────────────────────────────────────────
//
// Persistent named references, mirroring the omnivoice-server contract so a
// gateway can drive both engines the same way:
//   POST   /v1/voices          multipart: voice_id|id + audio (file) +
//                              transcript|text + language? + description?
//   GET    /v1/voices          {"voices":[{id,transcript,description,language,
//                              created_at}, …]}
//   GET    /v1/voices/{id}     metadata          (404 when unknown)
//   DELETE /v1/voices/{id}     {"id":…,"deleted":true}
// Storage: {voice-dir}/{id}.wav (bytes as uploaded) + {id}.json. Enabled by
// --voice-dir. Deliberate deviations from omnivoice: an unknown `voice` on a
// speech request is a 400 listing the known ids (omnivoice falls back to its
// auto-voice; silently giving a dialogue speaker the wrong voice is worse
// than failing), and /v1/voices/design is absent — MOSS designs voices with a
// separate checkpoint (MOSS-VoiceGenerator), not this one.
bool valid_voice_id(const std::string & s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

struct VoiceMeta {
    std::string transcript;
    std::string description;
    std::string language;
    std::string created_at;
};

json voice_meta_json(const std::string & id, const VoiceMeta & m) {
    return json{
        {"id",          id},
        {"transcript",  m.transcript},
        {"description", m.description},
        {"language",    m.language},
        {"created_at",  m.created_at},
    };
}

std::string iso8601_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

struct VoiceStore {
    std::string dir;   // empty = registry disabled

    bool enabled() const { return !dir.empty(); }
    std::string wav_path(const std::string & id)  const { return dir + "/" + id + ".wav"; }
    std::string meta_path(const std::string & id) const { return dir + "/" + id + ".json"; }

    bool exists(const std::string & id) const {
        std::error_code ec;
        return std::filesystem::exists(wav_path(id), ec);
    }

    std::vector<std::string> list() const {
        std::vector<std::string> out;
        std::error_code ec;
        for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
            const auto & p = e.path();
            if (p.extension() != ".wav") continue;
            const std::string id = p.stem().string();
            if (valid_voice_id(id)) out.push_back(id);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    bool load_meta(const std::string & id, VoiceMeta & out) const {
        std::ifstream f(meta_path(id));
        if (!f) return false;
        try {
            json j = json::parse(f);
            out.transcript  = j.value("transcript",  "");
            out.description = j.value("description", "");
            out.language    = j.value("language",    "");
            out.created_at  = j.value("created_at",  "");
        } catch (const std::exception &) {
            return false;
        }
        return true;
    }

    bool save_meta(const std::string & id, const VoiceMeta & m) const {
        std::ofstream f(meta_path(id), std::ios::trunc);
        if (!f) return false;
        f << voice_meta_json(id, m).dump(2) << "\n";
        return bool(f);
    }

    bool save_wav(const std::string & id, const std::string & bytes) const {
        std::ofstream f(wav_path(id), std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(bytes.data(), std::streamsize(bytes.size()));
        return bool(f);
    }

    std::vector<uint8_t> load_wav(const std::string & id) const {
        std::ifstream f(wav_path(id), std::ios::binary);
        if (!f) return {};
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
    }

    bool remove(const std::string & id) const {
        std::error_code ec;
        bool wav_ok = std::filesystem::remove(wav_path(id), ec);
        std::filesystem::remove(meta_path(id), ec);
        return wav_ok;
    }
};

// Which delay-family checkpoint this server is treating the model as. The two
// reduced-codebook checkpoints — MOSS-TTSD and MOSS-VoiceGenerator — ship
// different prompt templates and decoding hyperparameters but are
// indistinguishable from the GGUF alone (both have n_vq=16), so the operator
// says which one they loaded via --template; `Auto` keeps the historical
// n_vq<32 → voicegen guess.
enum class DelayTemplate { Auto, TTSD, VoiceGen };

// Per-model default decoding: the library's per-family defaults, then (delay
// family) the loaded checkpoint's own documented values. Mirrors the CLI.
// TTSD ships generation_config.json with 1.1/0.9/50/1.1; VoiceGenerator
// documents 1.5/0.6/50/1.1 in modeling_moss_tts.py:401-404 and its model card,
// with the warning that it is sensitive to these — at the flagship's struct
// defaults it degenerates (immediate end-of-speech). Callers can still
// override per request.
openmoss::SamplingConfig default_sampling(openmoss::Arch arch,
                                          bool ttsd_mode, bool voicegen_mode) {
    openmoss::SamplingConfig sc = openmoss::default_sampling(arch);
    if (ttsd_mode) {
        sc.audio_temperature        = 1.1f;
        sc.audio_top_p              = 0.9f;
        sc.audio_top_k              = 50;
        sc.audio_repetition_penalty = 1.1f;
    } else if (voicegen_mode) {
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
    // The hard bounds on segment length. finalize_voicegen_request() derives
    // these for MOSS-VoiceGenerator when the caller gives no duration hint, but
    // only fills a zero — so an explicit value here wins. Worth reaching for
    // when a model will not emit end-of-speech: it caps the damage without
    // needing to know why.
    sc.min_audio_frames        = jget(j, "min_audio_frames",        sc.min_audio_frames);
    sc.max_audio_frames        = jget(j, "max_audio_frames",        sc.max_audio_frames);
    return sc;
}

// MOSS-VoiceGenerator has no reference audio to anchor length: left free it
// either collapses the segment on the first frame (degenerate immediate
// end-of-speech) or rambles for minutes. Estimate a length from the text when the
// caller gave none, then bound the generated segment around it — floor at 3/4
// (render the text, no early collapse), cap at 3/2 (no ramble) — leaving a
// natural-ending window in between. Only in voicegen mode: MOSS-TTS and
// MOSS-TTSD are anchored by their reference audio and stop on their own — the
// word-count estimate would truncate a multi-speaker dialogue whose text also
// carries the reference transcripts.
void finalize_voicegen_request(openmoss::GenerateRequest & req, bool voicegen_mode) {
    if (!voicegen_mode) return;
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

// Thrown out of a stream callback when the socket is gone. generate() has no
// cancellation channel, so unwinding is the only way to stop it early.
struct StreamAborted {};

// ── response encoding ─────────────────────────────────────────────────────
//
// `wav` is a RIFF file; `pcm` is the same samples with no container, which is
// what a caller streaming straight into an audio device wants. Both carry the
// model's own rate and channel count — the delay family is 24 kHz mono,
// MOSS-TTS-Local 48 kHz stereo, MOSS-SoundEffect 48 kHz mono — so neither is
// hardcoded, and the previous unconditional use of the *mono* encoders wrote a
// stereo waveform under a 1-channel header.
struct Encoded {
    std::vector<uint8_t> bytes;
    const char *         mime;
};

Encoded encode_audio(const std::string & fmt, const float * pcm, int64_t n_samples,
                     int32_t sample_rate, int32_t channels) {
    if (fmt == "pcm") {
        return { openmoss::encode_pcm_s16le(pcm, n_samples), "audio/pcm" };
    }
    return { openmoss::encode_wav(pcm, n_samples, sample_rate, channels), "audio/wav" };
}

// The cookbooks let a caller ask for an approximate audio length inline, as a
// `${token:N}` prefix on the text. Strip it and return N; 0 when absent.
int extract_token_prefix(std::string & text) {
    const std::string open = "${token:";
    if (text.rfind(open, 0) != 0) return 0;
    const size_t close = text.find('}', open.size());
    if (close == std::string::npos) return 0;
    const std::string digits = text.substr(open.size(), close - open.size());
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) {
        return 0;   // not a token count after all; leave the text untouched
    }
    const int n = std::atoi(digits.c_str());
    text.erase(0, close + 1);
    if (!text.empty() && text.front() == ' ') text.erase(0, 1);
    return n;
}

// Pull reference audio (and optionally transcripts) out of a request. Two
// shapes are accepted: the legacy scalar `reference_wav_b64` (+ top-level
// `ref_text`), and the cookbooks' `references[]` array of {ref_audio, ref_text}.
//
// How many entries are allowed and what `ref_text` means is per-family and
// enforced by the caller, not here:
//   - moss_tts_local: one reference; `ref_text` selects continuation mode.
//   - delay family:   one reference per speaker, [S1], [S2], … order;
//                     transcripts belong in `text`, not in `ref_text`.
//
// Returns false and fills `err` on a malformed request. `out` is left empty
// when the request carries no reference at all.
struct RefEntry {
    std::string b64;
    std::string text;
};

bool extract_references(const json & body, std::vector<RefEntry> & out,
                        std::string & err) {
    if (body.contains("references")) {
        const json & refs = body["references"];
        if (!refs.is_array()) { err = "'references' must be an array"; return false; }
        for (size_t i = 0; i < refs.size(); ++i) {
            const json & r = refs[i];
            if (!r.is_object()) { err = "each entry in 'references' must be an object"; return false; }
            if (!r.contains("ref_audio") || !r["ref_audio"].is_string()) {
                err = "'references[" + std::to_string(i) + "].ref_audio' must be a base64 WAV string";
                return false;
            }
            RefEntry e;
            e.b64 = r["ref_audio"].get<std::string>();
            if (r.contains("ref_text") && r["ref_text"].is_string()) {
                e.text = r["ref_text"].get<std::string>();
            }
            out.push_back(std::move(e));
        }
        return true;
    }
    RefEntry e;
    if (body.contains("reference_wav_b64") && body["reference_wav_b64"].is_string()) {
        e.b64 = body["reference_wav_b64"].get<std::string>();
    }
    if (body.contains("ref_text") && body["ref_text"].is_string()) {
        e.text = body["ref_text"].get<std::string>();
    }
    if (!e.b64.empty() || !e.text.empty()) out.push_back(std::move(e));
    return true;
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
    DelayTemplate tpl = DelayTemplate::Auto;
    std::string voice_dir;
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
        else if (k == "--template") {
            const std::string v = next();
            if      (v == "auto")     tpl = DelayTemplate::Auto;
            else if (v == "ttsd")     tpl = DelayTemplate::TTSD;
            else if (v == "voicegen") tpl = DelayTemplate::VoiceGen;
            else {
                std::fprintf(stderr, "unknown --template: %s (expected auto, ttsd or voicegen)\n",
                             v.c_str());
                usage(2);
            }
        }
        else if (k == "--voice-dir")      voice_dir     = next();
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

    // Resolve which delay checkpoint this server is treating the model as.
    // Explicit --template wins; Auto keeps the historical n_vq guess.
    if (tpl != DelayTemplate::Auto &&
        model->dims().arch != openmoss::Arch::TTSDelay) {
        std::fprintf(stderr,
            "--template applies to the delay family only; this model is %s\n",
            openmoss::arch_name(model->dims().arch));
        return 2;
    }
    const bool delay_arch    = model->dims().arch == openmoss::Arch::TTSDelay;
    const bool ttsd_mode     = tpl == DelayTemplate::TTSD;
    const bool voicegen_mode = tpl == DelayTemplate::VoiceGen ||
        (tpl == DelayTemplate::Auto && delay_arch && model->dims().n_vq < 32);
    if (delay_arch) {
        std::fprintf(stderr, "[server] delay template: %s%s\n",
                     ttsd_mode ? "ttsd" : voicegen_mode ? "voicegen" : "moss_tts",
                     tpl == DelayTemplate::Auto
                         ? " (guessed from n_vq — pass --template ttsd when serving MOSS-TTSD)"
                         : "");
    }
    const openmoss::SamplingConfig base_sampling =
        default_sampling(model->dims().arch, ttsd_mode, voicegen_mode);

    // Shared by /tts and /v1/audio/speech: pull references out of the body,
    // decode them, and put them on the request in the family's own shape —
    // moss_tts_local's single reference (+ optional ref_text continuation), or
    // the delay family's one-reference-per-speaker with the concatenation
    // feeding the assistant-side continuation prefix, exactly as the CLI's
    // --speaker-ref does. Returns false and fills `err` (a 400) on a request
    // the family cannot honor.
    auto apply_references = [&](const json & body, openmoss::GenerateRequest & req,
                                std::string & err) -> bool {
        std::vector<RefEntry> refs;
        if (!extract_references(body, refs, err)) return false;
        if (refs.empty()) return true;
        for (const auto & r : refs) {
            if (r.b64.empty()) {
                err = "ref_text was supplied without reference audio; continuation "
                      "mode needs both";
                return false;
            }
        }
        if (!delay_arch && refs.size() > 1) {
            err = "only one reference is supported for " +
                  std::string(openmoss::arch_name(model->dims().arch)) +
                  "; got " + std::to_string(refs.size());
            return false;
        }
        if (delay_arch) {
            for (const auto & r : refs) {
                if (!r.text.empty()) {
                    err = "the delay family takes reference transcripts in `text` "
                          "(\"[S1] <transcript1> [S2] <transcript2> [S1] <turn> …\"), "
                          "not in ref_text — that field selects moss_tts_local's "
                          "continuation mode";
                    return false;
                }
            }
        } else if (!refs[0].text.empty()) {
            req.ref_text = refs[0].text;
        }
        std::vector<std::vector<float>> wavs;
        for (size_t i = 0; i < refs.size(); ++i) {
            try {
                auto wav_bytes = b64_decode(refs[i].b64);
                // Channel-aware for the same reason the encoder is: the codec
                // for MOSS-TTS-Local expects a 2-channel interleaved reference,
                // and decode_wav adapts the file (downmix, or duplicate a mono
                // one) rather than silently handing over half a signal.
                wavs.push_back(openmoss::decode_wav(
                    wav_bytes.data(), wav_bytes.size(),
                    model->dims().sampling_rate,
                    model->dims().n_channels));
            } catch (const std::exception & e) {
                err = "could not decode reference audio";
                if (refs.size() > 1) err += " [" + std::to_string(i) + "]";
                err += ": ";
                err += e.what();
                return false;
            }
        }
        if (wavs.size() == 1) {
            req.reference_wav = std::move(wavs[0]);
        } else {
            // One reference per speaker -> its own [S{i}] block in the user
            // turn; their concatenation feeds the assistant continuation prefix.
            std::vector<float> concat;
            for (const auto & w : wavs) concat.insert(concat.end(), w.begin(), w.end());
            req.reference_wavs = std::move(wavs);
            req.reference_wav  = std::move(concat);
        }
        return true;
    };

    // The pipeline refuses to stream together with the continuation prefix (the
    // reference audio would be streamed back before it could be trimmed), but by
    // the time it throws the status line is long gone and the client sees a
    // truncated stream. Catch the combination while a 400 is still possible.
    auto stream_prefix_error = [&](const openmoss::GenerateRequest & req,
                                   bool want_stream) -> const char * {
        const bool prefix_on =
            req.continuation_prefix.value_or(model->dims().n_vq < 32);
        if (want_stream && delay_arch && prefix_on &&
            (req.reference_wav || !req.reference_wavs.empty() ||
             !req.reference_codes.empty())) {
            return "streaming is not supported together with the continuation "
                   "prefix: the reference audio would be streamed back before it "
                   "could be trimmed. Pass continuation_prefix=false (the model "
                   "then re-speaks the reference transcripts) or drop stream.";
        }
        return nullptr;
    };

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

    // ── voice registry state ──────────────────────────────────────────────
    VoiceStore store{voice_dir};
    if (store.enabled()) {
        std::error_code ec;
        std::filesystem::create_directories(store.dir, ec);
        if (ec) {
            std::fprintf(stderr, "--voice-dir %s: cannot create directory (%s)\n",
                         store.dir.c_str(), ec.message().c_str());
            return 2;
        }
        if (model->dims().arch == openmoss::Arch::SoundEffect) {
            std::fprintf(stderr,
                "--voice-dir applies to the TTS families only; this model is moss_soundeffect\n");
            return 2;
        }
        std::fprintf(stderr, "[server] voice registry at %s (%zu voices)\n",
                     store.dir.c_str(), store.list().size());
    }
    std::mutex reg_mu;   // guards codes_cache and store mutations
    std::map<std::string, openmoss::EncodedReference> codes_cache;

    // Resolve registered voice ids to cached reference codes, encoding on
    // first use. The encode is model state, so it runs under the generation
    // mutex; every later request is a cache hit and skips it (~25 s per
    // reference on CPU). Fills req.reference_codes and the transcripts, in
    // order; false + `err` (a 400 text) on any unknown id.
    auto resolve_voices = [&](const std::vector<std::string> & ids,
                              openmoss::GenerateRequest & req,
                              std::vector<std::string> & transcripts,
                              std::string & err) -> bool {
        for (const auto & id : ids) {
            if (!valid_voice_id(id) || !store.exists(id)) {
                std::string all;
                for (const auto & k : store.list()) {
                    if (!all.empty()) all += ", ";
                    all += k;
                }
                err = "unknown voice '" + id + "'; registered: [" + all + "]";
                return false;
            }
        }
        for (const auto & id : ids) {
            VoiceMeta m;
            store.load_meta(id, m);
            transcripts.push_back(m.transcript);
            {
                std::lock_guard<std::mutex> rg(reg_mu);
                auto it = codes_cache.find(id);
                if (it != codes_cache.end()) {
                    req.reference_codes.push_back(it->second);
                    continue;
                }
            }
            auto wav_bytes = store.load_wav(id);
            if (wav_bytes.empty()) {
                err = "voice '" + id + "': cannot read its WAV from the registry";
                return false;
            }
            std::vector<float> wav;
            try {
                wav = openmoss::decode_wav(wav_bytes.data(), wav_bytes.size(),
                                           model->dims().sampling_rate,
                                           model->dims().n_channels);
            } catch (const std::exception & e) {
                err = "voice '" + id + "': " + e.what();
                return false;
            }
            openmoss::EncodedReference enc;
            try {
                std::lock_guard<std::mutex> g(gen_mu);
                const auto t0 = std::chrono::steady_clock::now();
                enc = openmoss::encode_reference(*model, std::move(wav));
                std::fprintf(stderr,
                    "[server] voice '%s' encoded: %d frames in %.2fs (now cached)\n",
                    id.c_str(), enc.n_frames,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
            } catch (const std::exception & e) {
                err = "voice '" + id + "': encode failed: ";
                err += e.what();
                return false;
            }
            {
                std::lock_guard<std::mutex> rg(reg_mu);
                codes_cache[id] = enc;
            }
            req.reference_codes.push_back(std::move(enc));
        }
        return true;
    };

    // Registered-voice request path: resolve ids to cached codes and, for the
    // delay family, prepend the stored transcripts so `text` carries only the
    // new material ("[S1] <t1> [S2] <t2> " for dialogues — the continuation
    // prefix means the model treats the transcripts as already spoken).
    // Callers have already checked mutual exclusion with inline references.
    auto use_voices = [&](const std::vector<std::string> & ids,
                          openmoss::GenerateRequest & req,
                          std::string & err) -> bool {
        if (!store.enabled()) {
            err = "voice registry disabled: start the server with --voice-dir";
            return false;
        }
        if (ids.size() > 1 && !delay_arch) {
            err = "'voices' (multi-speaker) is only supported by the delay family";
            return false;
        }
        std::vector<std::string> transcripts;
        if (!resolve_voices(ids, req, transcripts, err)) return false;
        if (delay_arch) {
            std::string prefix;
            for (size_t i = 0; i < transcripts.size(); ++i) {
                if (ids.size() > 1) prefix += "[S" + std::to_string(i + 1) + "] ";
                prefix += transcripts[i];
                if (!prefix.empty() && prefix.back() != ' ') prefix += ' ';
            }
            req.text = prefix + req.text;
        }
        return true;
    };

    // Pull `voice` (string) / `voices` (array) out of a request body. Returns
    // false + err on a malformed combination; `ids` stays empty when the
    // request names no registered voice ("" and "default" mean none).
    auto extract_voice_ids = [](const json & body, std::vector<std::string> & ids,
                                std::string & err) -> bool {
        if (body.contains("voices")) {
            if (!body["voices"].is_array()) {
                err = "'voices' must be an array of registered voice ids";
                return false;
            }
            for (const auto & v : body["voices"]) {
                if (!v.is_string()) {
                    err = "'voices' must be an array of strings";
                    return false;
                }
                ids.push_back(v.get<std::string>());
            }
        }
        if (body.contains("voice") && body["voice"].is_string()) {
            const std::string v = body["voice"].get<std::string>();
            if (!v.empty() && v != "default") {
                if (!ids.empty()) {
                    err = "pass either 'voice' or 'voices', not both";
                    return false;
                }
                ids.push_back(v);
            }
        }
        return true;
    };

    svr.Get("/health", [](const httplib::Request &, httplib::Response & rs) {
        rs.set_content("ok\n", "text/plain");
    });

    svr.Get("/info", [&](const httplib::Request &, httplib::Response & rs) {
        const auto & d = model->dims();
        json info = {
            {"version",           OPENMOSS_VERSION},
            {"architecture",      openmoss::arch_name(d.arch)},
            {"sampling_rate",     d.sampling_rate},
            {"n_channels",        d.n_channels},
            {"n_vq",              d.n_vq},
            {"audio_vocab_size",  d.audio_vocab_size},
            {"hidden_size",       d.hidden_size},
            {"frame_rate_hz",     double(d.sampling_rate) / double(d.downsample_rate)},
            {"codec_present",     model->codec_present()},
            {"codec_loaded",      model->codec_loaded()},
            {"requests_served",   uint64_t(n_requests.load())},
        };
        // Which checkpoint the delay family is being served as — the GGUF alone
        // cannot tell MOSS-TTSD and MOSS-VoiceGenerator apart (both n_vq=16),
        // so a gateway should check this matches the model it deployed.
        if (d.arch == openmoss::Arch::TTSDelay) {
            info["prompt_template"] =
                ttsd_mode ? "ttsd" : voicegen_mode ? "voicegen" : "moss_tts";
        }
        info["voice_registry"] = store.enabled();
        if (store.enabled()) info["n_voices"] = store.list().size();
        // MOSS-SoundEffect has no codec and no codebooks; what a caller needs to
        // know is the solver surface and the fixed duration ceiling.
        if (d.arch == openmoss::Arch::SoundEffect) {
            info["max_seconds"]         = d.max_seconds;
            info["default_steps"]       = 100;
            info["default_cfg_scale"]   = 4.0;
            info["default_sigma_shift"] = d.sched_shift;
            info["latent_frames"]       = int64_t(d.sampling_rate) * d.max_seconds / d.vae_hop;
        }
        rs.set_content(info.dump(), "application/json");
    });

    // OpenAI-compatible model list. Only ever one entry: the model is loaded at
    // startup and cannot be switched at runtime.
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & rs) {
        json j = {
            {"object", "list"},
            {"data", json::array({ json{
                {"id",       openmoss::arch_name(model->dims().arch)},
                {"object",   "model"},
                {"owned_by", "openmoss"},
                {"created",  0},
            }})},
        };
        rs.set_content(j.dump(), "application/json");
    });

    // Voices registry. There is no upstream standard for this — every MOSS
    // cookbook passes voice="default" — so the shape follows the vLLM-Omni
    // convention, which is the closest thing to one. For the TTS families a
    // "voice" is really just an instruction string, and cloning is driven by
    // reference audio rather than a named voice; MOSS-SoundEffect has no notion
    // of a voice at all and reports an empty list.
    svr.Get("/v1/audio/voices", [&](const httplib::Request &, httplib::Response & rs) {
        json voices = json::array();
        if (model->dims().arch != openmoss::Arch::SoundEffect) {
            voices.push_back(json{
                {"id",          "default"},
                {"name",        "default"},
                {"description", "the model's own voice; pass reference audio to clone another"},
            });
            if (store.enabled()) {
                for (const auto & id : store.list()) {
                    VoiceMeta m;
                    store.load_meta(id, m);
                    voices.push_back(json{
                        {"id",          id},
                        {"name",        id},
                        {"description", m.description.empty() ? m.transcript
                                                              : m.description},
                    });
                }
            }
        }
        rs.set_content(json{{"object", "list"}, {"data", voices}}.dump(),
                       "application/json");
    });

    // ── voice registry endpoints (contract in the header comment) ─────────
    auto registry_disabled = [&](httplib::Response & rs) -> bool {
        if (store.enabled()) return false;
        send_text_error(rs, 400,
            "voice registry disabled: start the server with --voice-dir");
        return true;
    };

    svr.Post("/v1/voices", [&](const httplib::Request & rq, httplib::Response & rs) {
        if (registry_disabled(rs)) return;
        auto field = [&](const char * n) -> std::string {
            return rq.has_file(n) ? rq.get_file_value(n).content : std::string();
        };
        std::string id = field("voice_id");
        if (id.empty()) id = field("id");
        std::string audio      = field("audio");
        std::string transcript = field("transcript");
        if (transcript.empty()) transcript = field("text");
        if (transcript.empty()) transcript = field("description");
        if (!valid_voice_id(id) || audio.empty() || transcript.empty()) {
            return send_text_error(rs, 400,
                "voice_id (alnum/_/-, <=64), audio and transcript are required "
                "(multipart form-data)");
        }
        // Validate the WAV now — a broken reference should fail at
        // registration, not on its first use in a request.
        try {
            (void)openmoss::decode_wav(
                reinterpret_cast<const uint8_t *>(audio.data()), audio.size(),
                model->dims().sampling_rate, model->dims().n_channels);
        } catch (const std::exception & e) {
            return send_text_error(rs, 400,
                std::string("audio is not a usable WAV: ") + e.what());
        }
        std::lock_guard<std::mutex> rg(reg_mu);
        if (store.exists(id)) {
            return send_text_error(rs, 409, "voice '" + id + "' already exists");
        }
        if (!store.save_wav(id, audio)) {
            return send_text_error(rs, 500, "failed to persist voice WAV");
        }
        VoiceMeta m;
        m.transcript  = transcript;
        m.description = field("description");
        m.language    = field("language");
        m.created_at  = iso8601_now();
        if (!store.save_meta(id, m)) {
            std::error_code ec;
            std::filesystem::remove(store.wav_path(id), ec);
            return send_text_error(rs, 500, "failed to persist voice metadata");
        }
        std::fprintf(stderr, "[server] voice '%s' registered (%zu bytes, lang=%s)\n",
                     id.c_str(), audio.size(), m.language.c_str());
        rs.status = 201;
        rs.set_content(voice_meta_json(id, m).dump(), "application/json");
    });

    svr.Get("/v1/voices", [&](const httplib::Request &, httplib::Response & rs) {
        if (registry_disabled(rs)) return;
        json out = json::array();
        for (const auto & id : store.list()) {
            VoiceMeta m;
            store.load_meta(id, m);
            out.push_back(voice_meta_json(id, m));
        }
        rs.set_content(json{{"voices", out}}.dump(), "application/json");
    });

    svr.Get(R"(/v1/voices/([A-Za-z0-9_\-]{1,64}))",
            [&](const httplib::Request & rq, httplib::Response & rs) {
        if (registry_disabled(rs)) return;
        const std::string id = rq.matches[1];
        if (!store.exists(id)) return send_text_error(rs, 404, "voice not found");
        VoiceMeta m;
        store.load_meta(id, m);
        rs.set_content(voice_meta_json(id, m).dump(), "application/json");
    });

    svr.Delete(R"(/v1/voices/([A-Za-z0-9_\-]{1,64}))",
               [&](const httplib::Request & rq, httplib::Response & rs) {
        if (registry_disabled(rs)) return;
        const std::string id = rq.matches[1];
        std::lock_guard<std::mutex> rg(reg_mu);
        if (!store.exists(id)) return send_text_error(rs, 404, "voice not found");
        if (!store.remove(id)) return send_text_error(rs, 500, "failed to delete voice");
        codes_cache.erase(id);
        std::fprintf(stderr, "[server] voice '%s' deleted\n", id.c_str());
        rs.set_content(json{{"id", id}, {"deleted", true}}.dump(),
                       "application/json");
    });

    // ── MOSS-SoundEffect ──────────────────────────────────────────────────
    //
    // Shared by POST /sfx and by /v1/audio/speech when the loaded model is this
    // family. Not autoregressive, so none of the TTS sampling knobs apply.
    auto parse_sfx = [&](const json & body, std::string & err) -> openmoss::SoundEffectRequest {
        openmoss::SoundEffectRequest req;
        // /sfx takes "prompt"; /v1/audio/speech takes "input". Accept either.
        for (const char * k : {"prompt", "input", "text"}) {
            if (body.contains(k) && body[k].is_string()) { req.prompt = body[k].get<std::string>(); break; }
        }
        if (req.prompt.empty()) { err = "missing required field 'prompt' (string)"; return req; }
        req.seconds             = jget(body, "seconds",             req.seconds);
        req.num_inference_steps = jget(body, "steps",               req.num_inference_steps);
        req.cfg_scale           = jget(body, "cfg_scale",           req.cfg_scale);
        req.sigma_shift         = jget(body, "sigma_shift",         req.sigma_shift);
        req.negative_prompt     = jget(body, "negative_prompt",     req.negative_prompt);
        req.seed                = jget(body, "seed",                req.seed);
        req.append_duration_suffix =
            jget(body, "append_duration_suffix", req.append_duration_suffix);
        if (req.num_inference_steps < 1)  err = "steps must be >= 1";
        if (req.seconds <= 0.0f)          err = "seconds must be > 0";
        return req;
    };

    // Chunked-transfer TTS: hand each codec chunk to the socket as it is
    // decoded, instead of buffering the whole utterance.
    //
    // Two consequences the caller has to live with, both inherent to answering
    // before the audio exists:
    //
    //   * The format is raw PCM, not WAV. A RIFF header states its own data
    //     length in its first 44 bytes, and that is not known until generation
    //     stops. Rather than emit a header that lies, the endpoint rejects
    //     response_format=wav for a streamed request and reports the rate and
    //     channel count in headers instead.
    //
    //   * The status line goes out before generation starts, so a failure
    //     partway through truncates the stream — it cannot become a 500. The
    //     request is validated fully before we commit to streaming, which
    //     leaves model-level failures as the only case, and those already show
    //     up in the server log.
    //
    // The provider runs on the connection's own thread after the handler
    // returns, so `req` is captured by value and the generation mutex is taken
    // inside it — requests still serialise exactly as they did before.
    auto stream_tts = [&](openmoss::GenerateRequest req, uint64_t req_id,
                          httplib::Response & rs) {
        const auto  t0       = std::chrono::steady_clock::now();
        const int32_t rate   = model->dims().sampling_rate;
        const int32_t chans  = model->dims().n_channels;

        rs.set_header("X-MOSS-Sample-Rate", std::to_string(rate));
        rs.set_header("X-MOSS-Channels",    std::to_string(chans));
        rs.set_header("X-MOSS-Stream-Chunk-Frames",
                      std::to_string(req.stream_chunk_frames));

        rs.set_chunked_content_provider("audio/pcm",
            [this_model = model.get(), &gen_mu, req, req_id, t0, rate, chans]
            (size_t /*offset*/, httplib::DataSink & sink) -> bool {
                // Everything happens in the first invocation; sink.done() ends it.
                int64_t total = 0;
                bool    first = true;
                double  ttfa  = 0.0;
                try {
                    std::lock_guard<std::mutex> g(gen_mu);
                    std::fprintf(stderr,
                        "[server] req#%llu stream text=%zu chars ref=%s chunk=%d frames\n",
                        (unsigned long long)req_id, req.text.size(),
                        req.reference_wav ? "yes" : "no", req.stream_chunk_frames);
                    openmoss::generate(*this_model, req,
                        [&](const float * pcm, int64_t n_samples) {
                            const auto bytes = openmoss::encode_pcm_s16le(pcm, n_samples);
                            if (!sink.write(reinterpret_cast<const char *>(bytes.data()),
                                            bytes.size())) {
                                // The client hung up. There is no way to ask
                                // generate() to stop politely, so unwind — the
                                // catch below treats it as a normal end.
                                throw StreamAborted{};
                            }
                            total += n_samples;
                            if (first) {
                                ttfa = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - t0).count();
                                first = false;
                            }
                        });
                } catch (const StreamAborted &) {
                    std::fprintf(stderr, "[server] req#%llu stream: client disconnected "
                                 "after %lld samples\n",
                                 (unsigned long long)req_id, (long long)total);
                } catch (const std::exception & e) {
                    std::fprintf(stderr, "[server] req#%llu stream FAILED after %lld "
                                 "samples: %s\n",
                                 (unsigned long long)req_id, (long long)total, e.what());
                }
                const double total_s = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr,
                    "[server] req#%llu stream done: %lld samples (%.2fs audio) — "
                    "first audio %.2fs, total %.2fs\n",
                    (unsigned long long)req_id, (long long)total,
                    double(total) / double(rate) / double(chans), ttfa, total_s);
                sink.done();
                return true;
            });
    };

    auto handle_sfx = [&](const json & body, const std::string & fmt,
                          uint64_t req_id, httplib::Response & rs) {
        const auto t0 = std::chrono::steady_clock::now();
        std::string err;
        openmoss::SoundEffectRequest req = parse_sfx(body, err);
        if (!err.empty()) return send_text_error(rs, 400, err);

        openmoss::SoundEffectResult result;
        try {
            std::lock_guard<std::mutex> g(gen_mu);
            std::fprintf(stderr,
                "[server] req#%llu /sfx prompt=%zu chars seconds=%.1f steps=%d cfg=%.2f\n",
                (unsigned long long)req_id, req.prompt.size(),
                double(req.seconds), req.num_inference_steps, double(req.cfg_scale));
            result = openmoss::generate_sound_effect(*model, req);
        } catch (const std::exception & e) {
            return send_text_error(rs, 500, std::string("generation failed: ") + e.what());
        }
        if (result.waveform.empty()) {
            return send_text_error(rs, 500, "generation produced no audio");
        }

        auto enc = encode_audio(fmt, result.waveform.data(),
                                int64_t(result.waveform.size()),
                                result.sampling_rate, /*channels=*/1);
        const std::chrono::duration<double> total_s = std::chrono::steady_clock::now() - t0;
        std::fprintf(stderr,
            "[server] req#%llu ok: %zu samples (%.2fs audio) — text %.2fs solve %.2fs decode %.2fs total %.2fs\n",
            (unsigned long long)req_id, result.waveform.size(),
            result.waveform.size() / double(result.sampling_rate),
            result.text_seconds, result.sample_seconds, result.decode_seconds,
            total_s.count());

        rs.set_header("X-MOSS-Prompt", result.prompt);
        rs.set_header("X-MOSS-Latent-Frames", std::to_string(result.n_latents));
        rs.set_header("X-MOSS-Solve-Seconds", std::to_string(result.sample_seconds));
        rs.set_header("X-MOSS-Decode-Seconds", std::to_string(result.decode_seconds));
        rs.set_content(reinterpret_cast<const char *>(enc.bytes.data()),
                       enc.bytes.size(), enc.mime);
    };

    svr.Post("/sfx", [&](const httplib::Request & rq, httplib::Response & rs) {
        const uint64_t req_id = ++n_requests;
        if (model->dims().arch != openmoss::Arch::SoundEffect) {
            return send_text_error(rs, 400,
                std::string("/sfx requires a moss_soundeffect model; this server loaded ")
                + openmoss::arch_name(model->dims().arch) + " — use /tts");
        }
        json body;
        try { body = json::parse(rq.body); }
        catch (const std::exception & e) {
            return send_text_error(rs, 400, std::string("invalid JSON body: ") + e.what());
        }
        if (!body.is_object()) return send_text_error(rs, 400, "JSON body must be an object");
        std::string fmt = jget(body, "response_format", std::string("wav"));
        if (fmt != "wav" && fmt != "pcm") {
            return send_text_error(rs, 400,
                "unsupported response_format '" + fmt + "'; use 'wav' or 'pcm'");
        }
        // Refuse rather than quietly ignore: a caller asking for a stream and
        // getting one 90-second response instead would look like a hang.
        if (jget(body, "stream", false)) {
            return send_text_error(rs, 400,
                "moss_soundeffect cannot stream: flow matching refines every latent "
                "frame simultaneously, so no prefix of the audio exists until the last "
                "solver step finishes");
        }
        handle_sfx(body, fmt, req_id, rs);
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
        // One endpoint per family: /tts drives the autoregressive path, which
        // MOSS-SoundEffect has no equivalent of.
        if (model->dims().arch == openmoss::Arch::SoundEffect) {
            return send_text_error(rs, 400,
                "this server loaded moss_soundeffect, which is not autoregressive — use /sfx");
        }
        if (!body.contains("text") || !body["text"].is_string())
            return send_text_error(rs, 400, "missing required field 'text' (string)");

        std::string fmt = jget(body, "response_format", std::string("wav"));
        if (fmt != "wav" && fmt != "pcm") {
            return send_text_error(rs, 400,
                "unsupported response_format '" + fmt + "'; use 'wav' or 'pcm'");
        }
        const bool want_stream = jget(body, "stream", false);
        if (want_stream && fmt != "pcm") {
            return send_text_error(rs, 400,
                "stream=true requires response_format 'pcm': a WAV header declares its "
                "own data length, which is not known until generation ends. The "
                "X-MOSS-Sample-Rate and X-MOSS-Channels response headers carry what a "
                "player needs.");
        }

        openmoss::GenerateRequest req;
        req.text          = body["text"].get<std::string>();
        // Duration hint, either as a field or as the cookbooks' inline prefix.
        // An explicit field wins over the prefix.
        if (const int inline_n = extract_token_prefix(req.text); inline_n > 0) {
            req.tokens = inline_n;
        }
        if (body.contains("token_count") && body["token_count"].is_number_integer())
            req.tokens      = body["token_count"].get<int>();
        if (body.contains("instruction") && body["instruction"].is_string())
            req.instruction = body["instruction"].get<std::string>();
        if (body.contains("language") && body["language"].is_string())
            req.language    = body["language"].get<std::string>();
        if (body.contains("quality") && body["quality"].is_string())
            req.quality     = body["quality"].get<std::string>();
        if (body.contains("tokens") && body["tokens"].is_number_integer())
            req.tokens      = body["tokens"].get<int>();
        req.max_new_tokens  = jget(body, "max_new_tokens", req.max_new_tokens);
        req.stream_chunk_frames =
            std::max(1, jget(body, "stream_chunk_frames", req.stream_chunk_frames));
        if (ttsd_mode) req.prompt_template = openmoss::PromptTemplate::TTSD;
        if (auto it = body.find("continuation_prefix");
            it != body.end() && !it->is_null()) {
            req.continuation_prefix = it->get<bool>();
        }
        req.sampling        = parse_sampling(body.value("sampling", json::object()),
                                             base_sampling);
        finalize_voicegen_request(req, voicegen_mode);

        {
            std::string ref_err;
            if (!apply_references(body, req, ref_err)) {
                return send_text_error(rs, 400, ref_err);
            }
        }
        {
            std::vector<std::string> voice_ids;
            std::string verr;
            if (!extract_voice_ids(body, voice_ids, verr)) {
                return send_text_error(rs, 400, verr);
            }
            if (!voice_ids.empty()) {
                if (req.reference_wav || !req.reference_wavs.empty()) {
                    return send_text_error(rs, 400,
                        "pass either registered voices or inline references, not both");
                }
                if (!use_voices(voice_ids, req, verr)) {
                    return send_text_error(rs, 400, verr);
                }
            }
        }
        if (const char * msg = stream_prefix_error(req, want_stream)) {
            return send_text_error(rs, 400, msg);
        }

        if (want_stream) return stream_tts(std::move(req), req_id, rs);

        openmoss::GenerateResult result;
        try {
            std::lock_guard<std::mutex> g(gen_mu);
            std::fprintf(stderr,
                "[server] req#%llu text=%zu chars refs=%zu max=%d\n",
                (unsigned long long)req_id,
                req.text.size(),
                !req.reference_codes.empty()
                    ? req.reference_codes.size()
                    : (req.reference_wavs.empty() ? size_t(req.reference_wav ? 1 : 0)
                                                  : req.reference_wavs.size()),
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

        // Channel-aware: MOSS-TTS-Local is 48 kHz *stereo*, and the mono
        // encoder used to write its interleaved waveform under a 1-channel
        // header, halving the playback rate.
        auto enc = encode_audio(fmt, result.waveform.data(),
                                int64_t(result.waveform.size()),
                                model->dims().sampling_rate, result.n_channels);

        const std::chrono::duration<double> total_s =
            std::chrono::steady_clock::now() - t0;
        std::fprintf(stderr,
            "[server] req#%llu ok: %zu samples (%.2fs audio) — prefill %.2fs gen %.2fs decode %.2fs total %.2fs\n",
            (unsigned long long)req_id,
            result.waveform.size(),
            result.waveform.size()
                / double(model->dims().sampling_rate) / double(result.n_channels),
            result.prefill_seconds,
            result.generate_seconds,
            result.decode_seconds,
            total_s.count());

        rs.set_header("X-MOSS-Audio-Frames", std::to_string(result.n_audio_frames));
        rs.set_header("X-MOSS-Generate-Seconds",
                       std::to_string(result.generate_seconds));
        rs.set_header("X-MOSS-Decode-Seconds",
                       std::to_string(result.decode_seconds));
        rs.set_header("X-MOSS-Channels", std::to_string(result.n_channels));
        rs.set_content(reinterpret_cast<const char *>(enc.bytes.data()),
                        enc.bytes.size(), enc.mime);
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

        std::string fmt = jget(body, "response_format", std::string("wav"));
        if (fmt != "wav" && fmt != "pcm") {
            return send_text_error(rs, 400,
                "unsupported response_format '" + fmt + "'; use 'wav' or 'pcm'");
        }
        const bool want_stream = jget(body, "stream", false);
        if (want_stream && fmt != "pcm") {
            return send_text_error(rs, 400,
                "stream=true requires response_format 'pcm': a WAV header declares its "
                "own data length, which is not known until generation ends. The "
                "X-MOSS-Sample-Rate and X-MOSS-Channels response headers carry what a "
                "player needs.");
        }

        // MOSS-SoundEffect answers this route too, treating `input` as the sound
        // description. Its solver options are accepted as extensions; `voice` and
        // `speed` have no meaning for it and are ignored.
        if (model->dims().arch == openmoss::Arch::SoundEffect) {
            if (want_stream) {
                return send_text_error(rs, 400,
                    "this server loaded moss_soundeffect, which cannot stream: flow "
                    "matching refines every latent frame simultaneously, so no prefix "
                    "of the audio exists until the last solver step finishes");
            }
            return handle_sfx(body, fmt, req_id, rs);
        }

        // Translate OpenAI fields → native MOSS GenerateRequest.
        openmoss::GenerateRequest req;
        req.text = body["input"].get<std::string>();
        if (const int inline_n = extract_token_prefix(req.text); inline_n > 0) {
            req.tokens = inline_n;
        }
        if (body.contains("token_count") && body["token_count"].is_number_integer())
            req.tokens = body["token_count"].get<int>();

        // "voice": with the registry enabled it names a registered voice
        // ("" and "default" mean none); without --voice-dir it stays the
        // legacy instruction hint (e.g. "alloy") it always was — except that
        // ""/"default" no longer become a literal instruction string.
        std::vector<std::string> voice_ids;
        if (store.enabled()) {
            std::string verr;
            if (!extract_voice_ids(body, voice_ids, verr)) {
                return send_text_error(rs, 400, verr);
            }
        } else {
            if (body.contains("voices")) {
                return send_text_error(rs, 400,
                    "voice registry disabled: start the server with --voice-dir");
            }
            if (body.contains("voice") && body["voice"].is_string()) {
                const std::string v = body["voice"].get<std::string>();
                if (!v.empty() && v != "default") req.instruction = v;
            }
        }

        // Voice cloning: base64 WAV reference(s) the model continues in.
        {
            std::string ref_err;
            if (!apply_references(body, req, ref_err)) {
                return send_text_error(rs, 400, ref_err);
            }
        }
        if (!voice_ids.empty()) {
            if (req.reference_wav || !req.reference_wavs.empty()) {
                return send_text_error(rs, 400,
                    "pass either registered voices or inline references, not both");
            }
            std::string verr;
            if (!use_voices(voice_ids, req, verr)) {
                return send_text_error(rs, 400, verr);
            }
        }

        // "speed" scales the token budget.  speed=1.0 → default max_new_tokens.
        // We approximate: 1s of audio ≈ 12.5 tokens, and the default model output
        // is roughly 4096 tokens ≈ 328s.  speed just multiplies the budget.
        double speed = jget(body, "speed", 1.0);
        if (speed < 0.25 || speed > 4.0) {
            return send_text_error(rs, 400,
                "speed must be between 0.25 and 4.0");
        }
        // Scale the default token budget by speed (lower speed = fewer tokens = shorter audio).
        req.max_new_tokens = std::max(1, int(4096 / speed));
        // Sampling can arrive nested under "sampling" (our own shape) or
        // flattened at the top level, which is what the cookbooks send. The
        // nested form is applied first so explicit flat keys win.
        if (ttsd_mode) req.prompt_template = openmoss::PromptTemplate::TTSD;
        if (auto it = body.find("continuation_prefix");
            it != body.end() && !it->is_null()) {
            req.continuation_prefix = it->get<bool>();
        }
        req.sampling       = parse_sampling(body.value("sampling", json::object()),
                                            base_sampling);
        req.sampling       = parse_sampling(body, req.sampling);
        req.stream_chunk_frames =
            std::max(1, jget(body, "stream_chunk_frames", req.stream_chunk_frames));
        finalize_voicegen_request(req, voicegen_mode);

        if (const char * msg = stream_prefix_error(req, want_stream)) {
            return send_text_error(rs, 400, msg);
        }
        if (want_stream) return stream_tts(std::move(req), req_id, rs);

        openmoss::GenerateResult result;
        try {
            std::lock_guard<std::mutex> g(gen_mu);
            std::fprintf(stderr,
                "[server] req#%llu /v1/audio/speech input=%zu chars speed=%.2f max=%d ref=%s\n",
                (unsigned long long)req_id,
                req.text.size(),
                speed,
                req.max_new_tokens,
                req.reference_wav ? "yes" : "no");
            result = openmoss::generate(*model, req);
        } catch (const std::exception & e) {
            return send_text_error(rs, 500,
                std::string("generation failed: ") + e.what());
        }

        if (result.waveform.empty()) {
            return send_text_error(rs, 500,
                "generation produced no audio (codec missing or model emitted EOS too early)");
        }

        auto enc = encode_audio(fmt, result.waveform.data(),
                                int64_t(result.waveform.size()),
                                model->dims().sampling_rate, result.n_channels);

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
        rs.set_header("X-MOSS-Channels", std::to_string(result.n_channels));
        rs.set_content(reinterpret_cast<const char *>(enc.bytes.data()),
                        enc.bytes.size(), enc.mime);
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