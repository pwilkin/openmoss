// SPDX-License-Identifier: Apache-2.0
//
// High-level end-to-end TTS pipeline: text (+ optional reference audio) →
// audio codes → waveform.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "openmoss/delay.h"
#include "openmoss/model.h"

namespace openmoss {

// In-house: which <user_inst> template to emit for the delay family. The two
// checkpoints ship *different* processors: MOSS-TTSD hard-codes "- Tokens:\nNone"
// and carries an extra "- Scene:" field (its processing_moss_tts.py:79-80, 89-90,
// 117); MOSS-VoiceGenerator has a real {tokens} placeholder and no Scene field
// (its processing_moss_tts.py:141-142). VoiceGen is the default and matches the
// prompt layout upstream's moss-prompt-probe validates.
enum class PromptTemplate { VoiceGen, TTSD };

// In-house: a codec-encoded reference; `codes` is (n_vq, n_frames) row-major.
// Produced by encode_reference() and consumed via
// GenerateRequest::reference_codes so callers can cache the encode.
struct EncodedReference {
    std::vector<int32_t> codes;
    int32_t              n_frames = 0;
};

struct GenerateRequest {
    std::string text;
    // Channel-interleaved f32 at the model's sampling rate / channel count.
    std::optional<std::vector<float>> reference_wav;
    // Transcript of `reference_wav`. Supplying it selects *continuation mode*:
    // the reference goes in the assistant channel behind the generation slot
    // with no closing audio_end, and the model continues speaking from it,
    // rather than being conditioned on it as a user-channel exemplar. The grid's
    // synthesis text becomes ref_text followed by `text` — the model has already
    // "said" the reference, so it carries on into the new material.
    // MOSS-TTS-Local only.
    std::optional<std::string>        ref_text;

    // ── In-house additions for the delay family (multi-speaker cloning) ────
    // Per-speaker reference wavs: each gets its own [S{i}] block in the user
    // turn, mirroring MossTTSDelayProcessor's reference list handling
    // (processing_moss_tts.py:98-106). When set, `reference_wav` must hold the
    // concatenation of all speakers — that is what feeds the assistant-side
    // continuation prefix.
    std::vector<std::vector<float>>   reference_wavs;
    // In-house: pre-encoded references, used INSTEAD of encoding
    // reference_wav/reference_wavs when non-empty. One entry = the
    // single-reference layout; several = one per speaker, [S1]..[Sn] (delay
    // family only; moss_tts_local takes exactly one). The reference encode
    // dominates request latency (~25 s for a 7 s reference on CPU), so a
    // caller reusing voices — the server's voice registry — encodes once via
    // encode_reference() and passes the codes here. Delay-family nuance: the
    // continuation prefix becomes the frame-wise concatenation of these
    // codes, which keeps it identical to the [S{i}] block codes; the wav path
    // re-encodes the concatenated waveform instead, so boundary frames can
    // differ slightly between the two paths.
    std::vector<EncodedReference>     reference_codes;
    PromptTemplate                    prompt_template = PromptTemplate::VoiceGen;
    // Assistant-side continuation prefix: encode `reference_wav` and place it
    // behind the assistant's audio_start as gen_slot rows (upstream's
    // mode="continuation" + truncation=True), so the model does not re-speak
    // the references and the delay ramp lands inside the prefix. The prefix
    // audio is trimmed from the output waveform. Delay family only — this is
    // the in-house feature upstream's STATUS.md lists as "not done".
    bool                              continuation_prefix = true;
    // Loudness-normalize each reference to −20 dBFS before encoding, as
    // upstream's processor does (processing_moss_tts.py:767-779).
    bool                              normalize_reference = true;

    std::optional<std::string>        instruction;     // e.g. voice description
    std::optional<std::string>        language;        // "en", "zh", ...
    std::optional<std::string>        quality;         // upstream "quality" hint
    std::optional<int>                tokens;          // duration hint (1s ≈ 12.5 tokens)
    int                               max_new_tokens = 4096;
    SamplingConfig                    sampling;

    // How many audio frames to accumulate before handing them to the codec,
    // when a StreamCallback is supplied. Ignored otherwise.
    //
    // A frame is 80 ms of audio, so this is the output granularity. It is also
    // the throughput/latency dial: an incremental decode costs ~50 ms of fixed
    // overhead plus its own compute, and that runs inline with generation, so
    // small chunks stall the LM more often. Measured on a Radeon 8060S with
    // MOSS-TTS-Local Q8 — 8 frames gives 0.64 s granularity at ~1.46x real time
    // against 1.74x for a batch decode; 32 frames recovers most of the
    // throughput at 2.56 s granularity.
    int                               stream_chunk_frames = 8;
};

struct GenerateResult {
    // Channel-interleaved f32 at the model's sampling rate. Length is
    // n_audio_frames * downsample_rate * n_channels.
    std::vector<float> waveform;
    int32_t            n_channels = 1;
    int32_t            n_audio_frames; // before upsampling

    // The raw codec codes behind `waveform`, (n_codebooks, n_audio_frames)
    // row-major. Kept so callers can diff against a reference implementation
    // without having to compare audio.
    std::vector<int32_t> audio_codes;
    int32_t              n_codebooks = 0;
    double             prefill_seconds = 0.0;
    double             generate_seconds = 0.0;
    double             decode_seconds   = 0.0;
};

// Optional callback invoked once the codec produces a chunk of waveform; lets
// callers stream output. The callback's argument is owned by the pipeline and
// only valid for the duration of the call.
using StreamCallback = std::function<void(const float * pcm, int64_t n_samples)>;

GenerateResult generate(Model & model,
                        const GenerateRequest & req,
                        StreamCallback cb = {});

// In-house: encode one reference waveform (channel-interleaved f32 at the
// model's rate) to codec codes, applying the family's normalization policy
// (delay family loudness-normalizes to −20 dBFS unless `normalize` is false;
// moss_tts_local does not normalize, matching generate()). This is the same
// path generate() takes internally — cache the result and pass it via
// GenerateRequest::reference_codes to skip the encode on later requests.
EncodedReference encode_reference(Model & model, std::vector<float> wav,
                                  bool normalize = true);

// Test seam: build the prompt grid alone, with reference codes supplied
// directly rather than encoded from audio. Prompt construction has no numerical
// tolerance — an id is right or it is not — so diffing ids against the
// reference processor is the cheapest correctness check available, and it needs
// neither the codec nor the backbone.
//   ref_codes: (n_vq, T_ref) row-major, or null
//   returns:   (n_pos, 1 + n_vq) row-major
std::vector<int32_t> debug_build_prompt_grid(Model & model,
                                             const GenerateRequest & req,
                                             const std::vector<int32_t> * ref_codes,
                                             int32_t  T_ref,
                                             int32_t & n_pos_out);

} // namespace openmoss
