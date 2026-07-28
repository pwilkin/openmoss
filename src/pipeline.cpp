// SPDX-License-Identifier: Apache-2.0
//
// End-to-end pipeline glue.
//
// What's wired up here:
//   - A prompt builder that mirrors the Python reference, including the
//     voice-clone reference path and (MOSS-TTS-Local only) continuation mode.
//   - The autoregressive generation loop:
//       1) build prompt grid (S, 1+n_vq) of int32
//       2) compute summed input embeddings (S, hidden) on the aux GGML backend
//       3) prefill libllama with batch.embd
//       4) at each step: pull text logits + hidden state from libllama,
//          run audio LM heads, run DelayState, embed the next row, decode
//       5) once DelayState reports stopping, extract audio codes
//
// Not wired up: continuation mode for the delay family. Its slot layout differs
// and it has no verified reference here, so GenerateRequest::ref_text is
// rejected for that architecture rather than silently ignored.

#include "openmoss/pipeline.h"
#include "openmoss/codec.h"
#include "openmoss/frame_decoder.h"
#include "openmoss/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"

namespace openmoss {

namespace {

using clock_t_  = std::chrono::steady_clock;
using seconds_t = std::chrono::duration<double>;

std::string default_or_none(const std::optional<std::string> & s) {
    return s ? *s : "None";
}

std::string default_or_none(const std::optional<int> & v) {
    return v ? std::to_string(*v) : "None";
}

// Convert a token id to its string form. Mirrors the Python helper.
std::string id_to_token(const Tokenizer & tok, int32_t id) {
    return tok.decode({id});
}

// Build a literal token-string form of the reference-audio block:
//   <audio_start><user_slot>…<delay_slot>…<audio_end>
// — exactly what `_replace_audio_placeholders` produces upstream when the
// reference is for a `user` role (`audio_user_slot` for *both* the gen and
// delay positions). The block has length `T_ref + n_vq + 1`.
std::string build_reference_audio_block(const Tokenizer & tok,
                                         const ModelDims & d,
                                         int32_t T_ref) {
    const std::string audio_start = id_to_token(tok, d.audio_start_token_id);
    const std::string audio_end   = id_to_token(tok, d.audio_end_token_id);
    const std::string user_slot   = id_to_token(tok, d.audio_user_slot_token_id);
    std::string s;
    s += audio_start;
    for (int t = 0; t < T_ref; ++t) s += user_slot;
    for (int i = 0; i < d.n_vq - 1; ++i) s += user_slot;
    s += audio_end;
    return s;
}

// Build the user-instruction body that wraps the synthesis target.
//   reference_block: literal token string for the encoded reference audio,
//                    or empty when no reference is supplied.
std::string build_user_inst(const GenerateRequest & req,
                             const std::string & reference_block) {
    std::string s;
    s += "<user_inst>\n";
    s += "- Reference(s):\n";
    if (reference_block.empty()) s += "None\n";
    else                          s += "[S1]:\n" + reference_block + "\n";
    s += "- Instruction:\n" + default_or_none(req.instruction) + "\n";
    s += "- Tokens:\n"      + default_or_none(req.tokens)      + "\n";
    s += "- Quality:\n"     + default_or_none(req.quality)     + "\n";
    s += "- Sound Event:\nNone\n";
    s += "- Ambient Sound:\nNone\n";
    s += "- Language:\n"    + default_or_none(req.language)    + "\n";
    s += "- Text:\n"        + req.text                         + "\n";
    s += "</user_inst>";
    return s;
}

// Build the full assistant-prompt string:
//   <im_start>user\n…<im_end>\n<im_start>assistant\n<audio_start>
std::string build_prompt_text(const Tokenizer & tok, const ModelDims & d,
                               const GenerateRequest & req,
                               const std::string & reference_block) {
    const std::string im_start    = id_to_token(tok, d.im_start_token_id);
    const std::string im_end      = id_to_token(tok, d.im_end_token_id);
    const std::string audio_start = id_to_token(tok, d.audio_start_token_id);

    std::string body = build_user_inst(req, reference_block);
    std::string out;
    out += im_start + "user\n" + body + im_end + "\n"
         + im_start + "assistant\n" + audio_start;
    return out;
}

// Apply the delay-pattern shift to a (n_vq, T_ref) row-major code matrix.
//   Output: (T_ref + n_vq - 1, n_vq) row-major, where row r col i =
//     codes[i, r - i]   if 0 <= r - i < T_ref
//     pad_code          otherwise
// — equivalent to MossTTSDelayProcessor.apply_delay_pattern.
std::vector<int32_t> apply_delay_pattern(const int32_t * codes,
                                          int32_t n_vq, int32_t T_ref,
                                          int32_t pad_code) {
    const int64_t T = int64_t(T_ref) + int64_t(n_vq) - 1;
    std::vector<int32_t> out(size_t(T) * size_t(n_vq), pad_code);
    for (int32_t i = 0; i < n_vq; ++i) {
        for (int32_t t = 0; t < T_ref; ++t) {
            out[size_t(i + t) * size_t(n_vq) + size_t(i)] = codes[i * T_ref + t];
        }
    }
    return out;
}

// Build the (S, 1+n_vq) prompt grid: text ids in column 0, audio_pad_code
// elsewhere — except for the reference-audio rows where we splice in the
// delay-pattern-shifted codes from the encoded reference.
std::vector<int32_t> build_prompt_grid(const Tokenizer & tok,
                                       const ModelDims & d,
                                       const GenerateRequest & req,
                                       const std::string & reference_block,
                                       const std::vector<int32_t> * ref_codes,
                                       int32_t T_ref,
                                       int32_t & n_pos_out) {
    const std::string prompt = build_prompt_text(tok, d, req, reference_block);
    auto ids = tok.encode(prompt, /*add_special=*/false);
    n_pos_out = int32_t(ids.size());
    const int32_t cols = 1 + d.n_vq;

    std::vector<int32_t> grid(size_t(n_pos_out) * size_t(cols), d.audio_pad_code);
    for (int32_t r = 0; r < n_pos_out; ++r) {
        grid[size_t(r) * size_t(cols) + 0] = ids[r];
    }

    if (!ref_codes || T_ref <= 0) return grid;

    // Locate the (single) audio_start / audio_end pair that bounds the user
    // reference. The trailing audio_start the assistant turn ends with does
    // NOT have a matching audio_end and is therefore skipped naturally.
    int32_t a_start = -1, a_end = -1;
    for (int32_t r = 0; r < n_pos_out; ++r) {
        if (a_start < 0 && ids[r] == d.audio_start_token_id) {
            a_start = r;
        } else if (a_start >= 0 && ids[r] == d.audio_end_token_id) {
            a_end = r;
            break;
        }
    }
    if (a_start < 0 || a_end < 0) {
        throw std::runtime_error("build_prompt_grid: reference audio markers not found in tokenized prompt");
    }

    const int32_t span = a_end - a_start - 1;        // tokens strictly between markers
    const int32_t expected = T_ref + d.n_vq - 1;
    if (span != expected) {
        throw std::runtime_error("build_prompt_grid: reference audio span mismatch (got " +
                                  std::to_string(span) + ", expected " + std::to_string(expected) + ")");
    }

    const auto delayed = apply_delay_pattern(ref_codes->data(), d.n_vq, T_ref, d.audio_pad_code);
    for (int32_t k = 0; k < span; ++k) {
        const int32_t r = a_start + 1 + k;
        for (int32_t i = 0; i < d.n_vq; ++i) {
            grid[size_t(r) * size_t(cols) + 1 + i] =
                delayed[size_t(k) * size_t(d.n_vq) + size_t(i)];
        }
    }
    return grid;
}

// ───────────────────────────────────────────────────────────────────────────
// MOSS-TTS-Local prompt builder
//
// Two things differ from the delay family beyond the template text:
//
//  1. Reference audio codes go in 1:1 and unshifted — one grid row per codec
//     frame, text column = audio_user_slot — with no delay ramp, no trailing
//     (n_vq - 1) slot padding, and NO "[S1]:" speaker label. Upstream drops the
//     label deliberately: it changes the token sequence and degrades voice-clone
//     conditioning.
//
//  2. The prompt is tokenized **segment by segment**, never as one string.
//     One-pass encoding produces different ids: BPE merges the synthesis text's
//     trailing "." with the following "\n" into a single token (id 624) where
//     the segmented encoder emits 13 ("." ) + 198 ("\n"). Several other seams
//     behave the same way ('。', '!', '  ', and ':\n\n' when the text starts
//     with a newline). Getting this wrong shifts every downstream position.
// ───────────────────────────────────────────────────────────────────────────

void append_text_rows(std::vector<int32_t> & grid, int32_t cols, int32_t pad,
                      const std::vector<int32_t> & ids) {
    for (int32_t id : ids) {
        grid.push_back(id);
        for (int32_t c = 1; c < cols; ++c) grid.push_back(pad);
    }
}

std::vector<int32_t> build_prompt_grid_local(const Tokenizer & tok,
                                             const ModelDims & d,
                                             const GenerateRequest & req,
                                             const std::vector<int32_t> * ref_codes,
                                             int32_t T_ref,
                                             int32_t & n_pos_out) {
    const int32_t cols = 1 + d.n_vq;
    const int32_t pad  = d.audio_pad_code;

    auto enc = [&](const std::string & s) { return tok.encode(s, /*add_special=*/false); };

    std::vector<int32_t> grid;

    // <|im_start|>user\n<user_inst>\n- Reference(s):\n
    append_text_rows(grid, cols, pad, { d.im_start_token_id });
    append_text_rows(grid, cols, pad, enc("user\n"));
    append_text_rows(grid, cols, pad, enc("<user_inst>\n- Reference(s):\n"));

    // Continuation mode puts nothing here — it takes the same "None" the
    // no-reference path takes, and the reference audio goes after the trailing
    // audio_start instead. See the tail of this function.
    const bool continuation = req.ref_text.has_value() && ref_codes && T_ref > 0;

    if (ref_codes && T_ref > 0 && !continuation) {
        append_text_rows(grid, cols, pad, { d.audio_start_token_id });
        for (int32_t t = 0; t < T_ref; ++t) {
            grid.push_back(d.audio_user_slot_token_id);
            for (int32_t i = 0; i < d.n_vq; ++i) {
                grid.push_back((*ref_codes)[size_t(i) * size_t(T_ref) + size_t(t)]);
            }
        }
        append_text_rows(grid, cols, pad, { d.audio_end_token_id });
    } else {
        append_text_rows(grid, cols, pad, enc("None"));
    }

    // Everything between the reference block and the synthesis text is one
    // segment upstream, so encode it as one.
    {
        std::string mid;
        mid += "\n- Instruction:\n"   + default_or_none(req.instruction);
        mid += "\n- Tokens:\n"        + default_or_none(req.tokens);
        mid += "\n- Quality:\n"       + default_or_none(req.quality);
        mid += "\n- Sound Event:\nNone";
        mid += "\n- Ambient Sound:\nNone";
        mid += "\n- Language:\n"      + default_or_none(req.language);
        mid += "\n- Text:\n";
        append_text_rows(grid, cols, pad, enc(mid));
    }

    // The synthesis text is its own segment — see (2) above.
    //
    // In continuation mode the model has already "said" the reference audio, so
    // the text it is working from is the reference's transcript followed by the
    // new material. Upstream takes a single combined string; a space is inserted
    // only when neither side already provides a boundary, since scripts without
    // word spacing must not gain one.
    {
        std::string text = req.text;
        if (continuation) {
            const std::string & rt = *req.ref_text;
            std::string joined = rt;
            const bool have_boundary =
                (!rt.empty()  && std::isspace(static_cast<unsigned char>(rt.back()))) ||
                (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())));
            if (!rt.empty() && !text.empty() && !have_boundary) joined += ' ';
            joined += text;
            text = joined;
        }
        append_text_rows(grid, cols, pad, enc(text));
    }

    // \n</user_inst><|im_end|>\n<|im_start|>assistant\n<|audio_start|>
    append_text_rows(grid, cols, pad, enc("\n</user_inst>"));
    append_text_rows(grid, cols, pad, { d.im_end_token_id });
    append_text_rows(grid, cols, pad, enc("\n"));
    append_text_rows(grid, cols, pad, { d.im_start_token_id });
    append_text_rows(grid, cols, pad, enc("assistant\n"));
    append_text_rows(grid, cols, pad, { d.audio_start_token_id });

    // Continuation mode: the reference codes sit here, in the *assistant*
    // channel, and there is deliberately no audio_end — generation picks up
    // straight from the last reference frame.
    if (continuation) {
        for (int32_t t = 0; t < T_ref; ++t) {
            grid.push_back(d.audio_assistant_gen_slot_token_id);
            for (int32_t i = 0; i < d.n_vq; ++i) {
                grid.push_back((*ref_codes)[size_t(i) * size_t(T_ref) + size_t(t)]);
            }
        }
    }

    n_pos_out = int32_t(grid.size() / size_t(cols));
    return grid;
}

// Feed an (n_tokens, hidden) f32 buffer into libllama via batch.embd.
// `pos_start` is the position id for the first row.
// `output_last` controls whether the last row should produce logits.
//
// Split across n_batch-sized pieces. This is not an optimisation: libllama
// asserts `n_tokens_all <= cparams.n_batch` and a failed GGML_ASSERT calls
// abort(), so oversizing a single batch does not return an error code — it
// takes the whole process down. A voice-clone reference is caller-supplied and
// contributes one prompt token per codec frame (12.5 per second of audio), so
// without this a long enough reference_wav_b64 is a remote kill switch for the
// server. Found via a 324 s reference: 4113 prompt tokens against the default
// n_batch of 512.
//
// Splitting does change the result for a prompt that spans several batches —
// different matmul shapes, different summation order, and sampling diverges
// from there. That is not a regression: those prompts previously aborted the
// process. A prompt that fits in one batch stays byte-identical, so reference
// comparisons are unaffected.
void llama_decode_embeddings(llama_context * ctx,
                             const float *   embds,
                             int32_t         n_tokens,
                             int32_t         hidden,
                             int32_t         pos_start,
                             bool            output_last) {
    if (n_tokens <= 0) return;

    const int32_t n_batch = int32_t(llama_n_batch(ctx));
    if (n_batch <= 0) {
        throw std::runtime_error("llama_decode_embeddings: llama_n_batch returned "
                                 + std::to_string(n_batch));
    }

    for (int32_t off = 0; off < n_tokens; off += n_batch) {
        const int32_t n    = std::min(n_batch, n_tokens - off);
        const bool    last = (off + n == n_tokens);

        llama_batch batch = llama_batch_init(n, hidden, /*n_seq_max=*/1);
        batch.n_tokens = n;
        std::memcpy(batch.embd,
                    embds + size_t(off) * size_t(hidden),
                    size_t(n) * size_t(hidden) * sizeof(float));
        for (int32_t i = 0; i < n; ++i) {
            batch.pos[i]       = pos_start + off + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            // Only the very last row of the whole prompt produces logits.
            batch.logits[i]    = (output_last && last && i == n - 1) ? 1 : 0;
        }
        int32_t rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error("llama_decode_embeddings: llama_decode returned "
                                     + std::to_string(rc) + " for tokens ["
                                     + std::to_string(off) + ", "
                                     + std::to_string(off + n) + ")");
        }
    }
}

} // namespace

std::vector<int32_t> debug_build_prompt_grid(Model & model,
                                             const GenerateRequest & req,
                                             const std::vector<int32_t> * ref_codes,
                                             int32_t  T_ref,
                                             int32_t & n_pos_out) {
    const auto & d = model.dims();
    Tokenizer * tok = model.tokenizer();
    switch (d.arch) {
        case Arch::TTSLocal:
            return build_prompt_grid_local(*tok, d, req, ref_codes, T_ref, n_pos_out);
        case Arch::TTSDelay: {
            std::string block;
            if (ref_codes && T_ref > 0) block = build_reference_audio_block(*tok, d, T_ref);
            return build_prompt_grid(*tok, d, req, block, ref_codes, T_ref, n_pos_out);
        }
        case Arch::SoundEffect:
            break;
    }
    throw std::runtime_error("debug_build_prompt_grid: this architecture has no prompt grid");
}

GenerateResult generate(Model & model,
                        const GenerateRequest & req,
                        StreamCallback cb) {
    const auto & d = model.dims();
    const int32_t n_vq    = d.n_vq;
    const int32_t hidden  = d.hidden_size;
    const int32_t Vfull   = d.audio_vocab_size + 1;
    Tokenizer * tok       = model.tokenizer();

    GenerateResult result;
    auto t_total = clock_t_::now();

    // ── 0. Encode reference audio (voice cloning) ──────────────────────────
    std::vector<int32_t> ref_codes;
    int32_t T_ref = 0;
    std::string reference_block;
    if (req.reference_wav && !req.reference_wav->empty()) {
        if (!model.codec_loaded())
            throw std::runtime_error("generate: reference_wav supplied but codec is not loaded");

        auto t_enc = clock_t_::now();
        int32_t nvq_enc = 0;
        ref_codes = codec_encode(model,
                                  req.reference_wav->data(),
                                  int64_t(req.reference_wav->size()),
                                  nvq_enc, T_ref);
        if (nvq_enc != n_vq) {
            throw std::runtime_error("generate: codec_encode returned n_vq=" +
                                      std::to_string(nvq_enc) + ", expected " + std::to_string(n_vq));
        }
        std::fprintf(stderr, "[generate] encoded reference: %d frames (%.2fs) in %.2fs\n",
                     T_ref,
                     T_ref * double(d.downsample_rate) / double(d.sampling_rate),
                     seconds_t(clock_t_::now() - t_enc).count());
        if (d.arch == Arch::TTSDelay) {
            reference_block = build_reference_audio_block(*tok, d, T_ref);
        }
    }

    // ── 1. Prompt grid ─────────────────────────────────────────────────────
    int32_t prompt_len = 0;
    std::vector<int32_t> grid;
    switch (d.arch) {
        case Arch::TTSDelay:
            if (req.ref_text) {
                // The delay family's continuation layout differs and is
                // unverified here; failing beats silently ignoring the field
                // and cloning by a different mechanism than the caller asked for.
                throw std::runtime_error(
                    "generate(): ref_text (continuation mode) is only implemented "
                    "for moss_tts_local");
            }
            grid = build_prompt_grid(*tok, d, req, reference_block,
                                     T_ref > 0 ? &ref_codes : nullptr, T_ref,
                                     prompt_len);
            break;
        case Arch::TTSLocal:
            grid = build_prompt_grid_local(*tok, d, req,
                                           T_ref > 0 ? &ref_codes : nullptr, T_ref,
                                           prompt_len);
            break;
        case Arch::SoundEffect:
            // Not autoregressive: there is no prompt grid and no frame decoder.
            // It is sampled by a flow-matching solver instead — see
            // openmoss/soundeffect.h.
            throw std::runtime_error(
                "generate(): moss_soundeffect is not an autoregressive model; "
                "use the flow-matching entry point instead");
    }
    std::fprintf(stderr, "[generate] prompt_len = %d tokens\n", prompt_len);

    // Chunking the prefill removes the n_batch limit but not the context limit:
    // the KV cache is sized at load time. Say so here rather than let libllama
    // fail several seconds later with no indication of which input was at fault.
    // Reference audio is the usual cause — it costs one prompt token per codec
    // frame, 12.5 per second of audio, so a few minutes of it fills a default
    // 8192-token context on its own.
    {
        const int32_t n_ctx = int32_t(llama_n_ctx(model.backbone_ctx()));
        std::string   because;
        if (T_ref > 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          " The %.1fs reference accounts for %d of them.",
                          double(T_ref) * double(d.downsample_rate) / double(d.sampling_rate),
                          T_ref);
            because = buf;
        }
        if (prompt_len >= n_ctx) {
            throw std::runtime_error(
                "generate: the prompt is " + std::to_string(prompt_len) +
                " tokens but the context is " + std::to_string(n_ctx) + "." + because +
                " Shorten the reference audio or raise --n-ctx.");
        }
        if (prompt_len + req.max_new_tokens > n_ctx) {
            std::fprintf(stderr,
                "[generate] WARNING: prompt (%d) + max_new_tokens (%d) exceeds the context "
                "(%d); generation will be cut short at %d tokens.%s\n",
                prompt_len, req.max_new_tokens, n_ctx, n_ctx - prompt_len, because.c_str());
        }
    }

    // Seed the family's frame decoder from the prompt grid.
    std::vector<std::vector<int32_t>> history;
    history.reserve(size_t(prompt_len) + size_t(req.max_new_tokens));
    for (int32_t r = 0; r < prompt_len; ++r) {
        std::vector<int32_t> row(grid.begin() + r * (1 + n_vq),
                                  grid.begin() + (r + 1) * (1 + n_vq));
        history.push_back(std::move(row));
    }
    std::unique_ptr<IFrameDecoder> decoder = make_frame_decoder(model, history);

    // ── 1b. Streaming decode ───────────────────────────────────────────────
    //
    // With a callback, the codec runs incrementally alongside generation
    // instead of once at the end. `result.waveform` is still assembled either
    // way — a streaming caller usually wants the whole thing too — so the only
    // thing that changes is *when* the samples exist.
    //
    // Both families surface new frames the same way: extract_audio_codes() is
    // prefix-stable, so whatever it reports beyond what has already been sent is
    // exactly the new material. That is what makes this work for the delay
    // pattern, where a frame is not finished until its last codebook has been
    // sampled n_vq steps later — polling the extractor gets that right without
    // restating the un-shifting rule here.
    std::unique_ptr<CodecStreamDecoder> stream;
    int32_t streamed_frames = 0;
    double  stream_seconds  = 0.0;
    if (cb) {
        if (!model.codec_loaded()) {
            throw std::runtime_error(
                "generate: a StreamCallback was supplied but the codec is not loaded "
                "(skip_codec, or the GGUF was converted without --codec) — there is "
                "nothing to stream");
        }
        stream = std::make_unique<CodecStreamDecoder>(model, n_vq);
    }
    const int32_t chunk_frames = std::max(1, req.stream_chunk_frames);
    std::vector<int32_t> stream_slice;

    // Emit whatever frames have become complete since the last call.
    auto drain = [&]() {
        if (!stream) return;
        int32_t nvq_now = 0, t_now = 0;
        const auto codes_now = decoder->extract_audio_codes(nvq_now, t_now);
        const int32_t n = t_now - streamed_frames;
        if (n <= 0) return;
        if (nvq_now != n_vq) {
            throw std::runtime_error("generate: frame decoder reports n_vq=" +
                                      std::to_string(nvq_now) + " mid-stream, expected " +
                                      std::to_string(n_vq));
        }
        stream_slice.resize(size_t(n_vq) * size_t(n));
        for (int32_t c = 0; c < n_vq; ++c) {
            for (int32_t t = 0; t < n; ++t) {
                stream_slice[size_t(c) * size_t(n) + size_t(t)] =
                    codes_now[size_t(c) * size_t(t_now) + size_t(streamed_frames + t)];
            }
        }
        const auto t_s = clock_t_::now();
        const auto pcm = stream->push(stream_slice.data(), n);
        stream_seconds += seconds_t(clock_t_::now() - t_s).count();
        streamed_frames = t_now;
        cb(pcm.data(), int64_t(pcm.size()));
        result.waveform.insert(result.waveform.end(), pcm.begin(), pcm.end());
    };

    // ── 2. Prefill ─────────────────────────────────────────────────────────
    auto t0 = clock_t_::now();
    auto prefill_embeds = model.compute_input_embeddings(grid.data(), prompt_len);
    {
        // Clear KV before prefill — pipeline assumes a fresh sequence.
        llama_memory_clear(llama_get_memory(model.backbone_ctx()), /*data=*/true);
        llama_decode_embeddings(model.backbone_ctx(),
                                prefill_embeds.data(),
                                prompt_len, hidden, /*pos_start=*/0, /*output_last=*/true);
    }
    result.prefill_seconds = seconds_t(clock_t_::now() - t0).count();
    std::fprintf(stderr, "[generate] prefill done in %.2fs\n", result.prefill_seconds);

    // ── 3. Autoregressive loop ─────────────────────────────────────────────
    auto t_gen = clock_t_::now();
    int32_t pos = prompt_len;
    int32_t step = 0;
    FrameStep last_step;
    for (; step < req.max_new_tokens; ++step) {
        const float * text_logits = llama_get_logits_ith(model.backbone_ctx(), -1);
        const float * hidden_vec  = llama_get_embeddings_ith(model.backbone_ctx(), -1);
        if (!text_logits || !hidden_vec) {
            throw std::runtime_error("generate: llama_get_logits/embeddings_ith returned null");
        }

        // The decoder owns head invocation — the two families disagree on
        // whether the audio heads can run before any code is sampled.
        last_step = decoder->step(text_logits, hidden_vec, req.sampling);
        if (last_step.stop) {
            std::fprintf(stderr, "[generate] stop at step %d\n", step);
            break;
        }

        // Embed the (1, 1+n_vq) row and feed it as the next position.
        auto next_emb = model.compute_input_embeddings(last_step.ids.data(), 1);
        llama_decode_embeddings(model.backbone_ctx(),
                                next_emb.data(),
                                /*n_tokens=*/1, hidden,
                                /*pos_start=*/pos,
                                /*output_last=*/true);
        ++pos;

        // Poll on generation steps rather than audio frames: the delay family
        // spends its first n_vq steps filling the ramp and produces no complete
        // frame at all, and drain() emits only what is actually ready anyway.
        if (stream && (step + 1) % chunk_frames == 0) drain();
    }
    if (stream) drain();   // flush the tail
    result.generate_seconds = seconds_t(clock_t_::now() - t_gen).count();
    std::fprintf(stderr, "[generate] generated %d steps in %.2fs (%.1f tok/s)\n",
                 step, result.generate_seconds, step / std::max(result.generate_seconds, 1e-6));
    if (step >= req.max_new_tokens && !last_step.stop) {
        std::fprintf(stderr,
            "[generate] WARNING: hit max_new_tokens (%d) without an end-of-speech token; "
            "audio may be truncated and codec decode may be rejected as too long\n",
            req.max_new_tokens);
    }

    // ── 4. Extract audio codes ─────────────────────────────────────────────
    int32_t nvq_out = 0, t_audio = 0;
    auto codes = decoder->extract_audio_codes(nvq_out, t_audio);
    result.n_audio_frames = t_audio;
    result.n_codebooks    = nvq_out;
    result.audio_codes    = codes;
    std::fprintf(stderr, "[generate] %d audio frames extracted\n", t_audio);

    // ── 5. Codec decode → waveform ─────────────────────────────────────────
    result.n_channels = d.n_channels;
    if (stream) {
        // Already decoded, chunk by chunk, during the loop above.
        result.decode_seconds = stream_seconds;
        if (streamed_frames != t_audio) {
            std::fprintf(stderr,
                         "[generate] WARNING: streamed %d frames but the decoder reports %d; "
                         "the tail may be missing from the stream\n",
                         streamed_frames, t_audio);
        }
        std::fprintf(stderr,
                     "[generate] streamed %d frames (%.2fs of audio, %d ch) in %.2fs "
                     "of codec time, %d frames per chunk\n",
                     streamed_frames,
                     double(result.waveform.size())
                         / double(d.sampling_rate) / double(d.n_channels),
                     d.n_channels, result.decode_seconds, chunk_frames);
    } else if (t_audio > 0 && model.codec_loaded()) {
        try {
            auto t_dec = clock_t_::now();
            result.waveform = codec_decode(model, codes.data(), nvq_out, t_audio);
            result.decode_seconds = seconds_t(clock_t_::now() - t_dec).count();
            std::fprintf(stderr,
                         "[generate] codec decode produced %zu samples (%.2fs of audio, %d ch) in %.2fs\n",
                         result.waveform.size(),
                         double(result.waveform.size())
                             / double(d.sampling_rate) / double(d.n_channels),
                         d.n_channels, result.decode_seconds);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[generate] codec_decode failed: %s\n", e.what());
            result.decode_seconds = 0.0;
        }
    } else {
        if (!model.codec_loaded()) {
            std::fprintf(stderr, "[generate] codec not loaded (skip_codec or --codec wasn't used during conversion); emitting empty waveform\n");
        }
        result.decode_seconds = 0.0;
    }

    (void)Vfull;
    (void)t_total;
    return result;
}

} // namespace openmoss
