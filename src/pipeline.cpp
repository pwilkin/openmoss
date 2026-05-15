// SPDX-License-Identifier: Apache-2.0
//
// End-to-end pipeline glue.
//
// What's wired up here:
//   - A prompt builder that mirrors the Python reference (without the audio
//     reference / continuation paths — those will land with task #6).
//   - The autoregressive generation loop:
//       1) build prompt grid (S, 1+n_vq) of int32
//       2) compute summed input embeddings (S, hidden) on the aux GGML backend
//       3) prefill libllama with batch.embd
//       4) at each step: pull text logits + hidden state from libllama,
//          run audio LM heads, run DelayState, embed the next row, decode
//       5) once DelayState reports stopping, extract audio codes
//
// What's NOT wired up:
//   - Reference audio encoding (codec encoder is a future task)
//   - Codec decoding to a waveform (future task — for now we return an empty
//     waveform but expose `n_audio_frames` so callers can see the codes were
//     produced)

#include "openmoss/pipeline.h"
#include "openmoss/codec.h"
#include "openmoss/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
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

// Feed an (n_tokens, hidden) f32 buffer into libllama via batch.embd.
// `pos_start` is the position id for the first row.
// `output_last` controls whether the last row should produce logits.
void llama_decode_embeddings(llama_context * ctx,
                             const float *   embds,
                             int32_t         n_tokens,
                             int32_t         hidden,
                             int32_t         pos_start,
                             bool            output_last) {
    llama_batch batch = llama_batch_init(n_tokens, hidden, /*n_seq_max=*/1);
    batch.n_tokens = n_tokens;
    std::memcpy(batch.embd, embds, size_t(n_tokens) * size_t(hidden) * sizeof(float));
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.pos[i]       = pos_start + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (output_last && i == n_tokens - 1) ? 1 : 0;
    }
    int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode_embeddings: llama_decode returned "
                                 + std::to_string(rc));
    }
}

} // namespace

GenerateResult generate(Model & model,
                        const GenerateRequest & req,
                        StreamCallback cb) {
    (void)cb; // streaming will land alongside chunked codec decode

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
                     T_ref * 1920.0 / double(d.sampling_rate),
                     seconds_t(clock_t_::now() - t_enc).count());
        reference_block = build_reference_audio_block(*tok, d, T_ref);
    }

    // ── 1. Prompt grid ─────────────────────────────────────────────────────
    int32_t prompt_len = 0;
    auto grid = build_prompt_grid(*tok, d, req, reference_block,
                                   T_ref > 0 ? &ref_codes : nullptr, T_ref,
                                   prompt_len);
    std::fprintf(stderr, "[generate] prompt_len = %d tokens\n", prompt_len);

    // Initialise DelayState from the prompt grid.
    std::vector<std::vector<int32_t>> history;
    history.reserve(size_t(prompt_len) + size_t(req.max_new_tokens));
    for (int32_t r = 0; r < prompt_len; ++r) {
        std::vector<int32_t> row(grid.begin() + r * (1 + n_vq),
                                  grid.begin() + (r + 1) * (1 + n_vq));
        history.push_back(std::move(row));
    }
    DelayState state(d, history);

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
    DelayStep last_step;
    for (; step < req.max_new_tokens; ++step) {
        const float * text_logits = llama_get_logits_ith(model.backbone_ctx(), -1);
        const float * hidden_vec  = llama_get_embeddings_ith(model.backbone_ctx(), -1);
        if (!text_logits || !hidden_vec) {
            throw std::runtime_error("generate: llama_get_logits/embeddings_ith returned null");
        }

        auto audio_logits = model.compute_audio_logits(hidden_vec); // (n_vq, Vfull)

        last_step = state.step(text_logits, audio_logits.data(), req.sampling);
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
    }
    result.generate_seconds = seconds_t(clock_t_::now() - t_gen).count();
    std::fprintf(stderr, "[generate] generated %d steps in %.2fs (%.1f tok/s)\n",
                 step, result.generate_seconds, step / std::max(result.generate_seconds, 1e-6));

    // ── 4. Extract audio codes ─────────────────────────────────────────────
    int32_t nvq_out = 0, t_audio = 0;
    auto codes = state.extract_audio_codes(nvq_out, t_audio);
    result.n_audio_frames = t_audio;
    std::fprintf(stderr, "[generate] %d audio frames extracted\n", t_audio);

    // ── 5. Codec decode → waveform ─────────────────────────────────────────
    if (t_audio > 0 && model.codec_loaded()) {
        try {
            auto t_dec = clock_t_::now();
            result.waveform = codec_decode(model, codes.data(), nvq_out, t_audio);
            result.decode_seconds = seconds_t(clock_t_::now() - t_dec).count();
            std::fprintf(stderr, "[generate] codec decode produced %zu samples (%.2fs of audio) in %.2fs\n",
                         result.waveform.size(),
                         result.waveform.size() / double(d.sampling_rate),
                         result.decode_seconds);
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
