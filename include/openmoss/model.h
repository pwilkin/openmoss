// SPDX-License-Identifier: Apache-2.0
//
// Single-GGUF MOSS-TTS model handle: backbone (via libllama), audio extension
// embeddings + heads (raw GGML tensors), and the codec sub-graph.
//
// This header exposes only the load/free surface; the heavy lifting lives in
// the .cpp files and in pipeline.h / delay.h / codec.h.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_tensor;

namespace openmoss {

// Compile-time-fixed parameters of the MOSS-TTS-Delay design.
// (These match config.json on the upstream model card.)
struct ModelDims {
    int32_t hidden_size       = 4096;   // Qwen3-8B
    int32_t n_vq              = 32;     // audio codebooks
    int32_t audio_vocab_size  = 1024;   // codebook entries (1025 with pad)
    int32_t audio_pad_code    = 1024;
    int32_t sampling_rate     = 24000;
    int32_t downsample_rate   = 1920;   // codec hop size

    // Special token ids in the Qwen3 vocab — populated from GGUF metadata.
    int32_t pad_token_id                  = 151643;
    int32_t im_start_token_id             = 151644;
    int32_t im_end_token_id               = 151645;
    int32_t audio_start_token_id          = 151652;
    int32_t audio_end_token_id            = 151653;
    int32_t audio_user_slot_token_id      = 151654;
    int32_t audio_assistant_gen_slot_token_id   = 151656;
    int32_t audio_assistant_delay_slot_token_id = 151662;
};

struct LoadOptions {
    int32_t n_ctx        = 8192;
    int32_t n_batch      = 512;
    int32_t n_threads    = 0;        // 0 → libllama default
    int32_t n_gpu_layers = -1;       // -1 → all on GPU
    int32_t main_gpu     = -1;       // -1 → auto-pick the GPU with most free VRAM.
                                     // Index is into the GPU-type ggml device list
                                     // (same convention as libllama's `main_gpu`).
                                     // The backbone is pinned to this device via
                                     // LLAMA_SPLIT_MODE_NONE and the aux backend
                                     // (codec + embed/heads) is allocated on it too.
    bool    flash_attn   = true;

    // Skip loading the codec subgraph weights into the aux backend. Saves
    // ~3.4 GB of VRAM. Set to true when you only need the LM-side outputs
    // (audio codes, no waveform). Default: load codec when present.
    bool    skip_codec   = false;
};

// Forward decl; implemented in model.cpp.
class Model {
public:
    static std::unique_ptr<Model> load(const std::string & gguf_path, const LoadOptions & opts);
    ~Model();

    Model(const Model &)             = delete;
    Model & operator=(const Model &) = delete;

    const ModelDims & dims() const { return m_dims; }

    // Backbone access — used by the pipeline to feed embeddings + read hidden states.
    llama_model   * backbone_model() const { return m_backbone_model; }
    llama_context * backbone_ctx()   const { return m_backbone_ctx;   }

    // Audio extension tensors. Indexing convention: i in [0, n_vq).
    ggml_tensor * audio_embed(int i) const;   // (audio_vocab_size+1, hidden_size)
    ggml_tensor * audio_head (int i) const;   // (audio_vocab_size+1, hidden_size)

    // Compute the per-position summed input embedding for a prompt grid.
    //   prompt_grid: (n_pos, 1 + n_vq) row-major int32 — column 0 is text id,
    //                columns 1..n_vq are audio codebook ids (or audio_pad_code).
    //   Returns float32 (n_pos, hidden) — ready to feed into llama_batch.embd.
    //
    // Uses GGML on the aux backend; result is copied back to host memory.
    std::vector<float> compute_input_embeddings(const int32_t * prompt_grid,
                                                int32_t         n_pos) const;

    // Project the backbone's last hidden state through all 32 audio LM heads.
    //   hidden: float32 (hidden_size,)
    //   Returns float32 (n_vq, audio_vocab_size + 1) row-major.
    std::vector<float> compute_audio_logits(const float * hidden) const;

    // Codec graph wrapper (lazy-instantiated on first use).
    class CodecGraphs * codec();

    // Tokenizer wrapper (BPE, exposed via libllama's vocab API).
    class Tokenizer * tokenizer() const { return m_tokenizer.get(); }

    // Test-only / introspection helpers.
    int32_t n_audio_embed_loaded() const;
    int32_t n_audio_head_loaded()  const;
    bool    codec_present()        const;   // metadata says GGUF carries a codec
    bool    codec_loaded()         const;   // codec tensors are actually in aux memory

    // Public so free helpers in model.cpp can populate it; consumers should
    // ignore this (use the typed accessors above).
    struct Aux;
    Aux * aux() const { return m_aux.get(); }

private:
    Model();
    ModelDims m_dims;
    llama_model   * m_backbone_model = nullptr;
    llama_context * m_backbone_ctx   = nullptr;

    std::unique_ptr<Aux>                m_aux;
    std::unique_ptr<class Tokenizer>    m_tokenizer;
    std::unique_ptr<class CodecGraphs>  m_codec;
};

} // namespace openmoss
