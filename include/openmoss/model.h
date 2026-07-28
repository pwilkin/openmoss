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

// Which MOSS model family a GGUF holds, recorded as the `moss.architecture`
// string KV. GGUFs produced before multi-family support carry no such key and
// are all MOSS-TTS-Delay, so that stays the default.
enum class Arch {
    TTSDelay,     // "moss_tts_delay" — 32 codebooks, delay pattern, 24 kHz mono
    TTSLocal,     // "moss_tts_local" — 12 codebooks, local transformer, 48 kHz stereo
    SoundEffect,  // "moss_soundeffect" — not autoregressive: a Wan-style DiT sampled
                  // with flow matching, decoded by a continuous DAC VAE. The
                  // "backbone" is a plain Qwen3-1.7B used only as a text encoder.
};

// Out-of-line deleters for the MOSS-SoundEffect graphs Model caches below.
// Model's destructor is compiled where those types are incomplete — they live in
// soundeffect.h, which includes *this* header — so the deleters are declared
// here and defined next to each class. Same reason DiTContext has one.
struct DiTGraphDeleter   { void operator()(class DiTGraph *)   const; };
struct DacDecoderDeleter { void operator()(class DacDecoder *) const; };

const char * arch_name(Arch a);
bool         arch_from_name(const std::string & name, Arch & out);

// Parameters of the loaded model. Defaults describe MOSS-TTS-Delay; everything
// here is overridden from GGUF metadata at load time (see read_moss_kv).
struct ModelDims {
    Arch    arch              = Arch::TTSDelay;
    int32_t hidden_size       = 4096;   // Qwen3-8B
    int32_t n_vq              = 32;     // audio codebooks
    int32_t audio_vocab_size  = 1024;   // codebook entries (1025 with pad)
    int32_t audio_pad_code    = 1024;
    int32_t sampling_rate     = 24000;
    int32_t downsample_rate   = 1920;   // codec hop size, per channel
    int32_t n_channels        = 1;      // output channels; 2 = stereo, interleaved
    int32_t text_vocab_size   = 155648; // Qwen3 backbone vocab; set from token_embd at load

    // Which MOSS-Audio-Tokenizer generation the codec tensors came from.
    //   1 = 24 kHz mono, 4 stages/side, hop 1920
    //   2 = 48 kHz stereo, 6 stages/side, hop 3840/channel
    int32_t codec_version     = 1;

    // Local ("depth") transformer geometry — Arch::TTSLocal only, 0 when unused.
    // A 1-layer GPT-2-style block that expands each frame latent into n_vq codes.
    int32_t local_n_layer     = 0;
    int32_t local_n_embd      = 0;
    int32_t local_n_head      = 0;
    int32_t local_n_inner     = 0;
    float   local_rope_base   = 1000000.0f;
    float   local_ln_eps      = 1e-6f;

    // ── Arch::SoundEffect only; all zero/default when unused ──────────────
    //
    // The DiT (WanAudioModel). `dit_in_dim` == `dit_out_dim` == the VAE latent
    // width, since the model predicts a velocity in latent space.
    int32_t dit_dim           = 0;      // 1536
    int32_t dit_n_layers      = 0;      // 30
    int32_t dit_n_heads       = 0;      // 12  → head_dim 128
    int32_t dit_ffn_dim       = 0;      // 8960
    int32_t dit_in_dim        = 0;      // 128
    int32_t dit_out_dim       = 0;      // 128
    int32_t dit_text_dim      = 0;      // 2048 = the Qwen3-1.7B hidden size
    int32_t dit_freq_dim      = 0;      // 256, width of the sinusoidal time code
    float   dit_eps           = 1e-6f;  // shared by every LayerNorm and RMSNorm
    float   dit_rope_base     = 10000.0f;

    // Flow-match scheduler.
    float   sched_shift       = 5.0f;
    int32_t sched_train_steps = 1000;

    // DAC VAE decoder. `vae_hop` is the product of `vae_decoder_rates`.
    int32_t vae_latent_dim    = 0;      // 128
    int32_t vae_hop           = 0;      // 960 → 50 latent frames/s at 48 kHz
    std::vector<int32_t> vae_decoder_rates;   // {8, 5, 4, 3, 2}

    // Text conditioning is a fixed-width window: the prompt is right-padded to
    // `text_max_len` slots and the rows past the real length are zeroed.
    int32_t text_max_len      = 0;      // 512
    // The DiT always denoises this many seconds of latent regardless of the
    // requested duration; the waveform is cropped afterwards.
    int32_t max_seconds       = 0;      // 30

    // Special token ids in the Qwen3 vocab — populated from GGUF metadata.
    int32_t pad_token_id                  = 151643;
    int32_t im_start_token_id             = 151644;
    int32_t im_end_token_id               = 151645;
    int32_t audio_start_token_id          = 151652;
    int32_t audio_end_token_id            = 151653;
    int32_t audio_user_slot_token_id      = 151654;
    int32_t audio_assistant_gen_slot_token_id   = 151656;
    // Delay-pattern only; MOSS-TTS-Local has no delay slot.
    int32_t audio_assistant_delay_slot_token_id = 151662;

    // Samples emitted per codec frame across all channels.
    int64_t samples_per_frame() const {
        return int64_t(downsample_rate) * int64_t(n_channels);
    }
};

// Sampling knobs, shared by every family. Defaults are the MOSS-TTS-Delay
// values; per-family defaults come from default_sampling(Arch).
struct SamplingConfig {
    float text_temperature  = 1.5f;
    float text_top_p        = 1.0f;
    int   text_top_k        = 50;

    float audio_temperature       = 1.7f;
    float audio_top_p             = 0.8f;
    int   audio_top_k             = 25;
    float audio_repetition_penalty = 1.0f;

    // Forbid end-of-speech until at least this many audio frames have been
    // generated. Prevents the model from collapsing the segment on the first
    // frame (degenerate immediate end-of-speech). 0 = no floor.
    int min_audio_frames = 0;

    // Force end-of-speech once this many audio frames exist. Bounds the segment
    // so the model can't ramble far past the requested length. 0 = no cap.
    int max_audio_frames = 0;

    uint64_t seed = 0; // 0 = nondeterministic
};

// Sampling defaults for a family, as documented by the upstream cookbooks.
SamplingConfig default_sampling(Arch arch);

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

    // Force the aux backend (audio embeds + codec graphs) onto CPU even when
    // a GPU is available. Workaround for backends that don't implement every
    // op the codec needs (e.g. llama.cpp Metal lacks DIAG_MASK_INF). The
    // backbone still uses the GPU via libllama. Default: follow main_gpu.
    bool    aux_cpu      = false;
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

    // MOSS-SoundEffect graphs, lazy-instantiated and then cached for the
    // model's lifetime. Caching matters more here than it looks: constructing a
    // DacDecoder materialises ~300 MB of f32 convolution kernels on the backend,
    // which a server would otherwise redo on every request.
    class DiTGraph   * dit();
    class DacDecoder * dac_decoder();

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
    std::unique_ptr<class DiTGraph,   DiTGraphDeleter>   m_dit;
    std::unique_ptr<class DacDecoder, DacDecoderDeleter> m_dac;
};

} // namespace openmoss
