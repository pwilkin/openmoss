// SPDX-License-Identifier: Apache-2.0
//
// MOSS-TTS model loader.
//
// Single-GGUF layout we expect (produced by scripts/convert_hf_to_gguf.py):
//
//   1. Standard Qwen3 backbone tensors (token_embd, blk.*, output_norm, output)
//      — loaded by libllama via `llama_model_load_from_file`.
//   2. MOSS audio extension tensors with the "moss." prefix:
//        moss.audio_embed.{i}.weight     (audio_vocab_size+1, hidden_size)
//        moss.audio_head.{i}.weight      (audio_vocab_size+1, hidden_size)
//        moss.codec.<...>                (codec encoder/decoder/quantizer)
//      — these are unknown to libllama and ignored by it. We re-open the file
//        with `gguf_init_from_file` to enumerate them, then bind them to a
//        separate ggml_backend buffer that we own.
//   3. KV metadata under "moss.*" carrying n_vq, vocab sizes, special tokens,
//      sampling rate.
//
// The backend selection prefers a GPU device (CUDA/Vulkan/Metal) and falls
// back to CPU if no accelerator is registered.

#include "openmoss/model.h"
#include "openmoss/codec.h"
#include "openmoss/tokenizer.h"

#include "aux_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

namespace openmoss {

// CodecGraphs lives in codec.cpp now — declared in openmoss/codec.h.
// Aux lives in aux_internal.h, included above.

// ───────────────────────────────────────────────────────────────────────────
// Helpers
// ───────────────────────────────────────────────────────────────────────────

namespace {

// Pull a uint32 KV from the GGUF, falling back to a default if missing.
uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t fallback) {
    int64_t k = gguf_find_key(gctx, key);
    if (k < 0) return fallback;
    auto t = gguf_get_kv_type(gctx, k);
    switch (t) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gctx, k);
        case GGUF_TYPE_INT32:  return uint32_t(gguf_get_val_i32(gctx, k));
        case GGUF_TYPE_UINT64: return uint32_t(gguf_get_val_u64(gctx, k));
        case GGUF_TYPE_INT64:  return uint32_t(gguf_get_val_i64(gctx, k));
        default: return fallback;
    }
}

bool kv_bool(gguf_context * gctx, const char * key, bool fallback) {
    int64_t k = gguf_find_key(gctx, key);
    if (k < 0) return fallback;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(gctx, k);
}

// Enumerate registered ggml devices that report as GPU/IGPU/ACCEL. Order
// matches both ggml's registration order and libllama's `main_gpu` indexing,
// so we can pass the same index to `llama_model_params.main_gpu` and use the
// same device for the aux backend.
std::vector<ggml_backend_dev_t> list_gpu_devs() {
    std::vector<ggml_backend_dev_t> out;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU
         || t == GGML_BACKEND_DEVICE_TYPE_IGPU
         || t == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            out.push_back(d);
        }
    }
    return out;
}

// Pick a GPU device. If `hint >= 0`, use that index into the GPU list.
// Otherwise auto-pick the GPU with the most free VRAM at this moment.
// Returns nullptr (and gpu_index_out = -1) if no GPU is available.
ggml_backend_dev_t pick_gpu_device(int hint, int & gpu_index_out) {
    ggml_backend_load_all();
    auto devs = list_gpu_devs();
    gpu_index_out = -1;
    if (devs.empty()) return nullptr;

    if (hint >= 0) {
        if (size_t(hint) >= devs.size()) {
            throw std::runtime_error(
                "main_gpu=" + std::to_string(hint) + " is out of range (only "
                + std::to_string(devs.size()) + " GPU device(s) available)");
        }
        gpu_index_out = hint;
        return devs[size_t(hint)];
    }

    int    best_idx  = 0;
    size_t best_free = 0;
    for (size_t i = 0; i < devs.size(); ++i) {
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(devs[i], &free, &total);
        if (free > best_free) { best_free = free; best_idx = int(i); }
    }
    gpu_index_out = best_idx;
    return devs[size_t(best_idx)];
}

void read_moss_kv(gguf_context * gctx, ModelDims & d, bool & codec_present) {
    d.n_vq             = int32_t(kv_u32(gctx, "moss.n_vq", uint32_t(d.n_vq)));
    d.audio_vocab_size = int32_t(kv_u32(gctx, "moss.audio_vocab_size", uint32_t(d.audio_vocab_size)));
    d.audio_pad_code   = int32_t(kv_u32(gctx, "moss.audio_pad_code", uint32_t(d.audio_pad_code)));
    d.sampling_rate    = int32_t(kv_u32(gctx, "moss.sampling_rate", uint32_t(d.sampling_rate)));
    d.downsample_rate  = int32_t(kv_u32(gctx, "moss.downsample_rate", uint32_t(d.downsample_rate)));

    d.audio_start_token_id          = int32_t(kv_u32(gctx, "moss.token.audio_start", uint32_t(d.audio_start_token_id)));
    d.audio_end_token_id            = int32_t(kv_u32(gctx, "moss.token.audio_end",   uint32_t(d.audio_end_token_id)));
    d.audio_user_slot_token_id      = int32_t(kv_u32(gctx, "moss.token.audio_user_slot",   uint32_t(d.audio_user_slot_token_id)));
    d.audio_assistant_gen_slot_token_id   = int32_t(kv_u32(gctx, "moss.token.audio_gen_slot",   uint32_t(d.audio_assistant_gen_slot_token_id)));
    d.audio_assistant_delay_slot_token_id = int32_t(kv_u32(gctx, "moss.token.audio_delay_slot", uint32_t(d.audio_assistant_delay_slot_token_id)));
    d.im_start_token_id             = int32_t(kv_u32(gctx, "moss.token.im_start", uint32_t(d.im_start_token_id)));
    d.im_end_token_id               = int32_t(kv_u32(gctx, "moss.token.im_end",   uint32_t(d.im_end_token_id)));
    d.pad_token_id                  = int32_t(kv_u32(gctx, "moss.token.pad",      uint32_t(d.pad_token_id)));

    codec_present = kv_bool(gctx, "moss.codec.present", false);
}

// One source: a GGUF file plus the names we want to copy out of it (with
// optional aliasing). All filtering is name-based.
struct LoadSpec {
    std::string         path;
    gguf_context *      gctx     = nullptr;
    ggml_context *      meta_ctx = nullptr;
    // pairs of (source_tensor_name, alias_name); alias must be unique across
    // all specs so the aux ctx never has duplicates.
    std::vector<std::pair<std::string,std::string>> wanted;
};

// Phase 1: dup descriptors for every wanted tensor into aux.ctx.
//          Aux owns no memory yet — that comes after all specs are accumulated.
void stage_tensor_descriptors(LoadSpec & spec, Model::Aux & aux,
                              size_t & extra_overhead) {
    // (lazy creation handled by caller)
    for (auto & [src_name, alias] : spec.wanted) {
        ggml_tensor * src = nullptr;
        for (ggml_tensor * cur = ggml_get_first_tensor(spec.meta_ctx);
             cur; cur = ggml_get_next_tensor(spec.meta_ctx, cur))
        {
            if (src_name == cur->name) { src = cur; break; }
        }
        if (!src) {
            throw std::runtime_error("Model::load: tensor not found in " + spec.path
                                     + ": " + src_name);
        }
        ggml_tensor * dup = ggml_dup_tensor(aux.ctx, src);
        ggml_set_name(dup, alias.c_str());
        if (alias == "_text_embed") {
            aux.text_embed = dup;
        } else {
            aux.tensors.emplace(alias, dup);
        }
        extra_overhead += ggml_nbytes(dup);
    }
}

// Phase 2: read the bytes from disk and upload to the already-bound backend.
void upload_tensor_data(const LoadSpec & spec, Model::Aux & aux) {
    if (spec.wanted.empty()) return;
    std::ifstream f(spec.path, std::ios::binary);
    if (!f) throw std::runtime_error("Model::load: cannot open " + spec.path);

    const size_t base_offset = gguf_get_data_offset(spec.gctx);
    std::vector<uint8_t> staging;

    for (const auto & [src_name, alias] : spec.wanted) {
        int64_t idx = gguf_find_tensor(spec.gctx, src_name.c_str());
        if (idx < 0) {
            throw std::runtime_error("Model::load: tensor missing from gguf table: "
                                     + src_name);
        }
        const size_t off  = base_offset + gguf_get_tensor_offset(spec.gctx, idx);
        ggml_tensor * dst = (alias == "_text_embed") ? aux.text_embed
                                                     : aux.tensors.at(alias);
        const size_t nbytes = ggml_nbytes(dst);
        if (staging.size() < nbytes) staging.resize(nbytes);
        f.seekg(std::streamoff(off));
        f.read(reinterpret_cast<char *>(staging.data()), std::streamsize(nbytes));
        if (size_t(f.gcount()) != nbytes) {
            throw std::runtime_error("Model::load: short read on tensor " + src_name);
        }
        ggml_backend_tensor_set(dst, staging.data(), 0, nbytes);
    }
}

// Walk a metadata ctx and pick the moss-prefixed tensor names.
//   include_codec=false skips the moss.codec.* tensors (saves ~3.4 GB VRAM
//   when the caller only needs the LM-side path).
void collect_moss_names(LoadSpec & spec, bool include_codec) {
    for (ggml_tensor * cur = ggml_get_first_tensor(spec.meta_ctx);
         cur; cur = ggml_get_next_tensor(spec.meta_ctx, cur))
    {
        std::string n = cur->name;
        if (n.rfind("moss.", 0) != 0) continue;
        if (!include_codec && n.rfind("moss.codec.", 0) == 0) continue;
        spec.wanted.emplace_back(n, n);
    }
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// Model
// ───────────────────────────────────────────────────────────────────────────

// Model::Model(), Model::~Model() and Model::codec() are defined in codec.cpp,
// where the CodecGraphs and Tokenizer types are both complete (model.h only
// forward-declares them, but unique_ptr<...> destructors need the full types).

ggml_tensor * Model::audio_embed(int i) const {
    if (!m_aux) return nullptr;
    auto it = m_aux->tensors.find("moss.audio_embed." + std::to_string(i) + ".weight");
    return it == m_aux->tensors.end() ? nullptr : it->second;
}

ggml_tensor * Model::audio_head(int i) const {
    if (!m_aux) return nullptr;
    auto it = m_aux->tensors.find("moss.audio_head." + std::to_string(i) + ".weight");
    return it == m_aux->tensors.end() ? nullptr : it->second;
}

int32_t Model::n_audio_embed_loaded() const {
    if (!m_aux) return 0;
    int32_t n = 0;
    for (auto & kv : m_aux->tensors)
        if (kv.first.rfind("moss.audio_embed.", 0) == 0) ++n;
    return n;
}

int32_t Model::n_audio_head_loaded() const {
    if (!m_aux) return 0;
    int32_t n = 0;
    for (auto & kv : m_aux->tensors)
        if (kv.first.rfind("moss.audio_head.", 0) == 0) ++n;
    return n;
}

bool Model::codec_present() const {
    return m_aux && m_aux->codec_present;
}

bool Model::codec_loaded() const {
    if (!m_aux) return false;
    return m_aux->tensors.count("moss.codec.quantizer.q.0.codebook.weight") > 0;
}

std::unique_ptr<Model> Model::load(const std::string & gguf_path, const LoadOptions & opts) {
    auto self = std::unique_ptr<Model>(new Model());

    // ── 0. Pick which GPU we want to live on (used for both libllama and the
    //       aux backend). On a multi-GPU box, leaving libllama in its default
    //       split mode would scatter the backbone across all visible CUDA
    //       devices, leaving the aux backend competing for whichever single
    //       GPU it lands on. Pinning everything to one device is simpler and
    //       avoids the cross-device aux-vs-backbone OOM pattern.
    int gpu_index = -1;
    ggml_backend_dev_t picked_dev = pick_gpu_device(opts.main_gpu, gpu_index);
    if (picked_dev) {
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(picked_dev, &free_b, &total_b);
        std::fprintf(stderr,
                     "Model::load: pinning to GPU %d (%s, %.0f/%.0f MiB free)\n",
                     gpu_index,
                     ggml_backend_dev_name(picked_dev),
                     double(free_b)  / (1024.0 * 1024.0),
                     double(total_b) / (1024.0 * 1024.0));
    } else {
        std::fprintf(stderr, "Model::load: no GPU device found; using CPU backend\n");
    }

    // ── 1. libllama loads the Qwen3 backbone ────────────────────────────────
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = opts.n_gpu_layers;
    if (picked_dev) {
        // Confine the backbone to the picked GPU. Without this, libllama's
        // default LLAMA_SPLIT_MODE_LAYER puts a slice on every visible GPU.
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu   = gpu_index;
    }
    self->m_backbone_model = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!self->m_backbone_model) {
        throw std::runtime_error("Model::load: libllama failed to load backbone from " + gguf_path);
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = uint32_t(opts.n_ctx);
    cp.n_batch         = uint32_t(opts.n_batch);
    cp.n_ubatch        = uint32_t(opts.n_batch);
    cp.n_threads       = opts.n_threads;
    cp.n_threads_batch = opts.n_threads;
    cp.embeddings      = true; // we feed and read embeddings directly
    cp.flash_attn_type = opts.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    self->m_backbone_ctx = llama_init_from_model(self->m_backbone_model, cp);
    if (!self->m_backbone_ctx) {
        llama_model_free(self->m_backbone_model);
        throw std::runtime_error("Model::load: failed to init backbone context");
    }

    // ── 2. Open the moss.* sidecar GGUF + the backbone GGUF for tensor data ─
    //
    // Convention: sidecar lives next to the backbone, same stem with the
    // ".extras.gguf" suffix. (E.g. "moss-tts.gguf" → "moss-tts.extras.gguf".)
    // The sidecar is written by scripts/convert_hf_to_gguf.py; it carries
    // tensors libllama would reject for being unknown. We *also* extract
    // `token_embd.weight` from the backbone GGUF — needed to compute summed
    // input embeddings before feeding them to libllama via batch.embd.
    self->m_aux = std::make_unique<Aux>();

    std::string sidecar_path;
    {
        const std::string p = gguf_path;
        const std::string ext = ".gguf";
        if (p.size() > ext.size() && p.compare(p.size() - ext.size(), ext.size(), ext) == 0) {
            sidecar_path = p.substr(0, p.size() - ext.size()) + ".extras.gguf";
        } else {
            sidecar_path = p + ".extras.gguf";
        }
    }

    LoadSpec sc_spec; sc_spec.path = sidecar_path;
    LoadSpec bb_spec; bb_spec.path = gguf_path;
    gguf_init_params gip{}; gip.no_alloc = true;

    gip.ctx = &sc_spec.meta_ctx;
    sc_spec.gctx = gguf_init_from_file(sidecar_path.c_str(), gip);

    gip.ctx = &bb_spec.meta_ctx;
    bb_spec.gctx = gguf_init_from_file(gguf_path.c_str(), gip);
    if (!bb_spec.gctx) {
        if (sc_spec.gctx) gguf_free(sc_spec.gctx);
        if (sc_spec.meta_ctx) ggml_free(sc_spec.meta_ctx);
        throw std::runtime_error("Model::load: gguf_init_from_file failed for backbone " + gguf_path);
    }

    if (!sc_spec.gctx) {
        std::fprintf(stderr,
            "Model::load: warning — sidecar %s not found; audio heads/embeds and codec are unavailable\n",
            sidecar_path.c_str());
    } else {
        read_moss_kv(sc_spec.gctx, self->m_dims, self->m_aux->codec_present);
        const bool include_codec = self->m_aux->codec_present && !opts.skip_codec;
        collect_moss_names(sc_spec, include_codec);
        if (self->m_aux->codec_present && opts.skip_codec) {
            std::fprintf(stderr, "Model::load: codec tensors skipped (LoadOptions::skip_codec=true)\n");
        }
    }
    bb_spec.wanted.emplace_back("token_embd.weight", "_text_embed");

    if (picked_dev) {
        self->m_aux->backend = ggml_backend_dev_init(picked_dev, nullptr);
    } else {
        // No GPU? Fall through to CPU backend so we still build something.
        self->m_aux->backend = ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }
    if (!self->m_aux->backend) {
        if (sc_spec.gctx) gguf_free(sc_spec.gctx);
        if (sc_spec.meta_ctx) ggml_free(sc_spec.meta_ctx);
        gguf_free(bb_spec.gctx);
        ggml_free(bb_spec.meta_ctx);
        throw std::runtime_error("Model::load: no GGML backend available");
    }
    std::fprintf(stderr, "Model::load: aux backend = %s\n",
                 ggml_backend_name(self->m_aux->backend));

    // Allocate the aux ctx to hold tensor descriptors from BOTH GGUFs.
    {
        size_t total_tensors = sc_spec.wanted.size() + bb_spec.wanted.size() + 16;
        ggml_init_params ip{};
        ip.mem_size   = ggml_tensor_overhead() * total_tensors;
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        self->m_aux->ctx = ggml_init(ip);
        if (!self->m_aux->ctx) {
            throw std::runtime_error("Model::load: ggml_init for aux ctx failed");
        }
    }

    try {
        size_t total_bytes = 0;
        if (sc_spec.gctx) stage_tensor_descriptors(sc_spec, *self->m_aux, total_bytes);
        stage_tensor_descriptors(bb_spec, *self->m_aux, total_bytes);

        self->m_aux->buffer = ggml_backend_alloc_ctx_tensors(self->m_aux->ctx, self->m_aux->backend);
        if (!self->m_aux->buffer) {
            throw std::runtime_error("Model::load: failed to allocate aux backend buffer ("
                                     + std::to_string(total_bytes / (1024 * 1024)) + " MiB)");
        }

        if (sc_spec.gctx) upload_tensor_data(sc_spec, *self->m_aux);
        upload_tensor_data(bb_spec, *self->m_aux);
    } catch (...) {
        if (sc_spec.gctx) gguf_free(sc_spec.gctx);
        if (sc_spec.meta_ctx) ggml_free(sc_spec.meta_ctx);
        gguf_free(bb_spec.gctx);
        ggml_free(bb_spec.meta_ctx);
        throw;
    }

    if (sc_spec.gctx) gguf_free(sc_spec.gctx);
    if (sc_spec.meta_ctx) ggml_free(sc_spec.meta_ctx);
    gguf_free(bb_spec.gctx);
    ggml_free(bb_spec.meta_ctx);

    // Cache convenient dim shortcuts on Aux.
    self->m_aux->hidden_size      = self->m_dims.hidden_size;
    self->m_aux->n_vq             = self->m_dims.n_vq;
    self->m_aux->audio_vocab_full = self->m_dims.audio_vocab_size + 1;
    self->m_aux->text_vocab_size  =
        self->m_aux->text_embed
        ? int32_t(self->m_aux->text_embed->ne[1])
        : 0;
    self->m_aux->galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(self->m_aux->backend));

    std::fprintf(stderr,
        "Model::load: loaded %d audio embeds, %d audio heads, codec=%s, text_embed=%s (text vocab %d)\n",
        self->n_audio_embed_loaded(),
        self->n_audio_head_loaded(),
        self->m_aux->codec_present ? "yes" : "no",
        self->m_aux->text_embed ? "yes" : "no",
        self->m_aux->text_vocab_size);

    // ── 3. Tokenizer wrapper ────────────────────────────────────────────────
    self->m_tokenizer = std::make_unique<Tokenizer>(self->m_backbone_model);

    return self;
}

// ───────────────────────────────────────────────────────────────────────────
// compute_input_embeddings:
//   prompt_grid: (n_pos, 1 + n_vq) row-major int32
//   returns:     (n_pos, hidden_size) row-major float32
//
// Builds a GGML graph each call (graph is tiny — one get_rows for text + n_vq
// gather-and-add steps). The graph's input tensors are populated from host
// arrays via ggml_backend_tensor_set, and the output is fetched back to host.
// ───────────────────────────────────────────────────────────────────────────

std::vector<float> Model::compute_input_embeddings(const int32_t * prompt_grid,
                                                   int32_t         n_pos) const {
    if (!m_aux || !m_aux->text_embed)
        throw std::runtime_error("compute_input_embeddings: text_embed not loaded");
    if (n_pos <= 0)
        return {};

    const int32_t n_vq    = m_aux->n_vq;
    const int32_t hidden  = m_aux->hidden_size;
    const int32_t stride  = 1 + n_vq;

    // Build graph context.
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (4 * n_vq + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("compute_input_embeddings: ggml_init failed");

    // Inputs: one int32 vector per channel (text + n_vq audio).
    ggml_tensor * text_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_pos);
    ggml_set_name(text_ids, "text_ids");
    ggml_set_input(text_ids);

    std::vector<ggml_tensor *> audio_ids(n_vq);
    for (int32_t i = 0; i < n_vq; ++i) {
        audio_ids[i] = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, n_pos);
        ggml_set_name(audio_ids[i], ("audio_ids_" + std::to_string(i)).c_str());
        ggml_set_input(audio_ids[i]);
    }

    // text_emb = get_rows(text_embed, text_ids)        → (hidden, n_pos)
    ggml_tensor * out = ggml_get_rows(gctx, m_aux->text_embed, text_ids);
    // Promote to f32 (text_embed is typically f16).
    out = ggml_cast(gctx, out, GGML_TYPE_F32);

    // Add each audio channel's embedding lookup.
    for (int32_t i = 0; i < n_vq; ++i) {
        ggml_tensor * w = m_aux->tensors.at("moss.audio_embed." + std::to_string(i) + ".weight");
        ggml_tensor * e = ggml_get_rows(gctx, w, audio_ids[i]);
        e = ggml_cast(gctx, e, GGML_TYPE_F32);
        out = ggml_add(gctx, out, e);
    }
    ggml_set_output(out);
    ggml_set_name(out, "input_embeds");

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, 4 * n_vq + 16, false);
    ggml_build_forward_expand(graph, out);

    // Allocate compute buffers from our reusable allocator.
    if (!ggml_gallocr_alloc_graph(m_aux->galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("compute_input_embeddings: gallocr_alloc_graph failed");
    }

    // Upload inputs.
    std::vector<int32_t> col(n_pos);
    for (int32_t r = 0; r < n_pos; ++r) col[r] = prompt_grid[r * stride + 0];
    ggml_backend_tensor_set(text_ids, col.data(), 0, n_pos * sizeof(int32_t));
    for (int32_t i = 0; i < n_vq; ++i) {
        for (int32_t r = 0; r < n_pos; ++r) col[r] = prompt_grid[r * stride + 1 + i];
        ggml_backend_tensor_set(audio_ids[i], col.data(), 0, n_pos * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(m_aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("compute_input_embeddings: graph_compute failed");
    }

    std::vector<float> result(size_t(n_pos) * size_t(hidden));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
    ggml_free(gctx);
    return result;
}

// ───────────────────────────────────────────────────────────────────────────
// compute_audio_logits:
//   hidden: (hidden_size,) f32
//   returns: (n_vq, audio_vocab_size + 1) row-major f32
//
// One ggml_mul_mat per audio head (32 of them); concatenated along ne[1] into
// a single output tensor.
// ───────────────────────────────────────────────────────────────────────────
std::vector<float> Model::compute_audio_logits(const float * hidden) const {
    if (!m_aux) throw std::runtime_error("compute_audio_logits: aux not loaded");
    const int32_t n_vq  = m_aux->n_vq;
    const int32_t Vfull = m_aux->audio_vocab_full;
    const int32_t hsz   = m_aux->hidden_size;

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * (n_vq * 2 + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) throw std::runtime_error("compute_audio_logits: ggml_init failed");

    // Input hidden state, treated as a column vector (hsz, 1).
    ggml_tensor * h = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, hsz, 1);
    ggml_set_name(h, "hidden");
    ggml_set_input(h);

    // Compute each head's logits and stack along ne[1].
    // Each head w_i has shape (hsz, Vfull) in GGML (= numpy (Vfull, hsz)).
    // mul_mat(w_i, h) → (Vfull, 1).
    ggml_tensor * stacked = nullptr;
    for (int32_t i = 0; i < n_vq; ++i) {
        ggml_tensor * w = m_aux->tensors.at("moss.audio_head." + std::to_string(i) + ".weight");
        ggml_tensor * y = ggml_mul_mat(gctx, w, h);  // (Vfull, 1)
        if (!stacked) stacked = y;
        else          stacked = ggml_concat(gctx, stacked, y, 1);
    }
    ggml_set_output(stacked);
    ggml_set_name(stacked, "audio_logits");

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_vq * 4 + 16, false);
    ggml_build_forward_expand(graph, stacked);

    if (!ggml_gallocr_alloc_graph(m_aux->galloc, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("compute_audio_logits: gallocr_alloc_graph failed");
    }

    ggml_backend_tensor_set(h, hidden, 0, hsz * sizeof(float));

    if (ggml_backend_graph_compute(m_aux->backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("compute_audio_logits: graph_compute failed");
    }

    std::vector<float> out(size_t(n_vq) * size_t(Vfull));
    ggml_backend_tensor_get(stacked, out.data(), 0, out.size() * sizeof(float));
    ggml_free(gctx);
    return out;
}

} // namespace openmoss
