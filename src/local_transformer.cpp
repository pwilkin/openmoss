// SPDX-License-Identifier: Apache-2.0
//
// Local ("depth") transformer for MOSS-TTS-Local — the piece that replaces the
// delay pattern.
//
// Per backbone frame the model emits exactly one frame of n_vq codes. A small
// GPT-2-style transformer (1 layer in v1.5) runs n_vq sequential micro-steps:
//
//   micro-step 0 : input is the backbone hidden state h_t. Its output feeds
//                  BOTH the binary continue/stop head and codebook 0's head.
//   micro-step k : input is the *raw embedding row* of the code sampled at
//                  k-1, i.e. audio_embeddings[k-1][c_{k-1}]. Output feeds
//                  codebook k's head.
//
// Two details that are easy to get wrong:
//
//   * The block is LayerNorm (weight AND bias) with a plain SiLU MLP —
//     fc_out(silu(fc_in(x))). There is a single fc_in, so it is NOT SwiGLU,
//     and NOT the RMSNorm/gated arrangement the Qwen3 backbone uses.
//
//   * RoPE here is the interleaved-pair variant (GGML_ROPE_TYPE_NORMAL) over
//     head_dim = n_embd / n_head = 80, while the backbone uses the NeoX
//     half-split variant at head_dim 128. Both at base 1e6. Two different rope
//     flavours in one model.
//
// The per-frame KV cache is reset for every frame — the depth transformer never
// attends across frames — and holds at most n_vq entries, so we keep it on the
// host and hand the prefix to each micro-step's graph as an input. That avoids
// in-graph cache mutation entirely.
//
// Why a KV prefix at all, when the sequence is only 12 long: re-running the
// whole prefix each micro-step would process 1+2+…+12 = 78 token-forwards per
// frame instead of 12. At ~152 MFLOP per token that is ~11.9 GFLOP/frame, more
// than the 8.04 GFLOP/frame the 4 B backbone itself costs. With the prefix it
// is ~1.8 GFLOP/frame.

#include "openmoss/frame_decoder.h"
#include "openmoss/model.h"

#include "aux_internal.h"
#include "sampling_internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace openmoss {

namespace {

// f16 weights must be promoted before elementwise ops: several backends only
// implement bin_bcast for (F32, F32, F32). Same reason as codec.cpp's to_f32_.
ggml_tensor * f32_(ggml_context * ctx, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                         ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, f32_(ctx, w));
    return ggml_add(ctx, x, f32_(ctx, b));
}

} // namespace

// ───────────────────────────────────────────────────────────────────────────
// LocalTransformer
// ───────────────────────────────────────────────────────────────────────────

class LocalTransformer {
public:
    explicit LocalTransformer(Model & owner) : m_owner(owner) {
        const auto & d = owner.dims();
        m_n_layer  = d.local_n_layer;
        m_n_embd   = d.local_n_embd;
        m_n_head   = d.local_n_head;
        m_n_inner  = d.local_n_inner;
        m_rope_base = d.local_rope_base;
        m_ln_eps    = d.local_ln_eps;

        if (m_n_layer <= 0 || m_n_embd <= 0 || m_n_head <= 0) {
            throw std::runtime_error("LocalTransformer: missing geometry in sidecar");
        }
        if (m_n_embd % m_n_head != 0) {
            throw std::runtime_error("LocalTransformer: n_embd " + std::to_string(m_n_embd)
                                     + " not divisible by n_head " + std::to_string(m_n_head));
        }
        m_head_dim = m_n_embd / m_n_head;
        if (m_n_embd != owner.dims().hidden_size) {
            // The depth transformer consumes the backbone hidden state directly
            // (_global_hidden_to_local is the identity upstream — there is no
            // projection matrix), so the widths must agree.
            throw std::runtime_error(
                "LocalTransformer: local n_embd (" + std::to_string(m_n_embd) +
                ") != backbone hidden_size (" + std::to_string(owner.dims().hidden_size) +
                "); no projection exists between them");
        }

        auto * aux = owner.aux();
        if (!aux || !aux->backend) {
            throw std::runtime_error("LocalTransformer: aux backend not initialised");
        }
        auto get = [&](const std::string & n) -> ggml_tensor * {
            auto it = aux->tensors.find(n);
            if (it == aux->tensors.end()) {
                throw std::runtime_error("LocalTransformer: missing tensor " + n);
            }
            return it->second;
        };

        m_layers.resize(size_t(m_n_layer));
        for (int i = 0; i < m_n_layer; ++i) {
            const std::string b = "moss.local.h." + std::to_string(i) + ".";
            LLayer & L = m_layers[size_t(i)];
            L.ln1_w    = get(b + "ln_1.weight");
            L.ln1_b    = get(b + "ln_1.bias");
            L.attn_w   = get(b + "attn.c_attn.weight");
            L.attn_b   = get(b + "attn.c_attn.bias");
            L.proj_w   = get(b + "attn.c_proj.weight");
            L.proj_b   = get(b + "attn.c_proj.bias");
            L.ln2_w    = get(b + "ln_2.weight");
            L.ln2_b    = get(b + "ln_2.bias");
            L.fc_in_w  = get(b + "mlp.fc_in.weight");
            L.fc_in_b  = get(b + "mlp.fc_in.bias");
            L.fc_out_w = get(b + "mlp.fc_out.weight");
            L.fc_out_b = get(b + "mlp.fc_out.bias");
        }
        m_lnf_w = get("moss.local.ln_f.weight");
        m_lnf_b = get("moss.local.ln_f.bias");

        m_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(aux->backend));
        if (!m_galloc) throw std::runtime_error("LocalTransformer: gallocr_new failed");
    }

    ~LocalTransformer() {
        if (m_galloc) ggml_gallocr_free(m_galloc);
    }

    int n_embd() const { return m_n_embd; }

    // Per-frame KV cache. Layout per layer: [pos][n_embd], i.e. the natural
    // (head_dim, n_head) flattening. Reset at the start of every frame.
    struct Cache {
        std::vector<float> k, v;   // n_layer * n_pos * n_embd
        int n_pos = 0;
    };

    void cache_reset(Cache & c) const {
        c.k.clear();
        c.v.clear();
        c.n_pos = 0;
    }

    // A head to project the micro-step output through: a weight tensor of shape
    // (n_embd, rows_total) and how many leading rows to actually use.
    //
    // The row limit matters: audio head k is the *embedding table* for codebook
    // k (they are tied), and the converter appends one all-zero pad row so the
    // summed input embedding can use a plain get_rows. Projecting through all
    // 1025 rows would make the pad code a sampleable class with logit 0.
    struct Head {
        ggml_tensor * w;
        int           rows;
    };

    // Run micro-step `pos`, appending its K/V to `cache`, and project the result
    // through each head. `x` is (n_embd,). Logits land in `logits_out`, one
    // vector per head. The post-ln_f state itself is never needed on the host —
    // micro-step k+1 is fed an embedding row, not u.
    void step(const float * x, int pos, Cache & cache,
              const std::vector<Head> & heads,
              std::vector<std::vector<float>> & logits_out) {
        auto * aux = m_owner.aux();
        const int T_prev = cache.n_pos;
        const int T      = T_prev + 1;
        if (pos != T_prev) {
            throw std::runtime_error("LocalTransformer::step: pos " + std::to_string(pos)
                                     + " does not follow cache length " + std::to_string(T_prev));
        }

        ggml_init_params ip{};
        ip.mem_size   = ggml_tensor_overhead() * size_t(64 * m_n_layer + 64)
                      + ggml_graph_overhead_custom(4096, false);
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) throw std::runtime_error("LocalTransformer::step: ggml_init failed");

        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m_n_embd, 1);
        ggml_set_name(inp, "x");
        ggml_set_input(inp);

        ggml_tensor * pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(pos_t, "pos");
        ggml_set_input(pos_t);

        std::vector<ggml_tensor *> k_prev(size_t(m_n_layer), nullptr);
        std::vector<ggml_tensor *> v_prev(size_t(m_n_layer), nullptr);
        std::vector<ggml_tensor *> k_new(size_t(m_n_layer), nullptr);
        std::vector<ggml_tensor *> v_new(size_t(m_n_layer), nullptr);

        ggml_tensor * h = inp;
        for (int li = 0; li < m_n_layer; ++li) {
            const LLayer & L = m_layers[size_t(li)];

            ggml_tensor * n = layer_norm(ctx, h, L.ln1_w, L.ln1_b, m_ln_eps);

            // Fused QKV: c_attn is (n_embd, 3*n_embd) in GGML ne order.
            ggml_tensor * qkv = ggml_mul_mat(ctx, L.attn_w, n);        // (3*n_embd, 1)
            qkv = ggml_add(ctx, qkv, f32_(ctx, L.attn_b));

            // torch splits along the last dim into contiguous q | k | v blocks.
            const size_t esz = ggml_element_size(qkv);
            auto slice = [&](int idx) {
                ggml_tensor * t = ggml_view_2d(ctx, qkv, m_n_embd, 1,
                                               qkv->nb[1], size_t(idx) * size_t(m_n_embd) * esz);
                return ggml_cont(ctx, ggml_reshape_3d(ctx, t, m_head_dim, m_n_head, 1));
            };
            ggml_tensor * q = slice(0);
            ggml_tensor * k = slice(1);
            ggml_tensor * v = slice(2);

            // Interleaved-pair RoPE (mode 0) over all head_dim dims. v is not rotated.
            q = ggml_rope_ext(ctx, q, pos_t, nullptr, m_head_dim, GGML_ROPE_TYPE_NORMAL,
                              /*n_ctx_orig=*/0, m_rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos_t, nullptr, m_head_dim, GGML_ROPE_TYPE_NORMAL,
                              /*n_ctx_orig=*/0, m_rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            k_new[size_t(li)] = k;   // (head_dim, n_head, 1) — post-RoPE, cached as-is
            v_new[size_t(li)] = v;
            ggml_set_output(k_new[size_t(li)]);
            ggml_set_output(v_new[size_t(li)]);

            // Concatenate the cached prefix along the token axis.
            ggml_tensor * k_all = k;
            ggml_tensor * v_all = v;
            if (T_prev > 0) {
                k_prev[size_t(li)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                        m_head_dim, m_n_head, T_prev);
                v_prev[size_t(li)] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                        m_head_dim, m_n_head, T_prev);
                ggml_set_input(k_prev[size_t(li)]);
                ggml_set_input(v_prev[size_t(li)]);
                k_all = ggml_concat(ctx, k_prev[size_t(li)], k, 2);   // (hd, nh, T)
                v_all = ggml_concat(ctx, v_prev[size_t(li)], v, 2);
            }

            // No mask needed: the cache holds exactly positions 0..pos, so
            // attending over all of it is already causal.
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q,     0, 2, 1, 3)); // (hd, 1, nh)
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k_all, 0, 2, 1, 3)); // (hd, T, nh)
            ggml_tensor * vh = ggml_cont(ctx, ggml_permute(ctx, v_all, 1, 2, 0, 3)); // (T, hd, nh)

            ggml_tensor * kq = ggml_mul_mat(ctx, kh, qh);                        // (T, 1, nh)
            kq = ggml_soft_max_ext(ctx, kq, nullptr,
                                   1.0f / std::sqrt(float(m_head_dim)), 0.0f);
            ggml_tensor * kqv = ggml_mul_mat(ctx, vh, kq);                       // (hd, 1, nh)
            kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));            // (hd, nh, 1)
            ggml_tensor * a = ggml_reshape_2d(ctx, kqv, m_n_embd, 1);

            a = ggml_mul_mat(ctx, L.proj_w, a);
            a = ggml_add(ctx, a, f32_(ctx, L.proj_b));
            h = ggml_add(ctx, h, a);

            // Plain SiLU MLP — one fc_in, no gate.
            ggml_tensor * m = layer_norm(ctx, h, L.ln2_w, L.ln2_b, m_ln_eps);
            m = ggml_mul_mat(ctx, L.fc_in_w, m);
            m = ggml_add(ctx, m, f32_(ctx, L.fc_in_b));
            m = ggml_silu(ctx, m);
            m = ggml_mul_mat(ctx, L.fc_out_w, m);
            m = ggml_add(ctx, m, f32_(ctx, L.fc_out_b));
            h = ggml_add(ctx, h, m);
        }

        ggml_tensor * u = layer_norm(ctx, h, m_lnf_w, m_lnf_b, m_ln_eps);
        ggml_set_name(u, "u");

        std::vector<ggml_tensor *> head_out(heads.size(), nullptr);
        for (size_t i = 0; i < heads.size(); ++i) {
            ggml_tensor * w = heads[i].w;
            if (heads[i].rows < w->ne[1]) {
                w = ggml_view_2d(ctx, w, w->ne[0], heads[i].rows, w->nb[1], 0);
            }
            head_out[i] = ggml_mul_mat(ctx, w, u);          // (rows, 1)
            ggml_set_output(head_out[i]);
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        for (auto * t : head_out) ggml_build_forward_expand(graph, t);
        for (int li = 0; li < m_n_layer; ++li) {
            ggml_build_forward_expand(graph, k_new[size_t(li)]);
            ggml_build_forward_expand(graph, v_new[size_t(li)]);
        }

        if (!ggml_gallocr_alloc_graph(m_galloc, graph)) {
            ggml_free(ctx);
            throw std::runtime_error("LocalTransformer::step: gallocr_alloc_graph failed");
        }

        ggml_backend_tensor_set(inp, x, 0, size_t(m_n_embd) * sizeof(float));
        const int32_t p = pos;
        ggml_backend_tensor_set(pos_t, &p, 0, sizeof(int32_t));
        if (T_prev > 0) {
            const size_t per_layer = size_t(T_prev) * size_t(m_n_embd) * sizeof(float);
            for (int li = 0; li < m_n_layer; ++li) {
                ggml_backend_tensor_set(k_prev[size_t(li)],
                                        cache.k.data() + size_t(li) * size_t(T_prev) * size_t(m_n_embd),
                                        0, per_layer);
                ggml_backend_tensor_set(v_prev[size_t(li)],
                                        cache.v.data() + size_t(li) * size_t(T_prev) * size_t(m_n_embd),
                                        0, per_layer);
            }
        }

        if (ggml_backend_graph_compute(aux->backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            throw std::runtime_error("LocalTransformer::step: graph_compute failed");
        }

        logits_out.resize(heads.size());
        for (size_t i = 0; i < heads.size(); ++i) {
            logits_out[i].resize(size_t(heads[i].rows));
            ggml_backend_tensor_get(head_out[i], logits_out[i].data(), 0,
                                    logits_out[i].size() * sizeof(float));
        }

        // Append the new K/V, keeping the per-layer blocks contiguous.
        {
            std::vector<float> nk(static_cast<size_t>(m_n_embd));
            std::vector<float> nv(static_cast<size_t>(m_n_embd));
            std::vector<float> new_k, new_v;
            new_k.reserve(size_t(m_n_layer) * size_t(T) * size_t(m_n_embd));
            new_v.reserve(size_t(m_n_layer) * size_t(T) * size_t(m_n_embd));
            for (int li = 0; li < m_n_layer; ++li) {
                ggml_backend_tensor_get(k_new[size_t(li)], nk.data(), 0,
                                        nk.size() * sizeof(float));
                ggml_backend_tensor_get(v_new[size_t(li)], nv.data(), 0,
                                        nv.size() * sizeof(float));
                const size_t off = size_t(li) * size_t(T_prev) * size_t(m_n_embd);
                new_k.insert(new_k.end(), cache.k.begin() + off,
                             cache.k.begin() + off + size_t(T_prev) * size_t(m_n_embd));
                new_k.insert(new_k.end(), nk.begin(), nk.end());
                new_v.insert(new_v.end(), cache.v.begin() + off,
                             cache.v.begin() + off + size_t(T_prev) * size_t(m_n_embd));
                new_v.insert(new_v.end(), nv.begin(), nv.end());
            }
            cache.k.swap(new_k);
            cache.v.swap(new_v);
            cache.n_pos = T;
        }

        ggml_free(ctx);
    }

private:
    struct LLayer {
        ggml_tensor *ln1_w, *ln1_b;
        ggml_tensor *attn_w, *attn_b;
        ggml_tensor *proj_w, *proj_b;
        ggml_tensor *ln2_w, *ln2_b;
        ggml_tensor *fc_in_w, *fc_in_b;
        ggml_tensor *fc_out_w, *fc_out_b;
    };

    Model & m_owner;
    int   m_n_layer = 0, m_n_embd = 0, m_n_head = 0, m_head_dim = 0, m_n_inner = 0;
    float m_rope_base = 1e6f, m_ln_eps = 1e-6f;
    std::vector<LLayer>  m_layers;
    ggml_tensor * m_lnf_w = nullptr;
    ggml_tensor * m_lnf_b = nullptr;
    ggml_gallocr_t m_galloc = nullptr;
};

// ───────────────────────────────────────────────────────────────────────────
// LocalFrameDecoder
// ───────────────────────────────────────────────────────────────────────────

namespace {

class LocalFrameDecoder final : public IFrameDecoder {
public:
    LocalFrameDecoder(Model & model,
                      const std::vector<std::vector<int32_t>> & prompt_rows)
        : m_model(model), m_dims(model.dims()), m_lt(model) {
        m_row_width = 1 + m_dims.n_vq;
        for (const auto & r : prompt_rows) {
            if (int32_t(r.size()) != m_row_width) {
                throw std::runtime_error(
                    "LocalFrameDecoder: prompt row width " + std::to_string(r.size())
                    + ", expected " + std::to_string(m_row_width));
            }
        }
        m_text_head = model.aux()->tensors.at("moss.local_text_head.weight");
        m_cb_history.resize(size_t(m_dims.n_vq));

        for (int k = 0; k < m_dims.n_vq; ++k) {
            ggml_tensor * e = model.audio_embed(k);
            if (!e) throw std::runtime_error("LocalFrameDecoder: missing audio embedding "
                                             + std::to_string(k));
            if (e->type != GGML_TYPE_F16) {
                throw std::runtime_error("LocalFrameDecoder: audio embedding " +
                                         std::to_string(k) + " is not f16");
            }
            m_embed.push_back(e);
        }
    }

    int32_t row_width() const override { return m_row_width; }

    FrameStep step(const float * text_logits,
                   const float * hidden,
                   const SamplingConfig & sc) override {
        // MOSS-TTS-Local never samples from the backbone's text vocabulary during
        // audio generation: continuation is decided by the 2-way local head, and
        // the text column of every generated row is the constant gen slot.
        (void)text_logits;

        if (!m_rng) m_rng = std::make_unique<Rng>(sc.seed);

        // The depth transformer never attends across frames — fresh cache each time.
        LocalTransformer::Cache cache;
        m_lt.cache_reset(cache);

        const int V = m_dims.audio_vocab_size;
        std::vector<std::vector<float>> logits;

        // Micro-step 0 drives BOTH the stop decision and codebook 0, from the
        // same output vector.
        m_lt.step(hidden, 0, cache,
                  { {m_text_head, 2}, {m_embed[0], V} }, logits);

        if (decide_stop(logits[0], sc)) {
            FrameStep s;
            s.stop = true;
            return s;      // the frame is discarded, not emitted
        }

        FrameStep out;
        out.ids.assign(size_t(m_row_width), m_dims.audio_pad_code);
        out.ids[0] = m_dims.audio_assistant_gen_slot_token_id;

        std::vector<int32_t> frame(size_t(m_dims.n_vq));
        frame[0]   = sample_code(logits[1], 0, sc);
        out.ids[1] = frame[0];

        std::vector<float> emb(size_t(m_dims.hidden_size));
        for (int k = 1; k < m_dims.n_vq; ++k) {
            // Micro-step k is fed the raw embedding row of the code sampled at
            // k-1 — not a projection of it, and not the previous output.
            read_embed_row(k - 1, frame[size_t(k - 1)], emb);
            m_lt.step(emb.data(), k, cache, { {m_embed[size_t(k)], V} }, logits);
            frame[size_t(k)] = sample_code(logits[0], k, sc);
            out.ids[size_t(1 + k)] = frame[size_t(k)];
        }

        m_frames.push_back(std::move(frame));
        return out;
    }

    std::vector<int32_t> extract_audio_codes(int32_t & n_cb_out,
                                             int32_t & t_audio_out) const override {
        n_cb_out    = m_dims.n_vq;
        t_audio_out = int32_t(m_frames.size());
        if (m_frames.empty()) return {};

        // Only the generated frames. Unlike the delay family there is nothing to
        // un-shift and no flush window to strip; and unlike upstream's tensor
        // layout there is no prompt audio to trim, because we never append the
        // prompt's rows here.
        std::vector<int32_t> out(size_t(n_cb_out) * size_t(t_audio_out));
        for (int cb = 0; cb < n_cb_out; ++cb) {
            for (size_t t = 0; t < m_frames.size(); ++t) {
                out[size_t(cb) * size_t(t_audio_out) + t] = m_frames[t][size_t(cb)];
            }
        }
        return out;
    }

private:
    // Returns true when the binary head says end-of-speech. Index 0 is the
    // assistant gen slot (continue), index 1 is <|audio_end|> (stop) — the row
    // order of local_text_lm_head, fixed by _local_text_candidate_ids upstream.
    bool decide_stop(std::vector<float> & lg, const SamplingConfig & sc) {
        const int n = int(m_frames.size());
        if (sc.min_audio_frames > 0 && n < sc.min_audio_frames) return false;
        if (sc.max_audio_frames > 0 && n >= sc.max_audio_frames) return true;
        // With the documented defaults (top_p 1.0, top_k 50) both filters are
        // inert over a 2-way vocab, so this reduces to a Bernoulli draw.
        // temperature <= 0 means greedy — used for reproducibility testing.
        return sampling::sample_one(lg.data(), 2, sc.text_temperature, sc.text_top_p,
                                    sc.text_top_k, sc.text_temperature > 0.f, *m_rng) == 1;
    }

    int32_t sample_code(std::vector<float> & lg, int cb, const SamplingConfig & sc) {
        const int V = m_dims.audio_vocab_size;
        // Penalty first, on the raw logits, over the unique ids this codebook has
        // already emitted during this generation — matching the reference order.
        sampling::apply_repetition_penalty(lg.data(), V, m_cb_history[size_t(cb)],
                                           sc.audio_repetition_penalty);
        const int32_t id = sampling::sample_one(lg.data(), V, sc.audio_temperature,
                                                sc.audio_top_p, sc.audio_top_k,
                                                sc.audio_temperature > 0.f, *m_rng);
        m_cb_history[size_t(cb)].push_back(id);
        return id;
    }

    // Pull one row out of an f16 embedding table straight off the backend. Far
    // cheaper than a graph for a single get_rows, and it keeps the micro-step
    // count at one dispatch each.
    void read_embed_row(int cb, int32_t code, std::vector<float> & out) {
        ggml_tensor * e = m_embed[size_t(cb)];
        const int64_t n = e->ne[0];
        m_row_f16.resize(size_t(n));
        ggml_backend_tensor_get(e, m_row_f16.data(),
                                size_t(code) * size_t(e->nb[1]),
                                size_t(n) * sizeof(ggml_fp16_t));
        out.resize(size_t(n));
        ggml_fp16_to_fp32_row(m_row_f16.data(), out.data(), n);
    }

    Model &          m_model;
    ModelDims        m_dims;
    LocalTransformer m_lt;
    int32_t          m_row_width = 0;
    ggml_tensor *    m_text_head = nullptr;
    std::vector<ggml_tensor *>        m_embed;
    std::vector<std::vector<int32_t>> m_frames;      // (T, n_vq)
    std::vector<std::vector<int32_t>> m_cb_history;  // per codebook, for the penalty
    std::vector<ggml_fp16_t>          m_row_f16;
    std::unique_ptr<Rng>              m_rng;
};

} // namespace

std::unique_ptr<IFrameDecoder> make_local_frame_decoder(
    Model & model, const std::vector<std::vector<int32_t>> & prompt_rows) {
    return std::make_unique<LocalFrameDecoder>(model, prompt_rows);
}

} // namespace openmoss
