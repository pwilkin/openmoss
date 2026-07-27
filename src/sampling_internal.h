// SPDX-License-Identifier: Apache-2.0
//
// Sampling primitives shared by every model family's frame decoder.
//
// These are deliberately family-agnostic: logits in, token id out. Both the
// delay-pattern sampler (delay.cpp) and the local-transformer sampler
// (local_transformer.cpp) must apply them in the *same order* as the reference
// implementation — repetition penalty first, on the raw logits, then
// temperature, then top-k, then top-p, then a multinomial draw.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace openmoss {

// Forward-declared in delay.h so DelayState can hold a unique_ptr to one.
// Held per generation so each request seeds its own stream — a single static
// RNG would ignore per-request seeds and leak state across requests on the
// persistent server.
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) {
        if (seed == 0) {
            std::random_device rd;
            seed = (uint64_t(rd()) << 32) | rd();
        }
        g.seed(seed);
    }
    float uniform01() {
        return std::uniform_real_distribution<float>(0.f, 1.f)(g);
    }
};

namespace sampling {

// Penalize each token that appears in the history exactly ONCE, no matter how
// often it occurs — the reference runs the penalty over `torch.unique(...)`.
// Penalizing per occurrence compounds to penalty^k for a token seen k times;
// with a 1024-code audio vocab at 12.5 frames/s that distorts the distribution
// more every second.
inline void apply_repetition_penalty(float * logits, int vocab,
                                     const std::vector<int32_t> & history,
                                     float penalty) {
    if (penalty == 1.0f) return;
    std::vector<bool> seen(size_t(vocab), false);
    for (int32_t id : history) {
        if (id < 0 || id >= vocab || seen[size_t(id)]) continue;
        seen[size_t(id)] = true;
        if (logits[id] > 0) logits[id] /= penalty;
        else                logits[id] *= penalty;
    }
}

// Softmax in place over a contiguous logits buffer (with -inf entries as masks).
inline void softmax_inplace(float * x, int n) {
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (x[i] > mx) mx = x[i];
    if (!std::isfinite(mx)) {
        // All-masked; keep zeros for caller to handle.
        std::fill(x, x + n, 0.f);
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.f) { std::fill(x, x + n, 0.f); return; }
    const float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

inline int32_t sample_one(float * logits, int vocab, float temperature,
                          float top_p, int top_k, bool do_sample, Rng & rng) {
    if (!do_sample) {
        // Argmax
        int best = 0;
        for (int i = 1; i < vocab; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    }

    if (temperature > 0.f && temperature != 1.f) {
        const float inv = 1.f / temperature;
        for (int i = 0; i < vocab; ++i) logits[i] *= inv;
    }

    // top-k: build a list of (id, logit) pairs and pick the K largest.
    if (top_k > 0 && top_k < vocab) {
        std::vector<std::pair<float,int32_t>> v;
        v.reserve(size_t(vocab));
        for (int i = 0; i < vocab; ++i) {
            if (std::isfinite(logits[i])) v.emplace_back(logits[i], i);
        }
        if (int(v.size()) > top_k) {
            std::nth_element(v.begin(), v.begin() + top_k, v.end(),
                             [](auto & a, auto & b){ return a.first > b.first; });
            v.resize(size_t(top_k));
        }
        // Mask everything except the top-k by zeroing logits.
        std::vector<bool> keep(size_t(vocab), false);
        for (auto & p : v) keep[size_t(p.second)] = true;
        for (int i = 0; i < vocab; ++i)
            if (!keep[size_t(i)]) logits[i] = -std::numeric_limits<float>::infinity();
    }

    softmax_inplace(logits, vocab);

    // top-p: sort, accumulate, mask the tail.
    if (top_p > 0.f && top_p < 1.f) {
        std::vector<int32_t> order(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) order[size_t(i)] = i;
        std::sort(order.begin(), order.end(),
                  [&](int32_t a, int32_t b){ return logits[a] > logits[b]; });
        float acc = 0.f;
        size_t cut = order.size();
        for (size_t i = 0; i < order.size(); ++i) {
            acc += logits[order[i]];
            if (acc >= top_p) { cut = i + 1; break; }
        }
        // Renormalise the kept set.
        float sum = 0.f;
        for (size_t i = 0; i < cut;            ++i) sum += logits[order[i]];
        for (size_t i = cut; i < order.size(); ++i) logits[order[i]] = 0.f;
        if (sum > 0.f) {
            const float inv = 1.f / sum;
            for (size_t i = 0; i < cut; ++i) logits[order[i]] *= inv;
        }
    }

    // Multinomial draw via inverse CDF.
    const float u = rng.uniform01();
    float acc = 0.f;
    for (int i = 0; i < vocab; ++i) {
        acc += logits[i];
        if (acc >= u) return i;
    }
    // Fallback for floating-point sum < 1.0
    for (int i = vocab - 1; i >= 0; --i) if (logits[i] > 0.f) return i;
    return 0;
}

} // namespace sampling
} // namespace openmoss
