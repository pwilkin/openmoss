// SPDX-License-Identifier: Apache-2.0

#include "openmoss/tokenizer.h"

#include <stdexcept>
#include <vector>

#include "llama.h"

namespace openmoss {

Tokenizer::Tokenizer(llama_model * model) : m_model(model) {}

int32_t Tokenizer::vocab_size() const {
    const llama_vocab * v = llama_model_get_vocab(m_model);
    return llama_vocab_n_tokens(v);
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool add_special) const {
    const llama_vocab * v = llama_model_get_vocab(m_model);

    // First call to size; second call to fill.
    const int32_t cap_guess = int32_t(text.size()) + 8;
    std::vector<llama_token> ids(cap_guess);
    int32_t n = llama_tokenize(v, text.c_str(), int32_t(text.size()),
                               ids.data(), int32_t(ids.size()),
                               add_special, /*parse_special=*/true);
    if (n < 0) {
        ids.resize(-n);
        n = llama_tokenize(v, text.c_str(), int32_t(text.size()),
                           ids.data(), int32_t(ids.size()),
                           add_special, /*parse_special=*/true);
        if (n < 0) throw std::runtime_error("Tokenizer::encode failed");
    }
    ids.resize(n);
    return std::vector<int32_t>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    const llama_vocab * v = llama_model_get_vocab(m_model);
    std::string out;
    out.resize(ids.size() * 4 + 16); // rough upper bound
    int32_t n = llama_detokenize(v, ids.data(), int32_t(ids.size()),
                                  out.data(), int32_t(out.size()),
                                  /*remove_special=*/false,
                                  /*unparse_special=*/true);
    if (n < 0) {
        out.resize(-n);
        n = llama_detokenize(v, ids.data(), int32_t(ids.size()),
                              out.data(), int32_t(out.size()),
                              /*remove_special=*/false,
                              /*unparse_special=*/true);
        if (n < 0) throw std::runtime_error("Tokenizer::decode failed");
    }
    out.resize(n);
    return out;
}

} // namespace openmoss
