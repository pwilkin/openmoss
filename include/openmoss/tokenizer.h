// SPDX-License-Identifier: Apache-2.0
//
// Thin wrapper over libllama's tokenizer (we reuse the Qwen3 BPE that the
// converter writes into the GGUF metadata).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

namespace openmoss {

class Tokenizer {
public:
    explicit Tokenizer(llama_model * model);

    std::vector<int32_t> encode(const std::string & text, bool add_special = false) const;
    std::string          decode(const std::vector<int32_t> & ids) const;

    int32_t vocab_size() const;

private:
    llama_model * m_model;
};

} // namespace openmoss
