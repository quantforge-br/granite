// tests/model/meta_adapters.hpp - registers an adapter per architecture that has the model's metadata rules and nothing
// else, so a CPU test links no model and no runtime.
#pragma once

#include "granite/model/adapter.hpp"
#include "models/inkling2/meta.hpp"

#include <memory>
#include <string>

namespace granite::test {

class MetaAdapter final : public model::ModelAdapter {
public:
    using Validate = std::string (*)(const GgufFile&);
    using Describe = model::ModelDesc (*)(const GgufFile&);
    MetaAdapter(const char* arch, const char* gguf_arch, Validate validate, Describe describe)
        : arch_(arch), gguf_arch_(gguf_arch), validate_(validate), describe_(describe) {}
    const char* arch() const override { return arch_; }
    const char* gguf_arch() const override { return gguf_arch_; }
    std::string validate(const GgufFile& g) const override { return validate_(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_(g); }
    model::TokenizerSpec tokenizer_spec() const override { return {}; }
    int max_expert_cards() const override { return 0; }
    model::TokenNames token_names() const override { return {}; }
    PreTokenizer pretokenizer() const override { return {}; }
    program::grammar::ToolForm tool_form() const override { return {}; }

private:
    const char* arch_;
    const char* gguf_arch_;
    Validate validate_;
    Describe describe_;
};

inline void register_meta_adapters() {
    if (model::find_adapter("inkling2") == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>(models::inkling2::kArch, models::inkling2::kGgufArch,
                                                              models::inkling2::validate_meta, models::inkling2::describe_meta));
}

}  // namespace granite::test
