// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard and Inkling 2
// geometry from `inkling2.*` metadata. Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <cstdio>
#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt;
    test::register_meta_adapters();
    check(model::find_adapter("inkling2") != nullptr, "the adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "basalt_adapter_registry_test";
    std::filesystem::create_directories(dir);
    using namespace fixture;

    write(dir / "other.gguf", {str("general.architecture", "llama")}, {});
    check(model::check_architecture(GgufFile((dir / "other.gguf").string())).find("no model adapter") != std::string::npos,
          "an unregistered architecture is refused by name");

    write(dir / "inkling.gguf", {str("general.architecture", "inkling2")}, {});
    check(model::check_architecture(GgufFile((dir / "inkling.gguf").string())).find("block_count") != std::string::npos,
          "inkling2 without its keys is refused");

    const std::vector<Kv> ink = {str("general.architecture", "inkling2"), u32("inkling2.block_count", 24),
                                 u32("inkling2.embedding_length", 2048), u32("inkling2.expert_count", 128),
                                 u32("inkling2.expert_used_count", 8), u32("inkling2.expert_feed_forward_length", 768),
                                 u32("inkling2.loop.start", 6), u32("inkling2.loop.end", 18), u32("inkling2.loop.count", 3),
                                 u32("inkling2.attention.sliding_window", 512)};
    write(dir / "ink.gguf", ink, {});
    const GgufFile g((dir / "ink.gguf").string());
    check(model::check_architecture(g).empty(), "inkling2 with its keys validates");
    std::string err;
    const model::ModelAdapter* a = model::adapter_for(g, err);
    check(a != nullptr && std::string(a->arch()) == "inkling2", "adapter_for picks inkling2");
    if (a) {
        const model::ModelDesc d = a->describe(g);
        check(d.n_layers == 24 && d.n_embd == 2048 && d.n_expert == 128 && d.n_expert_used == 8 && d.n_ff_exp == 768,
              "describe() reads the geometry from the metadata");
    }
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
