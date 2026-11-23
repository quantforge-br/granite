// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture (and by llama.cpp's name for
// it), the replaced guard and Inkling 2 geometry from `inkling2.*` and `inkling2_loop.*` metadata. Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "granite/model/adapter.hpp"
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
    using namespace granite;
    test::register_meta_adapters();
    check(model::find_adapter("inkling2") != nullptr, "the adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "granite_adapter_registry_test";
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
    check(model::find_adapter("inkling2_loop") == model::find_adapter("inkling2"),
          "llama.cpp's architecture name finds the same adapter");
    bool listed = false;
    for (const std::string& n : model::adapter_names()) listed |= n == "inkling2_loop";
    check(!listed, "the GGUF's name is not listed as an adapter of its own");
    const std::vector<Kv> loop = {str("general.architecture", "inkling2_loop"), u32("inkling2_loop.block_count", 24),
                                  u32("inkling2_loop.embedding_length", 2304), u32("inkling2_loop.expert_count", 128),
                                  u32("inkling2_loop.expert_used_count", 8),
                                  u32("inkling2_loop.expert_feed_forward_length", 1152), u32("inkling2_loop.loop.start", 6),
                                  u32("inkling2_loop.loop.end", 18), u32("inkling2_loop.loop.count", 3),
                                  u32("inkling2_loop.attention.sliding_window", 1024)};
    write(dir / "loop.gguf", loop, {});
    const GgufFile lg((dir / "loop.gguf").string());
    check(model::check_architecture(lg).empty(), "llama.cpp's inkling2_loop GGUF validates under its own key prefix");
    const model::ModelAdapter* la = model::adapter_for(lg, err);
    if (la) {
        const model::ModelDesc d = la->describe(lg);
        check(std::string(la->arch()) == "inkling2" && d.n_layers == 24 && d.n_embd == 2304 && d.n_ff_exp == 1152,
              "describe() reads inkling2_loop.* keys");
    } else {
        check(false, "adapter_for picks inkling2 for an inkling2_loop GGUF");
    }
    write(dir / "loop_bare.gguf", {str("general.architecture", "inkling2_loop"), u32("inkling2.block_count", 24)}, {});
    check(model::check_architecture(GgufFile((dir / "loop_bare.gguf").string())).find("inkling2_loop.block_count") !=
              std::string::npos,
          "an inkling2_loop GGUF is read under its own prefix, not the pack's");
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
