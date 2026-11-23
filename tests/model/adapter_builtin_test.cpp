// tests/model/adapter_builtin_test.cpp - the program's own registration: every adapter the build holds, once each. Linking
// the models needs the CUDA runtime, so this runs where that is installed; no GPU work.
#include "granite/model/adapter.hpp"
#include "granite/models/builtin.hpp"

#include <cstdio>
#include <string>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    // BASALT_MODELS decides which adapters exist: each built model is registered once, each left out is refused by name
    size_t built = 0;
    const auto expect = [&](const char* arch, bool is_built) {
        built += is_built ? 1 : 0;
        check((model::find_adapter(arch) != nullptr) == is_built, is_built ? "a built model is registered" : "a model left out is not registered");
        if (!is_built) {
            const std::string why = model::unavailable_architecture(arch);
            check(why.find("left out") != std::string::npos && why.find("BASALT_MODELS=") != std::string::npos &&
                      why.find(std::string("-DBASALT_MODELS=\"")) != std::string::npos && why.find(arch) != std::string::npos,
                  "a model left out is refused with the option that adds it");
        }
    };
    // Basalt's other models are not granite's: unknown architectures, refused as such
    for (const char* arch : {"qwen4exp", "deepseek41", "qwen35", "hy4flash", "hy_v4_flash"}) {
        check(model::find_adapter(arch) == nullptr, "a model granite does not ship is not registered");
        check(model::unavailable_architecture(arch).find("no model adapter is registered") != std::string::npos,
              "a model granite does not ship is refused as unknown");
    }
#ifdef BASALT_MODEL_INKLING2
    expect("inkling2", true);
#else
    expect("inkling2", false);
#endif
    check(model::adapter_names().size() == built, "registering twice keeps one adapter per name");
    return failures ? 1 : 0;
}
