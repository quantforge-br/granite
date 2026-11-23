// tests/core/model_file_test.cpp - ModelFile::open on header-only scratch files. CPU only, no model. A .granite file
// opens; an upstream Basalt pack (the same container, magic BASALTMF) is refused with the script that converts it;
// another magic, another version and a short file are refused as such.
#include "granite/core/model_file.hpp"
#include "granite/core/model_format.hpp"
#include "granite/platform/file.hpp"
#include "granite/platform/system.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("model_file %-60s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}

/// A file of `bytes` bytes that starts with a header of no sections.
bool write_header(const std::string& path, const char (&magic)[9], uint32_t version, uint64_t bytes) {
    granite::core::FileHeader h{};
    std::memcpy(h.magic, magic, sizeof h.magic);
    h.version = version;
    h.file_bytes = bytes;
    std::vector<uint8_t> buf((size_t) bytes, 0);
    std::memcpy(buf.data(), &h, std::min(sizeof h, buf.size()));
    std::FILE* f = granite::platform::open_file_utf8(path, "wb");
    if (!f) return false;
    const bool ok = std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
    return std::fclose(f) == 0 && ok;
}

/// Whether opening `path` fails with an error that contains `want`.
bool refused(const std::string& path, const std::string& want) {
    granite::core::ModelFile m;
    std::string err;
    if (m.open(path, err)) return false;
    std::printf("  %s\n", err.c_str());
    return err.find(want) != std::string::npos;
}

}  // namespace

int main() {
    using granite::core::kHeaderBytes;
    using granite::core::kModelVersion;
    const std::string base = granite::platform::temp_dir() + "/granite_model_file_test_" +
                             std::to_string((long long) granite::platform::process_id());
    const std::string ok = base + "_ok.granite", old = base + "_old.basalt", bad = base + "_bad.granite",
                      ver = base + "_ver.granite", cut = base + "_cut.granite";
    if (!write_header(ok, "GRANITMF", kModelVersion, kHeaderBytes) ||
        !write_header(old, "BASALTMF", kModelVersion, kHeaderBytes) ||
        !write_header(bad, "GGUF\3\0\0\0", kModelVersion, kHeaderBytes) ||
        !write_header(ver, "GRANITMF", kModelVersion + 1, kHeaderBytes) ||
        !write_header(cut, "GRANITMF", kModelVersion, kHeaderBytes - 1)) {
        std::printf("model_file: cannot write the scratch files under %s\n", base.c_str());
        return 1;
    }
    {
        granite::core::ModelFile m;
        std::string err;
        check(m.open(ok, err) && m.sections().empty() && m.file_bytes() == kHeaderBytes, "a .granite file opens");
    }
    check(refused(old, "a Basalt pack; convert it with tools/migrations/pack_to_granite.py"),
          "a Basalt pack is refused, naming the script that converts it");
    check(refused(bad, "not a .granite model file (bad magic)"), "another magic is refused");
    check(refused(ver, ".granite version " + std::to_string(kModelVersion + 1) + ", this build reads version " +
                           std::to_string(kModelVersion)),
          "another version is refused");
    check(refused(cut, "not a .granite model file (too short)"), "a file shorter than the header is refused");
    for (const std::string& p : {ok, old, bad, ver, cut}) std::remove(p.c_str());
    std::printf("model_file: %s\n", failures ? "FAILED" : "all passed");
    return failures;
}
