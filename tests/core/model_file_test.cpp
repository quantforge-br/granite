// tests/core/model_file_test.cpp - ModelFile::open on scratch files. CPU only, no model. A .granite file opens; an
// upstream Basalt pack (the same container, magic BASALTMF) is refused with the script that converts it; another
// magic, another version and a short file are refused as such.  A GGUF opens with its metadata and tensor directory as
// the meta, its tensor data as the `gguf` section and its tokenizer as the tokenizer files; a cut GGUF, one of a split
// and a vocabulary that names a token twice are refused.
#include "gguf_fixture.hpp"

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

std::string text(const std::vector<uint8_t>& b) { return std::string((const char*) b.data(), b.size()); }

// The GGUF cases: a file with a byte-level BPE tokenizer and two tensors.
void gguf_cases(const std::string& base) {
    using namespace fixture;
    const std::string path = base + "_tok.gguf", cut = base + "_cut.gguf", split = base + "_split.gguf",
                      dup = base + "_dup.gguf";
    const std::vector<std::string> tokens = {"a", "b", "ab", "q\"\\", "<|end|>"};
    std::vector<Kv> kv = {str("general.architecture", "toy"), str("general.name", "Toy Model"), u32("toy.block_count", 2),
                          f32("toy.eps", 0.5f), boolean("toy.flag", true), str("tokenizer.ggml.model", "gpt2"),
                          str("tokenizer.ggml.pre", "toypre"), strs("tokenizer.ggml.tokens", tokens),
                          strs("tokenizer.ggml.merges", {"a b"}), i32s("tokenizer.ggml.token_type", {1, 1, 1, 1, 3}),
                          str("tokenizer.chat_template", "{{ messages }}")};
    const std::vector<Tensor> ts = {{"w.q8", {32, 3}, 8, 3, {}}, {"w.f32", {5}, 0, 9, {}}};
    const Written w = write(path, kv, ts);
    granite::core::ModelFile m;
    std::string err;
    const bool opened = m.open(path, err);
    if (!opened) std::printf("  %s\n", err.c_str());
    check(opened && m.gguf(), "a GGUF opens");
    if (!opened) return;
    std::string lerr;
    check(m.load_meta(lerr) && m.register_gguf_view(lerr), "load_meta and register_gguf_view pass on a GGUF");
    const granite::core::Json& meta = m.meta();
    const granite::core::Json* fmt = meta.get("format");
    const granite::core::Json* arch = meta.get("arch");
    const granite::core::Json* name = meta.get("name");
    check(fmt && fmt->s == "gguf" && arch && arch->s == "toy" && name && name->s == "Toy Model",
          "the meta names the format, the architecture and the model");
    const granite::core::Json* gkv = meta.get("gguf_kv");
    const granite::core::Json* blocks = gkv ? gkv->get("toy.block_count") : nullptr;
    const granite::core::Json* eps = gkv ? gkv->get("toy.eps") : nullptr;
    const granite::core::Json* flag = gkv ? gkv->get("toy.flag") : nullptr;
    const granite::core::Json* toks = gkv ? gkv->get("tokenizer.ggml.tokens") : nullptr;
    check(blocks && blocks->u64() == 2 && eps && eps->num() == 0.5 && flag && flag->b && toks && toks->is_array() &&
              toks->count == tokens.size() && toks->items.size() == tokens.size() && toks->items[2].s == "ab",
          "gguf_kv holds the metadata: integers, floats, booleans, arrays");
    const granite::core::Json* tt = meta.get("tensors");
    const granite::core::Json* q8 = tt ? tt->get("w.q8") : nullptr;
    const granite::core::Json* f32t = tt ? tt->get("w.f32") : nullptr;
    check(q8 && q8->get("type_id")->u64() == 8 && q8->get("offset")->u64() == w.offsets[0] &&
              q8->get("bytes")->u64() == 3 * 34 && q8->get("shape")->count == 2 && q8->get("shape")->items[1].u64() == 3 &&
              f32t && f32t->get("offset")->u64() == w.offsets[1] && f32t->get("bytes")->u64() == 20,
          "the tensor table has the type, shape, offset in the data and bytes");
    const granite::core::FileRange data = m.range("gguf");
    std::vector<uint8_t> payload(20);
    std::string rerr;
    bool same = data.offset == w.data_start && data.bytes == w.size - w.data_start &&
                granite::core::read_range_at(data, w.offsets[1], payload.data(), payload.size(), rerr);
    for (size_t i = 0; same && i < payload.size(); ++i) same = payload[i] == pattern(9, i);
    check(same, "the gguf section is the tensor data");
    std::vector<uint8_t> vocab, merges, types, config, tpl;
    const bool read = m.read("tokenizer/vocab.json", vocab, rerr) && m.read("tokenizer/merges.txt", merges, rerr) &&
                      m.read("tokenizer/token_type.json", types, rerr) && m.read("tokenizer/tokenizer.json", config, rerr) &&
                      m.read("tokenizer/chat_template.jinja", tpl, rerr);
    if (!read) std::printf("  %s\n", rerr.c_str());
    granite::core::Json vj, cj;
    const bool parsed = read && granite::core::parse_json((const char*) vocab.data(), vocab.size(), vj, rerr) &&
                        granite::core::parse_json((const char*) config.data(), config.size(), cj, rerr);
    const granite::core::Json* quoted = parsed ? vj.get("q\"\\") : nullptr;
    const granite::core::Json* end = parsed ? vj.get("<|end|>") : nullptr;
    check(parsed && vj.members.size() == tokens.size() && quoted && quoted->u64() == 3 && end && end->u64() == 4,
          "tokenizer/vocab.json maps each token (escaped) to its id");
    check(read && text(merges) == "a b" && text(types) == "[1, 1, 1, 1, 3]" && text(tpl) == "{{ messages }}",
          "merges.txt, token_type.json and chat_template.jinja");
    check(parsed && text(config).find("\"pre\": \"toypre\"") != std::string::npos && cj.get("vocab_size")->u64() == 5,
          "tokenizer.json names the pre-tokenizer");
    check(m.range("tokenizer/vocab.json").empty(), "a tokenizer file held in memory has no range of the file");

    write(cut, kv, ts, 7);
    check(refused(cut, "runs past the end of the file"), "a cut GGUF is refused");
    std::vector<Kv> skv = kv;
    for (const Kv& k : split_keys(0, 2, 2)) skv.push_back(k);
    write(split, skv, ts);
    check(refused(split, "one of 2 split GGUF files"), "a GGUF of a split is refused");
    std::vector<Kv> dkv = kv;
    for (Kv& k : dkv)
        if (k.key == "tokenizer.ggml.tokens") k.strs[1] = "a";
    write(dup, dkv, ts);
    granite::core::ModelFile d;
    std::vector<uint8_t> v;
    std::string derr;
    check(d.open(dup, derr) && !d.read("tokenizer/vocab.json", v, derr) && derr.find("the token \"a\" twice") != std::string::npos,
          "a vocabulary with a token twice is refused when the tokenizer is read");
    for (const std::string& p : {path, cut, split, dup}) std::remove(p.c_str());
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
        !write_header(bad, "NOTAFILE", kModelVersion, kHeaderBytes) ||
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
    gguf_cases(base);
    std::printf("model_file: %s\n", failures ? "FAILED" : "all passed");
    return failures;
}
