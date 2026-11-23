// src/models/inkling2/pack.cpp - reads the Inkling 2 pack's meta section, or llama.cpp's inkling2_loop GGUF.
#include "models/inkling2/pack.hpp"

#include "granite/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

namespace granite::models::inkling2 {
namespace {

using core::Json;

double num_or(const Json* j, double d) { return j && j->is_number() ? j->num() : d; }

std::vector<bool> bools(const Json* j) {
    std::vector<bool> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back(x.is_number() ? x.num() != 0 : x.b);
    return v;
}

std::vector<int> ints(const Json* j) {
    std::vector<int> v;
    if (j && j->is_array())
        for (const Json& x : j->items) v.push_back((int) x.num());
    return v;
}

// A count the pack writes once and a GGUF once a layer: the value, or -1 when the layers differ.
int uniform(const Json* j) {
    if (j && j->is_number()) return (int) j->num();
    if (!j || !j->is_array() || j->items.empty() || j->items.size() != j->count) return -1;
    for (const Json& x : j->items)
        if (x.num() != j->items[0].num()) return -1;
    return (int) j->items[0].num();
}

constexpr int kTypeF32 = 0;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kTypeNvfp4 = 40;
constexpr uint64_t kTail = 16;   // an NVFP4 expert's {s_gate, s_up, s_down, 0}

std::string type_name(int t) {
    const std::string n = ggml_type_name((uint32_t) t);
    return n != "?" ? n : "type " + std::to_string(t);
}

// The F32 values of a GGUF tensor of `n` values; false with `err` when it is not that.
bool read_f32(const Pack& p, const TensorRef& t, uint64_t n, std::vector<float>& out, std::string& err) {
    if (t.type != kTypeF32 || t.bytes != n * 4) {
        err = t.name + " is not F32 of " + std::to_string(n) + " values";
        return false;
    }
    out.resize((size_t) n);
    return core::read_range_at(core::FileRange(p.file.path()), t.offset, out.data(), t.bytes, err);
}

// llama.cpp's names and shapes onto the pack's: the sink rows' name, the vocabulary's padding rows, the relative bias
// table's orientation, and the experts' fused tensors and scales.
bool map_gguf(Pack& p, std::string& err) {
    const Geometry& g = p.geo;
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        if (auto it = p.tensors.find(b + "attn_sink_proj.weight"); it != p.tensors.end()) {
            TensorRef r = it->second;
            r.name = b + "attn_sink.weight";
            p.tensors.erase(it);
            p.tensors[r.name] = std::move(r);
        }
    }
    // llama.cpp pads the vocabulary to a multiple of 64 rows; the rows past the tokenizer's are not the model's
    if (const Json* v = p.file.meta().get("gguf_kv")->get(std::string(kGgufArch) + ".vocab_size_unpadded");
        v && v->is_number() && v->u64() != (uint64_t) g.n_vocab) {
        err = "the tokenizer has " + std::to_string(g.n_vocab) + " tokens, the model " + std::to_string(v->u64()) + " (vocab_size_unpadded)";
        return false;
    }
    for (const char* name : {"token_embd.weight", "output.weight"}) {
        auto it = p.tensors.find(name);
        if (it == p.tensors.end()) continue;
        TensorRef& t = it->second;
        if (t.shape.size() != 2 || t.shape[1] < (uint64_t) g.n_vocab || t.bytes % t.shape[1]) {
            err = std::string(name) + " has fewer rows than the vocabulary";
            return false;
        }
        t.bytes = t.bytes / t.shape[1] * (uint64_t) g.n_vocab;
        t.shape[1] = (uint64_t) g.n_vocab;
    }
    // the relative bias table is [extent][rel] in llama.cpp's file, [rel][extent] in the pack
    const uint64_t rel = (uint64_t) g.rel, extent = (uint64_t) g.extent;
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string name = "blk." + std::to_string(l) + ".attn_rel_bias.weight";
        auto it = p.tensors.find(name);
        if (it == p.tensors.end()) continue;
        TensorRef& t = it->second;
        std::vector<float> in;
        if (t.shape != std::vector<uint64_t>{rel, extent} || !read_f32(p, t, rel * extent, in, err)) {
            if (err.empty()) err = name + " is not F32 [" + std::to_string(rel) + ", " + std::to_string(extent) + "]";
            return false;
        }
        std::vector<uint8_t>& out = p.held[name];
        out.resize((size_t) (rel * extent * 4));
        float* o = reinterpret_cast<float*>(out.data());
        for (uint64_t r = 0; r < rel; ++r)
            for (uint64_t x = 0; x < extent; ++x) o[r * extent + x] = in[(size_t) (x * rel + r)];
        t.shape = {extent, rel};
        t.mem = out.data();
    }
    // the experts: gate and up rows of every expert fused in one tensor, down rows in another; an NVFP4 layer's per-expert
    // scales in F32 tensors of their own (1 when absent)
    const uint64_t ne = (uint64_t) g.n_expert, nf = (uint64_t) g.n_ff_exp, nd = (uint64_t) g.n_embd;
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    for (int l = 0; l < g.n_layers; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const TensorRef* gu = p.find(b + "ffn_gate_up_exps.weight");
        const TensorRef* dn = p.find(b + "ffn_down_exps.weight");
        if (!gu || !dn) {
            err = "layer " + std::to_string(l) + " has no ffn_gate_up_exps and ffn_down_exps (separate gate and up expert "
                  "tensors are not read)";
            return false;
        }
        for (const TensorRef* t : {gu, dn})
            if (t->bytes == 0) {
                err = t->name + " is " + type_name(t->type) + ", which this engine does not read";
                return false;
            }
        if (gu->shape != std::vector<uint64_t>{nd, 2 * nf, ne} || dn->shape != std::vector<uint64_t>{nf, nd, ne} ||
            gu->bytes % (2 * nf * ne) || dn->bytes % (nd * ne)) {
            err = "layer " + std::to_string(l) + "'s expert tensors are not [" + std::to_string(nd) + ", " + std::to_string(2 * nf) +
                  ", " + std::to_string(ne) + "] and [" + std::to_string(nf) + ", " + std::to_string(nd) + ", " + std::to_string(ne) + "]";
            return false;
        }
        ExpertLayer& el = p.experts[(size_t) l];
        el.gu_type = gu->type;
        el.d_type = dn->type;
        el.offset = gu->offset;
        el.gu_bytes = gu->bytes / ne;
        el.d_offset = dn->offset;
        el.d_bytes = dn->bytes / ne;
        const TensorRef* gs = p.find(b + "ffn_gate_up_exps.scale");
        const TensorRef* ds = p.find(b + "ffn_down_exps.scale");
        if ((el.gu_type == kTypeNvfp4) != (el.d_type == kTypeNvfp4)) {
            err = "layer " + std::to_string(l) + "'s experts mix NVFP4 and " +
                  type_name(el.gu_type == kTypeNvfp4 ? el.d_type : el.gu_type) + " (one scale tail covers an NVFP4 expert's three matrices)";
            return false;
        }
        if (el.gu_type != kTypeNvfp4) {
            if (gs || ds) {
                err = "layer " + std::to_string(l) + "'s " + type_name(el.gu_type) + " experts carry per-expert scales, which "
                      "only NVFP4 experts keep";
                return false;
            }
            el.blob_bytes = el.gu_bytes + el.d_bytes;
            continue;
        }
        std::vector<float> sg(ne, 1.f), sd(ne, 1.f);
        if ((gs && !read_f32(p, *gs, ne, sg, err)) || (ds && !read_f32(p, *ds, ne, sd, err))) return false;
        std::vector<uint8_t>& tails = p.held[b + "experts.tails"];
        tails.assign((size_t) (ne * kTail), 0);
        for (uint64_t e = 0; e < ne; ++e) {
            const float t[4] = {sg[e], sg[e], sd[e], 0.f};
            std::memcpy(tails.data() + e * kTail, t, kTail);
        }
        el.tails = tails.data();
        el.blob_bytes = el.gu_bytes + el.d_bytes + kTail;
    }
    return true;
}

}  // namespace

std::vector<Visit> loop_visits(int layers, int start, int end, int passes) {
    std::vector<Visit> v;
    for (int l = 0; l < start; ++l) v.push_back({l, 0, l});
    for (int p = 0; p < passes; ++p)
        for (int l = start; l < end; ++l) v.push_back({l, p, p == 0 ? l : layers + (p - 1) * (end - start) + (l - start)});
    for (int l = end; l < layers; ++l) v.push_back({l, 0, l});
    return v;
}

bool open_pack(const std::string& path, Pack& p, std::string& err) {
    if (!p.file.open(path, err) || !p.file.load_meta(err)) return false;
    const bool gguf = p.file.gguf();
    const std::string want = gguf ? kGgufArch : kArch;
    const Json& meta = p.file.meta();
    const Json* arch = meta.get("arch");
    if (!arch || !arch->is_string() || arch->s != want) {
        err = path + (gguf ? ": a GGUF of architecture '" + (arch ? arch->s : std::string()) + "', not llama.cpp's " + kGgufArch
                           : std::string(": not an inkling2 pack"));
        return false;
    }
    const Json* kv = meta.get("gguf_kv");
    const Json* tj = meta.get("tensors");
    if (!kv || !kv->is_object() || !tj || !tj->is_object()) {
        err = path + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto k = [&](const char* s) { return kv->get(want + "." + s); };
    auto u = [&](const char* s, double d = 0) { return (int) num_or(k(s), d); };
    auto f = [&](const char* s, double d = 0) { return (float) num_or(k(s), d); };
    Geometry& g = p.geo;
    g.n_layers = u("block_count");
    g.n_embd = u("embedding_length");
    g.n_head = uniform(k("attention.head_count"));
    g.n_head_kv = uniform(k("attention.head_count_kv"));
    g.head_dim = u("attention.key_length");
    g.rms_eps = f("attention.layer_norm_rms_epsilon", 1e-6);
    g.window = u("attention.sliding_window");
    g.swa = bools(k("attention.sliding_window_pattern"));
    g.rel = u("attention.relative_length");
    g.extent = u("attention.relative_extent");
    g.qk_clip = f("attention.qk_clip");
    g.rel_clip = f("attention.relative_clip");
    g.log_floor = f("attention.log_scale_floor");
    g.log_alpha = f("attention.log_scale_alpha");
    g.conv = u("shortconv.l_cache");
    g.block = u("attn_res.block_size");
    g.n_expert = u("expert_count");
    g.n_used = u("expert_used_count");
    g.n_ff_exp = u("expert_feed_forward_length");
    g.expert_scale = f("expert_weights_scale", 1.0);
    // llama.cpp names the softcapped SwiGLU's caps after its activation
    g.gate_softcap = f(gguf ? "activation.situ_beta" : "expert_gate_softcap");
    g.up_softcap = f(gguf ? "activation.situ_linear_beta" : "expert_up_softcap");
    g.logit_softcap = f("final_logit_softcapping");
    g.loop_start = u("loop.start");
    g.loop_end = u("loop.end");
    g.loop_passes = u("loop.count", 1);
    g.gate_rank = u("loop.gate_rank");
    g.image_token = u("image_token_id", -1);
    g.context_length = (int64_t) num_or(k("context_length"), 0);
    if (const Json* t = kv->get("tokenizer.ggml.tokens")) g.n_vocab = (int) t->count;
    if (g.n_layers <= 0 || (int) g.swa.size() != g.n_layers || g.n_vocab <= 0 || g.block <= 0) {
        err = path + ": meta.gguf_kv's " + want + " geometry is incomplete";
        return false;
    }
    if (const Json* v = k("loop.varying"); v && (v->b || v->num() != 0)) {
        err = path + ": a loop whose passes have weights of their own (loop.varying), which Inkling 2 Mini has not";
        return false;
    }
    // the layout the kernels are built for (granite/models/inkling2/kernels.hpp)
    if (g.n_embd != 2304 || g.n_head != 24 || g.n_head_kv != 8 || g.head_dim != 128 || g.rel != 16 || g.extent != 1024 ||
        g.window != g.extent || g.conv != 4 || g.n_expert > 512 || g.n_expert % 32 || g.n_used > 16 ||
        g.loop_start <= 0 || g.loop_start >= g.loop_end || g.loop_end > g.n_layers || g.loop_start % g.block ||
        g.loop_end % g.block || g.loop_passes < 1 || g.gate_rank != 64) {
        err = path + ": an Inkling 2 layout this engine does not run (width 2304, 24 query and 8 key/value heads of 128, "
                     "relative states of 16 over 1,024 positions, the window 1,024, convolutions of 4, a loop on block "
                     "boundaries, controllers of rank 64)";
        return false;
    }
    g.visits = loop_visits(g.n_layers, g.loop_start, g.loop_end, g.loop_passes);

    DsparkGeometry& d = p.dspark;
    if (const Json* dl = k("dspark.block_count"); dl && dl->is_number()) {
        d.present = true;
        d.n_layers = u("dspark.block_count");
        d.block = u("dspark.block_size");
        d.targets = ints(k("dspark.target_layers"));
        d.swa = bools(k("dspark.sliding_window_pattern"));
        d.window = u("dspark.sliding_window");
        d.n_head = u("dspark.attention.head_count");
        d.n_head_kv = u("dspark.attention.head_count_kv");
        d.head_dim = u("dspark.attention.key_length");
        d.n_ff = u("dspark.feed_forward_length");
        d.rope_base = f("dspark.rope.freq_base", 1e6);
        d.rms_eps = f("dspark.attention.layer_norm_rms_epsilon", 1e-6);
        d.markov_rank = u("dspark.markov_rank");
    }

    p.vision = p.file.range("vision/mmproj");
    for (const auto& [name, t] : tj->members) {
        TensorRef r;
        r.name = name;
        r.type = (int) num_or(t.get("type_id"), 0);
        if (const Json* sh = t.get("shape"))
            for (const Json& dd : sh->items) r.shape.push_back(dd.u64());
        r.bytes = (uint64_t) num_or(t.get("bytes"), 0);
        std::string section = "gguf";
        if (const Json* s = t.get("section"); s && s->is_string()) section = s->s;
        const core::ModelFile::Section* sec = p.file.find(section);
        if (!sec) {
            err = path + ": tensor " + name + " names a missing section " + section;
            return false;
        }
        r.offset = sec->offset + t.get("offset")->u64();
        p.tensors.emplace(name, std::move(r));
    }
    if (gguf) {
        if (!map_gguf(p, err)) {
            err = path + ": " + err;
            return false;
        }
        return true;
    }

    const core::ModelFile::Section* experts = p.file.find("experts");
    if (!experts) {
        err = path + ": no experts section";
        return false;
    }
    std::vector<uint8_t> text;
    if (!p.file.read("experts.layout", text, err)) return false;
    std::istringstream in(std::string(text.begin(), text.end()));
    std::string line;
    p.experts.assign((size_t) g.n_layers, ExpertLayer{});
    std::vector<bool> seen((size_t) g.n_layers, false);
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        long long l, gu, dt;
        unsigned long long off, by;
        if (!(ls >> l >> gu >> dt >> off >> by) || l < 0 || l >= g.n_layers || seen[(size_t) l] ||
            off > experts->bytes || by * (uint64_t) g.n_expert > experts->bytes - off) {
            err = "experts.layout: bad line '" + line + "'";
            return false;
        }
        p.experts[(size_t) l] = {(int) gu, (int) dt, experts->offset + off, by};
        seen[(size_t) l] = true;
    }
    for (int l = 0; l < g.n_layers; ++l)
        if (!seen[(size_t) l]) {
            err = "experts.layout lists no experts for layer " + std::to_string(l);
            return false;
        }
    return true;
}

void TensorRef::source(int file, core::LoadItem& it) const {
    if (mem) {
        it.from = mem;
    } else {
        it.file = file;
        it.off = offset;
    }
    it.bytes = bytes;
}

bool Pack::source(const std::string& name, Form form, uint64_t values, int file, core::LoadItem& it, std::string& err) const {
    const TensorRef* t = find(name);
    if (!t) {
        err = name + " is missing";
        return false;
    }
    const bool f32 = t->type == kTypeF32 && t->bytes == values * 4;
    const bool bf16 = t->type == kTypeBF16 && t->bytes == values * 2;
    const bool q8 = t->type == kTypeQ8_0 && values % 32 == 0 && t->bytes == values / 32 * 34;
    core::Conv conv = core::Conv::Copy;
    bool ok = false;
    const char* want = "";
    switch (form) {
        case Form::F32: ok = f32, want = "F32"; break;
        case Form::Bf16: ok = bf16 || f32, want = "BF16 or F32"; conv = f32 ? core::Conv::Bf16High : conv; break;
        case Form::Q8_0: ok = q8 || bf16, want = "Q8_0 or BF16"; conv = bf16 ? core::Conv::Bf16ToQ8_0 : conv; break;
    }
    if (!ok) {
        err = name + " is " + type_name(t->type) + " of " + std::to_string(t->bytes) + " bytes, not " + want + " of " +
              std::to_string(values) + " values";
        return false;
    }
    t->source(file, it);
    it.conv = conv;
    return true;
}

bool Pack::experts_contiguous(int l) const { return experts[(size_t) l].gu_bytes == 0; }

void Pack::expert_spans(int l, int e, uint64_t at, uint64_t n, std::vector<BlobSpan>& out) const {
    const ExpertLayer& el = experts[(size_t) l];
    if (el.gu_bytes == 0) {
        out.push_back({el.offset + (uint64_t) e * el.blob_bytes + at, n, nullptr});
        return;
    }
    const uint64_t tail = el.blob_bytes - el.gu_bytes - el.d_bytes;
    const BlobSpan parts[3] = {{el.offset + (uint64_t) e * el.gu_bytes, el.gu_bytes, nullptr},
                               {el.d_offset + (uint64_t) e * el.d_bytes, el.d_bytes, nullptr},
                               {0, tail, el.tails ? el.tails + (uint64_t) e * tail : nullptr}};
    uint64_t start = 0;
    for (const BlobSpan& s : parts) {
        const uint64_t lo = std::max(at, start), hi = std::min(at + n, start + s.bytes);
        if (lo < hi) out.push_back({s.mem ? 0 : s.off + (lo - start), hi - lo, s.mem ? s.mem + (lo - start) : nullptr});
        start += s.bytes;
    }
}

}  // namespace granite::models::inkling2
