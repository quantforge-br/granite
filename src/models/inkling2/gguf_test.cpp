// src/models/inkling2/gguf_test.cpp - llama.cpp's inkling2_loop GGUF read as the pack.
//
// No arguments: open_pack and take_drafter on synthetic GGUFs (no model, no GPU): the names, shapes, relative bias table,
// expert pieces and drafter keys they map, and the files they refuse.  A pack and the GGUF of the same weights: every dense
// tensor the pack holds, as the model loads it (Pack::source through the loader), and sampled expert blobs, byte for byte.
// --drafters PACK FILE...: each file's drafter (take_drafter) against the pack's own, as the model loads it, byte for byte.
// The runs on files skip when a file is missing.
#include "models/inkling2/pack.hpp"

#include "core/gguf_fixture.hpp"
#include "granite/core/loader.hpp"
#include "granite/platform/file.hpp"
#include "granite/platform/system.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace granite::models::inkling2;
namespace core = granite::core;

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++failures;
    }
}

constexpr uint32_t kF32 = 0, kQ8_0 = 8, kBF16 = 30, kNvfp4 = 40;
constexpr uint64_t kEmbd = 2304, kFf = 64, kExperts = 32, kRel = 16, kExtent = 1024, kVocab = 4, kPadded = 64;
const std::string kA = "inkling2_loop.";

std::vector<uint8_t> f32_bytes(const std::vector<float>& v) {
    std::vector<uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

struct Spec {
    std::vector<fixture::Kv> kv;
    std::vector<fixture::Tensor> ts;
};

// Two layers at the model's width: layer 0 with NVFP4 experts and their scales, layer 1 (the loop) with Q8_0 experts;
// four tokens, the embedding padded to 64 rows, the relative bias table holding its own index.
Spec synthetic() {
    using namespace fixture;
    Spec s;
    s.kv = {str("general.architecture", "inkling2_loop"), strs("tokenizer.ggml.tokens", {"a", "b", "c", "d"}),
            u32(kA + "block_count", 2), u32(kA + "embedding_length", kEmbd), i32s(kA + "attention.head_count", {24, 24}),
            i32s(kA + "attention.head_count_kv", {8, 8}), u32(kA + "attention.key_length", 128),
            u32(kA + "attention.sliding_window", 1024), i32s(kA + "attention.sliding_window_pattern", {1, 0}),
            u32(kA + "attention.relative_length", kRel), u32(kA + "attention.relative_extent", kExtent),
            u32(kA + "shortconv.l_cache", 4), u32(kA + "attn_res.block_size", 1), u32(kA + "expert_count", kExperts),
            u32(kA + "expert_used_count", 8), u32(kA + "expert_feed_forward_length", kFf),
            f32(kA + "activation.situ_beta", 4.f), f32(kA + "activation.situ_linear_beta", 25.f),
            u32(kA + "loop.start", 1), u32(kA + "loop.end", 2), u32(kA + "loop.count", 3), u32(kA + "loop.gate_rank", 64),
            boolean(kA + "loop.varying", false), u32(kA + "vocab_size_unpadded", kVocab)};
    std::vector<float> rel(kRel * kExtent), sg(kExperts), sd(kExperts);
    for (size_t i = 0; i < rel.size(); ++i) rel[i] = (float) i;
    for (uint64_t e = 0; e < kExperts; ++e) {
        sg[e] = 0.5f + (float) e;
        sd[e] = 100.f + (float) e;
    }
    s.ts = {{"token_embd.weight", {kEmbd, kPadded}, kQ8_0, 1, {}},
            {"blk.0.attn_sink_proj.weight", {kEmbd, 24}, kBF16, 2, {}},
            {"blk.0.attn_rel_bias.weight", {kRel, kExtent}, kF32, 3, f32_bytes(rel)},
            {"blk.0.ffn_gate_up_exps.weight", {kEmbd, 2 * kFf, kExperts}, kNvfp4, 4, {}},
            {"blk.0.ffn_down_exps.weight", {kFf, kEmbd, kExperts}, kNvfp4, 5, {}},
            {"blk.0.ffn_gate_up_exps.scale", {kExperts}, kF32, 6, f32_bytes(sg)},
            {"blk.0.ffn_down_exps.scale", {kExperts}, kF32, 7, f32_bytes(sd)},
            {"blk.1.ffn_gate_up_exps.weight", {kEmbd, 2 * kFf, kExperts}, kQ8_0, 8, {}},
            {"blk.1.ffn_down_exps.weight", {kFf, kEmbd, kExperts}, kQ8_0, 9, {}}};
    return s;
}

bool open(const std::string& path, const Spec& s, Pack& p, std::string& err, fixture::Written* w = nullptr) {
    const fixture::Written x = fixture::write(path, s.kv, s.ts);
    if (w) *w = x;
    return open_pack(path, p, err);
}

void synthetic_cases(const std::string& base) {
    const Spec s = synthetic();
    Pack p;
    std::string err;
    fixture::Written w;
    if (!open(base + ".gguf", s, p, err, &w)) {
        expect(false, "the synthetic GGUF opens: " + err);
        return;
    }
    auto at = [&](size_t i) { return w.data_start + w.offsets[i]; };   // tensor i's file offset

    expect(p.geo.n_head == 24 && p.geo.n_head_kv == 8, "the per-layer head counts read as one");
    expect(p.geo.gate_softcap == 4.f && p.geo.up_softcap == 25.f, "the softcaps read from the activation's betas");
    expect(p.geo.n_vocab == (int) kVocab && p.geo.image_token == -1 && !p.dspark.present, "the tokenizer's vocabulary, no image token, no drafter");

    expect(p.find("blk.0.attn_sink.weight") && !p.find("blk.0.attn_sink_proj.weight"), "attn_sink_proj reads as attn_sink");
    const TensorRef* emb = p.find("token_embd.weight");
    expect(emb && emb->shape == std::vector<uint64_t>{kEmbd, kVocab} && emb->bytes == kVocab * kEmbd / 32 * 34 && emb->offset == at(0),
           "the embedding's padding rows are cut");
    const TensorRef* rel = p.find("blk.0.attn_rel_bias.weight");
    bool transposed = rel && rel->mem && rel->shape == std::vector<uint64_t>{kExtent, kRel} && rel->bytes == kRel * kExtent * 4;
    for (uint64_t r = 0; transposed && r < kRel; ++r)
        for (uint64_t x = 0; transposed && x < kExtent; ++x) {
            float v;
            std::memcpy(&v, rel->mem + (r * kExtent + x) * 4, 4);
            transposed = v == (float) (x * kRel + r);
        }
    expect(transposed, "the relative bias table is held transposed");

    const uint64_t gu4 = 2 * kFf * (kEmbd / 64 * 36), d4 = kEmbd * (kFf / 64 * 36);
    const uint64_t gu8 = 2 * kFf * (kEmbd / 32 * 34), d8 = kEmbd * (kFf / 32 * 34);
    const ExpertLayer& e0 = p.experts[0];
    const ExpertLayer& e1 = p.experts[1];
    expect(e0.gu_type == (int) kNvfp4 && e0.d_type == (int) kNvfp4 && e0.gu_bytes == gu4 && e0.d_bytes == d4 &&
               e0.blob_bytes == gu4 + d4 + 16 && e0.offset == at(3) && e0.d_offset == at(4) && e0.tails,
           "layer 0: NVFP4 blobs with a tail");
    expect(e1.gu_type == (int) kQ8_0 && e1.d_type == (int) kQ8_0 && e1.blob_bytes == gu8 + d8 && !e1.tails, "layer 1: Q8_0 blobs without one");
    expect(!p.experts_contiguous(0) && !p.experts_contiguous(1), "a GGUF's blobs are not one range of the file");
    bool tails = true;
    for (uint64_t e = 0; e < kExperts; ++e) {
        float t[4];
        std::memcpy(t, e0.tails + e * 16, 16);
        tails = tails && t[0] == 0.5f + (float) e && t[1] == t[0] && t[2] == 100.f + (float) e && t[3] == 0.f;
    }
    expect(tails, "each tail is {s_gate_up, s_gate_up, s_down, 0}");

    const uint64_t e = 5;
    std::vector<BlobSpan> sp;
    p.expert_spans(0, (int) e, 0, e0.blob_bytes, sp);
    expect(sp.size() == 3 && sp[0].off == at(3) + e * gu4 && sp[0].bytes == gu4 && !sp[0].mem && sp[1].off == at(4) + e * d4 &&
               sp[1].bytes == d4 && !sp[1].mem && sp[2].mem == e0.tails + e * 16 && sp[2].bytes == 16,
           "a whole NVFP4 blob: its gate and up rows, its down rows, its tail");
    sp.clear();
    p.expert_spans(0, (int) e, gu4 - 10, 20, sp);
    expect(sp.size() == 2 && sp[0].off == at(3) + (e + 1) * gu4 - 10 && sp[0].bytes == 10 && sp[1].off == at(4) + e * d4 && sp[1].bytes == 10,
           "a range across the gate/up and the down tensors");
    sp.clear();
    p.expert_spans(0, (int) e, e0.blob_bytes - 8, 4, sp);
    expect(sp.size() == 1 && sp[0].mem == e0.tails + e * 16 + 8 && sp[0].bytes == 4, "a range inside the tail");
    sp.clear();
    p.expert_spans(1, (int) e, gu8 + d8 - 4, 4, sp);
    expect(sp.size() == 1 && sp[0].off == at(8) + (e + 1) * d8 - 4 && sp[0].bytes == 4, "a Q8_0 blob ends with its down rows");

    // the files it refuses, each saying why
    auto refuses = [&](const std::string& what, const Spec& bad, const char* says) {
        Pack q;
        std::string why;
        const bool opened = open(base + "_bad.gguf", bad, q, why);
        expect(!opened && why.find(says) != std::string::npos, what + " is refused (" + (opened ? "it opened" : why) + ")");
    };
    auto with = [&](const std::string& key, uint64_t u) {
        Spec b = s;
        for (fixture::Kv& k : b.kv)
            if (k.key == key) k.u = u;
        return b;
    };
    Spec b = s;
    b.kv[0].s = "llama";
    refuses("another architecture", b, "not llama.cpp's inkling2_loop");
    refuses("a loop with weights of each pass", with(kA + "loop.varying", 1), "loop.varying");
    refuses("a vocabulary the tokenizer does not hold", with(kA + "vocab_size_unpadded", 5), "vocab_size_unpadded");
    b = s;
    b.ts[2].shape = {kExtent, kRel};
    refuses("a relative bias table of another shape", b, "attn_rel_bias.weight is not F32 [16, 1024]");
    b = s;
    b.ts[8].type = kNvfp4;
    refuses("Q8_0 gate/up experts with NVFP4 down rows", b, "mix NVFP4");
    b = s;
    b.ts.push_back({"blk.1.ffn_down_exps.scale", {kExperts}, kF32, 10, {}});
    refuses("a scale on Q8_0 experts", b, "per-expert scales");
    b = s;
    b.ts.pop_back();
    refuses("a layer without its down experts", b, "no ffn_gate_up_exps and ffn_down_exps");
    b = s;
    b.ts[3].shape = {kEmbd, kFf, kExperts};
    refuses("gate/up experts of half the rows", b, "expert tensors are not");
    std::remove((base + ".gguf").c_str());
    std::remove((base + "_bad.gguf").c_str());
}

// A dflash drafter in the layout of Unsloth's Inkling 2 DSpark files, at toy sizes (take_drafter maps names and keys; the
// model checks the sizes): one layer's tensors, the Markov head's token rows padded to 64.
Spec dflash() {
    using namespace fixture;
    const std::string d = "dflash.";
    Spec s;
    s.kv = {str("general.architecture", "dflash"), u32(d + "embedding_length", kEmbd), u32(d + "vocab_size_unpadded", kVocab),
            u32(d + "block_count", 3), u32(d + "block_size", 8), i32s(d + "target_layers", {5, 23, 41, 48}),
            i32s(d + "attention.sliding_window_pattern", {1, 0, 1}), u32(d + "attention.sliding_window", 1024),
            u32(d + "attention.head_count", 24), u32(d + "attention.head_count_kv", 8), u32(d + "attention.key_length", 128),
            u32(d + "feed_forward_length", 9216), f32(d + "rope.freq_base", 1e6f), f32(d + "attention.layer_norm_rms_epsilon", 1e-6f),
            u32(d + "dspark.markov_rank", 256)};
    s.ts = {{"dspark_fuse.weight", {4, 4}, kF32, 1, {}},
            {"markov_w1.weight", {256, kPadded}, kBF16, 2, {}},
            {"blk.0.attn_q.weight", {64, 2}, kQ8_0, 3, {}},
            {"blk.0.attn_k.weight", {64, 2}, kBF16, 4, {}},
            {"blk.0.ffn_norm.weight", {64}, kF32, 5, {}},
            {"dspark_norm.weight", {64}, kF32, 6, {}},
            {"dspark_hidden_out.weight", {64, 2}, kBF16, 7, {}},
            {"joint_proj.bias", {64}, kF32, 8, {}},
            {"mask_embd.weight", {64}, kF32, 9, {}},
            {"output_norm.weight", {64}, kF32, 10, {}}};
    return s;
}

void drafter_cases(const std::string& base) {
    Pack target;
    std::string err;
    if (!open(base + ".gguf", synthetic(), target, err)) {
        expect(false, "the synthetic GGUF opens: " + err);
        return;
    }
    const Spec s = dflash();
    const fixture::Written w = fixture::write(base + "_dflash.gguf", s.kv, s.ts);
    auto at = [&](size_t i) { return w.data_start + w.offsets[i]; };
    Pack p = target;
    if (!take_drafter(p, base + "_dflash.gguf", err)) {
        expect(false, "the synthetic drafter is taken: " + err);
        return;
    }
    const DsparkGeometry& d = p.dspark;
    expect(d.present && d.n_layers == 3 && d.block == 8 && d.targets == std::vector<int>{5, 23, 41, 48} &&
               d.swa == std::vector<bool>{true, false, true} && d.window == 1024 && d.n_head == 24 && d.n_head_kv == 8 &&
               d.head_dim == 128 && d.n_ff == 9216 && d.rope_base == 1e6f && d.rms_eps == 1e-6f && d.markov_rank == 256,
           "the drafter's settings read from the dflash keys");
    expect(p.drafter_path == base + "_dflash.gguf" && p.find("token_embd.weight"), "the drafter's file noted, the target's tensors kept");
    const TensorRef* fuse = p.find("dspark.fuse_weight");
    const TensorRef* w1 = p.find("dspark.markov.w1.weight");
    expect(fuse && fuse->offset == at(0), "dspark_fuse reads as dspark.fuse_weight");
    expect(w1 && w1->shape == std::vector<uint64_t>{256, kVocab} && w1->bytes == 256 * kVocab * 2 && w1->offset == at(1),
           "the Markov head's padding rows are cut");
    expect(p.find("dspark.blk.0.post_attention_norm.weight") && !p.find("dspark.blk.0.ffn_norm.weight"),
           "a layer's ffn_norm reads as post_attention_norm");
    expect(p.find("dspark.markov.dflare_norm.weight") && p.find("dspark.markov.out.weight") && p.find("dspark.markov.rnn.bias") &&
               p.find("dspark.mask_embed") && p.find("dspark.norm.weight"),
           "the Markov head's, the mask's and the final norm's names");
    core::LoadItem q, k;
    expect(p.source("dspark.blk.0.attn_q.weight", Form::Q8_0, 128, 0, q, err) && q.conv == core::Conv::Copy &&
               p.source("dspark.blk.0.attn_k.weight", Form::Q8_0, 128, 0, k, err) && k.conv == core::Conv::Bf16ToQ8_0,
           "Q8_0 projections load as they are, BF16 ones rounded");

    auto refuses = [&](const std::string& what, const Spec& bad, const char* says) {
        fixture::write(base + "_bad.gguf", bad.kv, bad.ts);
        Pack q = target;
        std::string why;
        const bool taken = take_drafter(q, base + "_bad.gguf", why);
        expect(!taken && why.find(says) != std::string::npos, what + " is refused (" + (taken ? "it was taken" : why) + ")");
    };
    Spec b = s;
    b.kv[0].s = "inkling2_loop";
    refuses("a GGUF of the target's architecture", b, "not a dflash drafter");
    b = s;
    b.kv[2].u = kVocab + 1;
    refuses("a drafter over another vocabulary", b, "over 5 tokens");
    b = s;
    b.ts.push_back({"markov_w2.weight", {256, kVocab}, kBF16, 11, {}});
    refuses("llama.cpp's later DSpark layout", b, "markov_w2");
    b = s;
    b.ts.push_back({"blk.0.ctx_mix.weight", {4}, kF32, 11, {}});
    refuses("a tensor the drafter does not have", b, "blk.0.ctx_mix.weight");
    std::remove((base + ".gguf").c_str());
    std::remove((base + "_dflash.gguf").c_str());
    std::remove((base + "_bad.gguf").c_str());
}

// ---- a pack and a GGUF of the same weights

// Bytes [at, at + n) of expert e's blob in layer l, through the pieces the loads read.
bool blob(const Pack& p, int l, int e, uint64_t at, uint64_t n, std::vector<uint8_t>& out, std::string& err) {
    std::vector<BlobSpan> sp;
    p.expert_spans(l, e, at, n, sp);
    out.assign(n, 0);
    uint64_t done = 0;
    for (const BlobSpan& s : sp) {
        if (done + s.bytes > n) break;
        if (s.mem) std::memcpy(out.data() + done, s.mem, s.bytes);
        else if (!core::read_range_at(core::FileRange(p.file.path()), s.off, out.data() + done, s.bytes, err)) return false;
        done += s.bytes;
    }
    if (done != n) err = "the pieces of a blob range cover " + std::to_string(done) + " of its " + std::to_string(n) + " bytes";
    return done == n;
}

struct Load {
    std::string name;
    Form form = Form::F32;
    uint64_t values = 0;
    std::vector<uint8_t> out[2];   ///< as the pack, as the GGUF loads it
};

uint64_t form_bytes(Form f, uint64_t values) { return f == Form::F32 ? values * 4 : f == Form::Bf16 ? values * 2 : values / 32 * 34; }

// Loads `batch` from `p` into host memory, as the model sources it: the drafter's tensors from its file.
bool load(const Pack& p, int which, std::vector<Load>& batch, std::string& err) {
    core::LoadPlan plan;
    const int file = plan.file(p.file.path());
    const int drafter = p.drafter_path.empty() ? file : plan.file(p.drafter_path);
    for (Load& x : batch) {
        x.out[which].assign(form_bytes(x.form, x.values), 0);
        core::LoadItem it;
        if (!p.source(x.name, x.form, x.values, x.name.rfind("dspark.", 0) == 0 ? drafter : file, it, err)) return false;
        it.host = x.out[which].data();
        plan.add(it);
    }
    core::Loader loader(8);
    return loader.run(plan, err);
}

int parity(const std::string& pack_path, const std::string& gguf_path) {
    Pack a, b;
    std::string err;
    if (!open_pack(pack_path, a, err) || !open_pack(gguf_path, b, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (a.file.gguf() || !b.file.gguf()) {
        std::fprintf(stderr, "usage: the pack first, then the GGUF\n");
        return 2;
    }
    const Geometry& x = a.geo;
    const Geometry& y = b.geo;
#define SAME(f) expect(x.f == y.f, "the geometry's " #f)
    SAME(n_layers); SAME(n_embd); SAME(n_vocab); SAME(n_head); SAME(n_head_kv); SAME(head_dim); SAME(window); SAME(rel);
    SAME(extent); SAME(conv); SAME(block); SAME(qk_clip); SAME(rel_clip); SAME(log_floor); SAME(log_alpha); SAME(rms_eps);
    SAME(n_expert); SAME(n_used); SAME(n_ff_exp); SAME(expert_scale); SAME(gate_softcap); SAME(up_softcap);
    SAME(logit_softcap); SAME(loop_start); SAME(loop_end); SAME(loop_passes); SAME(gate_rank); SAME(context_length); SAME(swa);
#undef SAME

    // every dense tensor the pack holds (the drafter's aside: the GGUF has none) as the model loads it: F32 as it is, BF16
    // controllers as BF16, the rest as Q8_0.  The model reads the pack's BF16 embedding as it is; the GGUF's is those rows
    // rounded to Q8_0 as the loader rounds them, which is what is compared.
    std::vector<Load> batch;
    uint64_t held = 0, bytes = 0;
    size_t dense = 0;
    auto flush = [&]() {
        if (batch.empty()) return;
        std::string e;
        if (!load(a, 0, batch, e) || !load(b, 1, batch, e)) expect(false, "loading the dense tensors: " + e);
        else
            for (const Load& l : batch) expect(l.out[0] == l.out[1], l.name + " loads to the same bytes");
        batch.clear();
        held = 0;
    };
    for (const auto& [name, t] : a.tensors) {
        if (name.rfind("dspark.", 0) == 0) continue;
        const TensorRef* u = b.find(name);
        if (!u) {
            expect(false, name + " is in the GGUF");
            continue;
        }
        expect(u->shape == t.shape, name + " has the pack's shape");
        uint64_t values = 1;
        for (uint64_t d : t.shape) values *= d;
        const Form form = t.type == (int) kF32 ? Form::F32
                          : t.type == (int) kBF16 && name != "token_embd.weight" && name != "output.weight" ? Form::Bf16
                                                                                                          : Form::Q8_0;
        batch.push_back({name, form, values, {}});
        held += form_bytes(form, values);
        bytes += form_bytes(form, values);
        ++dense;
        if (held > (512ull << 20)) flush();
    }
    flush();

    // each layer's first, second, middle and last experts: the whole blob, and ranges across the GGUF's pieces
    int blobs = 0;
    const int ne = x.n_expert;
    for (int l = 0; l < x.n_layers; ++l) {
        const ExpertLayer& p = a.experts[(size_t) l];
        const ExpertLayer& q = b.experts[(size_t) l];
        const std::string L = "layer " + std::to_string(l);
        expect(p.gu_type == q.gu_type && p.d_type == q.d_type && p.blob_bytes == q.blob_bytes, L + "'s expert types and blob bytes");
        if (p.blob_bytes != q.blob_bytes) continue;
        for (int e : {0, 1, ne / 2, ne - 1}) {
            std::vector<uint8_t> u, v, part;
            if (!blob(a, l, e, 0, p.blob_bytes, u, err) || !blob(b, l, e, 0, q.blob_bytes, v, err)) {
                expect(false, L + ": " + err);
                continue;
            }
            const std::string E = L + " expert " + std::to_string(e);
            expect(u == v, E + "'s blob");
            ++blobs;
            const std::pair<uint64_t, uint64_t> ranges[] = {{q.gu_bytes - 64, 128}, {q.gu_bytes / 2 - 3, q.gu_bytes / 2 + 70},
                                                            {q.blob_bytes - 40, 40}};
            for (const auto& [at, n] : ranges) {
                if (!blob(b, l, e, at, n, part, err)) {
                    expect(false, E + ": " + err);
                    continue;
                }
                expect(std::equal(part.begin(), part.end(), u.begin() + (ptrdiff_t) at),
                       E + "'s bytes [" + std::to_string(at) + ", " + std::to_string(at + n) + ")");
            }
        }
    }
    std::printf("inkling2 gguf parity: %zu dense tensors (%.2f GiB as loaded) and %d expert blobs, %s\n", dense,
                (double) bytes / (double) (1ull << 30), blobs, failures ? "FAILED" : "the same bytes");
    return failures ? 1 : 0;
}

// Each file's drafter in place of the pack's own: its settings, and every drafter tensor as the model loads it (F32 as it is,
// the Markov head's BF16 rows as BF16, the layers' projections as Q8_0).
int drafter_parity(const std::string& pack_path, const std::vector<std::string>& files) {
    Pack a;
    std::string err;
    if (!open_pack(pack_path, a, err) || !a.dspark.present) {
        std::fprintf(stderr, "%s\n", err.empty() ? "the pack carries no drafter" : err.c_str());
        return 1;
    }
    for (const std::string& path : files) {
        Pack b;
        if (!open_pack(pack_path, b, err) || !take_drafter(b, path, err)) {
            expect(false, path + ": " + err);
            continue;
        }
        const DsparkGeometry& x = a.dspark;
        const DsparkGeometry& y = b.dspark;
        expect(x.n_layers == y.n_layers && x.block == y.block && x.n_head == y.n_head && x.n_head_kv == y.n_head_kv &&
                   x.head_dim == y.head_dim && x.n_ff == y.n_ff && x.window == y.window && x.markov_rank == y.markov_rank &&
                   x.rope_base == y.rope_base && x.rms_eps == y.rms_eps && x.targets == y.targets && x.swa == y.swa,
               path + ": the drafter's settings");
        std::vector<Load> batch;
        size_t n = 0, theirs = 0;
        for (const auto& [name, t] : b.tensors) theirs += name.rfind("dspark.", 0) == 0;
        for (const auto& [name, t] : a.tensors) {
            if (name.rfind("dspark.", 0) != 0) continue;
            uint64_t values = 1;
            for (uint64_t dd : t.shape) values *= dd;
            const Form form = t.type == (int) kF32 ? Form::F32 : name.rfind("dspark.blk.", 0) == 0 ? Form::Q8_0 : Form::Bf16;
            batch.push_back({name, form, values, {}});
            ++n;
        }
        expect(n == theirs, path + ": " + std::to_string(theirs) + " drafter tensors, the pack's " + std::to_string(n));
        std::string e;
        if (!load(a, 0, batch, e) || !load(b, 1, batch, e)) expect(false, path + ": loading the drafter: " + e);
        else
            for (const Load& l : batch) expect(l.out[0] == l.out[1], path + ": " + l.name + " loads to the pack's bytes");
        std::printf("inkling2 drafter parity: %s, %zu tensors\n", path.c_str(), n);
    }
    std::printf("inkling2 drafter parity: %s\n", failures ? "FAILED" : "the pack's bytes");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        const std::string base =
            granite::platform::temp_dir() + "/inkling2_gguf_test_" + std::to_string((long long) granite::platform::process_id());
        synthetic_cases(base);
        drafter_cases(base);
        std::printf("inkling2 gguf: %s\n", failures ? "FAILED" : "all passed");
        return failures ? 1 : 0;
    }
    const bool drafters = std::string(argv[1]) == "--drafters";
    if (drafters ? argc < 4 : argc != 3) {
        std::fprintf(stderr, "usage: inkling2_gguf_test [<pack> <gguf> | --drafters <pack> <file>...]\n");
        return 2;
    }
    for (int i = drafters ? 2 : 1; i < argc; ++i)
        if (!std::ifstream(argv[i]).good()) {
            std::printf("no model at %s: skipped\n", argv[i]);
            return 77;
        }
    if (drafters) return drafter_parity(argv[2], std::vector<std::string>(argv + 3, argv + argc));
    return parity(argv[1], argv[2]);
}
