// src/models/inkling2/prompt.cpp - the prompt path of an Inkling 2 session. A read of more rows than a decode window runs in
// chunks of up to kChunk rows, each one forward through the visits of walk.hpp over chunk-wide row buffers: the Q8_0
// projections and the experts are MMQ products on the int8 tensor cores (the experts grouped by expert and read in place),
// the router's scores and the loop controllers' products run on the tf32 tensor cores, the attention flash style over the
// caches (kernels/attn_prompt.cu), the rest are the decode forward's kernels over all the rows. A chunk leaves the caches as
// the decode forward leaves them: a full visit's cells, a sliding-window visit's ring (the chunk's keys are staged, then
// its last positions moved in) and the convolution rings' last kConvRing positions.
//
// Tensor parallelism (model.hpp cards()): each card runs every visit over the chunk with its half of the experts; each
// writes its experts' sum into the other's receive buffer as it writes its own, and adds the two (IEEE addition commutes,
// so both cards hold the same bits).
#include "models/inkling2/prompt.hpp"

#include "granite/kernels/dense_ops.hpp"
#include "granite/kernels/elementwise.hpp"
#include "granite/kernels/iq_kernels.hpp"
#include "granite/kernels/router.hpp"
#include "granite/kernels/row_norm.hpp"
#include "granite/models/inkling2/kernels.hpp"
#include "granite/models/inkling2/prompt_kernels.hpp"
#include "granite/prefill/moe_mmq.hpp"
#include "models/inkling2/dspark.hpp"
#include "models/inkling2/walk.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace granite::models::inkling2 {
namespace {

namespace bk = granite::kernels;
namespace mmq = granite::prefill::mmq;

constexpr int kChunk = kPromptChunk;       // rows of one chunk forward
constexpr int kTypeNvfp4 = 40;
constexpr int kOut = kHeads * kHeadDim;    // the attention output's width
constexpr uint64_t kMmqSpare = 32ull << 20;   // MMQ's stream-k fixup pool

// One allocation carved into the chunk's buffers; with no base it only counts, so the sizing and the allocation agree.
struct Carve {
    uint8_t* base = nullptr;
    size_t at = 0;
    template <class T> T* take(size_t n) {
        at = (at + 255) & ~(size_t) 255;
        T* p = base ? reinterpret_cast<T*>(base + at) : nullptr;
        at += n * sizeof(T);
        return p;
    }
};

// The chunk-wide buffers. The arena holds the attention's buffers, then (over them, once they are spent) the experts':
// msg, mix, pair, hid and nrm keep their places into the MLP's input, the experts' rows start at the arena's base.
struct Bufs {
    float* role[kBufs] = {};
    float* resid = nullptr;
    float *msg = nullptr, *mix = nullptr, *hid = nullptr, *nrm = nullptr;
    float *yq = nullptr, *yk = nullptr, *yv = nullptr, *q = nullptr, *r = nullptr, *sink = nullptr, *attn = nullptr, *o = nullptr;
    void* xq = nullptr;          // the experts' q8_1 rows (their input, then H)
    float *gu = nullptr, *h = nullptr, *moe = nullptr;   // gu also holds the experts' outputs, h the sum
    uint16_t* p2p_loc = nullptr;                         // TP2 symmetric FP16 local buffer (moe)
    float* p2p_loc_sc = nullptr;                         // TP2 per-row power-of-2 scale (local moe)
    float* recv_attn_f32[2] = {};                        // TP2 exact FP32 peer receive buffers (attn out), by parity
    uint16_t* recv_f16[2] = {};                          // TP2 symmetric FP16 peer receive buffers (moe), by parity
    float* recv_sc[2] = {};                              // TP2 per-row power-of-2 scale (moe), by parity
    void* xqd = nullptr;         // a dense product's q8_1 rows
    float* rlog = nullptr;
    float* rw = nullptr;
    int32_t *ids = nullptr, *bounds = nullptr, *tok = nullptr, *dst = nullptr, *exp = nullptr;
    void* sort_scr = nullptr;
    void *k_stage = nullptr, *v_stage = nullptr;
    uint16_t *ks_stage = nullptr, *vs_stage = nullptr;
    int32_t *pos = nullptr, *tokd = nullptr, *ident = nullptr, *dbounds = nullptr;
};

// `F`: the card's experts' width; `tp`: with the receive buffers.
void carve(Carve& c, Bufs& b, const Geometry& g, int64_t F, bool tp) {
    const size_t E = (size_t) g.n_embd, R = kChunk, P = R * (size_t) g.n_used;
    for (float*& p : b.role) p = c.take<float>(R * E);
    b.resid = c.take<float>(R * E);
    // In TP2, keep msg and o outside the shared attention/experts arena so 2-micro-slice overlap can run slice 0's
    // experts while slice 1's msg[3] and DMA copy of B.o remain live (0 extra bytes: arena drops from 636 to 456 MiB > 437 MiB).
    if (tp) {
        b.msg = c.take<float>(4 * R * E);
        b.o = c.take<float>(R * E);
    }
    // the attention's buffers
    const size_t arena = c.at;
    if (!tp) b.msg = c.take<float>(4 * R * E);
    b.mix = c.take<float>(3 * R * E);
    b.hid = c.take<float>(R * (size_t) g.gate_rank);
    b.nrm = c.take<float>(3 * R * E);
    b.yq = c.take<float>(R * kProjQ);
    b.yk = c.take<float>(R * kProjKV);
    b.yv = c.take<float>(R * kProjKV);
    b.q = c.take<float>(R * kOut);
    b.r = c.take<float>(R * kHeads * kRel);
    b.sink = c.take<float>(R * kHeads);
    b.attn = c.take<float>(R * kOut);
    if (!tp) b.o = c.take<float>(R * E);
    const size_t end_a = c.at;
    // the experts' buffers, from the arena's base: the MLP's input (mix, its controller, nrm) is read before any is written.
    // gu holds the pairs' gate and up rows [P][2F], then their down rows [P][E]; h their SwiGLU [P][F], then the sum [R][E]
    Carve x = c;
    x.at = arena;
    b.xq = x.take<uint8_t>(mmq::q8_bytes((int64_t) P, (int64_t) E));
    b.gu = x.take<float>(P * std::max<size_t>(2 * (size_t) F, E));
    b.h = x.take<float>(std::max(P * (size_t) F, R * E));
    b.moe = b.h;
    // the experts' input rows are rounded from nrm while it is read, so they end before it
    if (c.base && (uint8_t*) b.xq + mmq::q8_bytes((int64_t) P, (int64_t) E) > (uint8_t*) b.nrm)
        throw std::logic_error("inkling2 prompt: the experts' input rows overlap the MLP's normed rows");
    c.at = std::max(end_a, x.at);
    if (tp) {
        b.p2p_loc = c.take<uint16_t>(R * E);
        b.p2p_loc_sc = c.take<float>(R);
        for (float*& p : b.recv_attn_f32) p = c.take<float>(R * E);
        for (uint16_t*& p : b.recv_f16) p = c.take<uint16_t>(R * E);
        for (float*& p : b.recv_sc) p = c.take<float>(R);
    }
    b.xqd = c.take<uint8_t>(mmq::q8_bytes((int64_t) R, std::max<int64_t>((int64_t) E, kOut)));
    b.rlog = c.take<float>(R * (size_t) g.n_expert);
    b.rw = c.take<float>(P);
    b.ids = c.take<int32_t>(P);
    b.bounds = c.take<int32_t>((size_t) g.n_expert + 1);
    b.tok = c.take<int32_t>(P);
    b.dst = c.take<int32_t>(P);
    b.exp = c.take<int32_t>(P);
    b.sort_scr = c.take<uint8_t>(moe_sort_scratch((int) P, g.n_expert));
    const size_t kv_h = tp ? (size_t) (kKvHeads / 2) : (size_t) kKvHeads;
    const size_t stage = kv_h * R * kHeadDim * 2;   // fp16 at most
    b.k_stage = c.take<uint8_t>(stage);
    b.v_stage = c.take<uint8_t>(stage);
    const size_t stage_sc = kv_h * R * (kHeadDim / 64);
    b.ks_stage = c.take<uint16_t>(stage_sc);
    b.vs_stage = c.take<uint16_t>(stage_sc);
    b.pos = c.take<int32_t>(R);
    b.tokd = c.take<int32_t>(R);
    b.ident = c.take<int32_t>(P);
    b.dbounds = c.take<int32_t>(6);   // [0..1]: full rows, [2..3]: slice 0, [4..5]: slice 1
}

// A card's side of the prompt path: its chunk buffers, products and stream (the reading session's), the reading session's
// caches there, and the P2P copy stream + events (tensor parallelism).
struct Lane {
    int device = 0;
    Bufs b;
    void* mem = nullptr;
    std::unique_ptr<mmq::Context> mmq;
    cudaStream_t stream = nullptr;
    cudaStream_t copy_st = nullptr;
    const std::vector<VisitCache>* vc = nullptr;
    cudaEvent_t attn_ready[2] = {};
    cudaEvent_t sent_attn[2] = {};
    cudaEvent_t recv_attn_free[2] = {};
    cudaEvent_t moe_ready[2] = {};
    cudaEvent_t sent[2] = {};
    cudaEvent_t recv_moe_free[2] = {};
};

// The current device for a scope.
struct OnCard {
    int prev = 0;
    explicit OnCard(int dev) {
        cudaGetDevice(&prev);
        if (dev != prev) cudaSetDevice(dev);
    }
    ~OnCard() {
        int now = 0;
        cudaGetDevice(&now);
        if (now != prev) cudaSetDevice(prev);
    }
};

class Prompt final : public PromptRunner {
public:
    Prompt(InklingModel& m, void* const streams[2]) : m_(m), g_(m.geo()), nc_(m.cards()) {
        for (int c = 0; c < nc_; ++c) {
            lane_[c].device = m.card_device(c);
            lane_[c].stream = (cudaStream_t) streams[c];
        }
    }
    ~Prompt() override {   // every read synced its streams before it returned
        OnCard on(lane_[0].device);
        for (cudaEvent_t& e : done_)
            if (e) cudaEventDestroy(e);
        if (fork_) cudaEventDestroy(fork_);
        for (int c = 0; c < nc_; ++c) {
            cudaSetDevice(lane_[c].device);
            for (int i = 0; i < 2; ++i) {
                if (lane_[c].attn_ready[i]) cudaEventDestroy(lane_[c].attn_ready[i]);
                if (lane_[c].sent_attn[i]) cudaEventDestroy(lane_[c].sent_attn[i]);
                if (lane_[c].recv_attn_free[i]) cudaEventDestroy(lane_[c].recv_attn_free[i]);
                if (lane_[c].moe_ready[i]) cudaEventDestroy(lane_[c].moe_ready[i]);
                if (lane_[c].sent[i]) cudaEventDestroy(lane_[c].sent[i]);
                if (lane_[c].recv_moe_free[i]) cudaEventDestroy(lane_[c].recv_moe_free[i]);
            }
            if (lane_[c].copy_st) cudaStreamDestroy(lane_[c].copy_st);
            lane_[c].mmq.reset();
            if (lane_[c].mem) cudaFree(lane_[c].mem);
        }
    }

    bool init(std::string& err) {
        if (!walk(g_, walk_, err)) return false;
        OnCard on(lane_[0].device);
        for (cudaEvent_t& e : done_)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                err = "cannot create the prompt path's events";
                return false;
            }
        if (cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming) != cudaSuccess) {
            err = "cannot create the prompt path's events";
            return false;
        }
        for (int c = 0; c < nc_; ++c)
            if (!init_lane(c, err)) return false;
        return true;
    }

    // Card c's chunk buffers and products.
    bool init_lane(int card, std::string& err) {
        Lane& ln = lane_[card];
        OnCard on(ln.device);
        const int E = g_.n_embd, F = (int) m_.layer(0, card).n_ff;
        Carve c;
        carve(c, ln.b, g_, F, nc_ > 1);
        bool ok = cudaMalloc(&ln.mem, c.at) == cudaSuccess;
        if (ok && nc_ > 1) {
            ok = cudaStreamCreateWithFlags(&ln.copy_st, cudaStreamNonBlocking) == cudaSuccess;
            for (int i = 0; i < 2 && ok; ++i) {
                ok = cudaEventCreateWithFlags(&ln.attn_ready[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&ln.sent_attn[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&ln.recv_attn_free[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&ln.moe_ready[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&ln.sent[i], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&ln.recv_moe_free[i], cudaEventDisableTiming) == cudaSuccess;
            }
        }
        if (!ok) {
            cudaGetLastError();
            err = "out of device memory for the prompt path (" + std::to_string(c.at >> 20) + " MiB on CUDA" + std::to_string(ln.device) + ")";
            return false;
        }
        c = Carve{};
        c.base = static_cast<uint8_t*>(ln.mem);
        carve(c, ln.b, g_, F, nc_ > 1);
        ln.mmq = std::make_unique<mmq::Context>();
        mmq::iota(ln.b.ident, (int64_t) kChunk * g_.n_used, ln.stream);
        // the products' tiles (each card raises its own kernels' shared memory)
        const auto ready = [&](int type, int64_t rows) {
            if (!mmq::fits(type, rows)) {
                err = "the prompt path's MMQ has no tile for type " + std::to_string(type) + " at " + std::to_string(rows) + " rows";
                return false;
            }
            mmq::load(type, rows);
            return true;
        };
        if (!ready(8, kProj) || !ready(8, kProjQ) || !ready(8, kProjKV) || !ready(8, 2 * kProjKV) ||
            !ready(8, (kHeads / 2) * kHeadDim) || !ready(8, kHeads * kRel) || !ready(8, kHeads) ||
            !ready(8, kKvHeads * kHeadDim) || !ready(8, E))
            return false;
        for (int l = 0; l < g_.n_layers; ++l) {
            const LayerWeights& w = m_.layer(l, card);
            if (!ready(w.gu_type, 2 * w.n_ff) || !ready(w.d_type, E)) return false;
        }
        if (cudaStreamSynchronize(ln.stream) != cudaSuccess) {
            err = "the prompt path's setup failed";
            return false;
        }
        return true;
    }

    model::Result read(const model::PrefillRequest& r, std::vector<int32_t>& hist, const PromptTarget& t) override {
        bind(t);
        return run(r.tokens, nullptr, r.n_tokens, r.pos, r, hist, 0);
    }

    model::Result read_rows(const float* rows, int n, int64_t pos, int32_t pad, std::vector<int32_t>& hist,
                            const PromptTarget& t) override {
        bind(t);
        model::PrefillRequest r;
        r.n_tokens = n;
        r.pos = pos;
        return run(nullptr, rows, n, pos, r, hist, pad);
    }

private:
    Bufs& b(int c) { return lane_[c].b; }
    cudaStream_t strm(int c) const { return lane_[c].stream; }
    float* buf(int c, Buf x) { return lane_[c].b.role[(int) x]; }
    void bind(const PromptTarget& t) {
        ds_ = t.dspark;
        for (int c = 0; c < nc_; ++c) {
            lane_[c].vc = t.caches[c];
            lane_[c].stream = (cudaStream_t) t.stream[c];
        }
    }

    // `n` rows (tokens, or host embedding rows) in chunks of about equal size; one chunk stays queued ahead of the host,
    // which reports a chunk once the card has run it.
    model::Result run(const int32_t* tokens, const float* emb, int n, int64_t pos, const model::PrefillRequest& r,
                      std::vector<int32_t>& hist, int32_t pad) {
        OnCard on(lane_[0].device);
        if (nc_ > 1) {   // the second card's stream after what the first's has queued (a decode graph runs both cards there)
            cudaEventRecord(fork_, strm(0));
            cudaStreamWaitEvent(strm(1), fork_, 0);
        }
        const int chunks = (n + kChunk - 1) / kChunk;
        const int per = (n + chunks - 1) / chunks;
        std::string err;
        int reported = 0, k = 0;
        for (int t0 = 0; t0 < n; t0 += per, ++k) {
            if (r.stop && r.stop()) break;
            const int rows = std::min(per, n - t0);
            if (!chunk(tokens ? tokens + t0 : nullptr, emb ? emb + (size_t) t0 * g_.n_embd : nullptr, rows, pos + t0, err))
                return {model::Status::Error, "inkling2: " + err};
            ds_->note(pos + t0, rows);
            if (tokens) hist.insert(hist.end(), tokens + t0, tokens + t0 + rows);
            else hist.insert(hist.end(), (size_t) rows, pad);
            cudaEventRecord(done_[k % 2], strm(0));
            if (k > 0) {   // the chunk before this one has run
                cudaEventSynchronize(done_[(k - 1) % 2]);
                if (r.progress) r.progress(reported);
            }
            reported = t0 + rows;
        }
        bool ok = true;
        for (int c = 0; c < nc_; ++c) {
            ok = cudaStreamSynchronize(strm(c)) == cudaSuccess && ok;
            if (lane_[c].copy_st) ok = cudaStreamSynchronize(lane_[c].copy_st) == cudaSuccess && ok;
        }
        if (!ok) return {model::Status::Error, "inkling2: the prompt failed"};
        if (r.progress && reported > 0) r.progress(reported);
        return {};
    }

    // One chunk of `rows` rows at pos0..: the embedding, the visits, the caches.
    bool chunk(const int32_t* tokens, const float* emb, int rows, int64_t pos0, std::string& err) {
        const int E = g_.n_embd;
        const bool overlap = nc_ > 1 && rows >= 1024;
        const int r0 = overlap ? ((rows / 2) & ~15) : rows;
        const int r1 = rows - r0;
        const int off[2] = {0, r0};
        const int cnt[2] = {r0, r1};
        for (int c = 0; c < nc_; ++c) {
            OnCard on(lane_[c].device);
            prompt_rows(b(c).pos, pos0, rows, b(c).dbounds, strm(c));
            if (overlap) {
                prompt_rows(b(c).pos, pos0, r0, b(c).dbounds + 2, strm(c));
                prompt_rows(b(c).pos + r0, pos0 + r0, r1, b(c).dbounds + 4, strm(c));
            }
            float* x0 = buf(c, Buf::X0);
            if (emb) {
                cudaMemcpyAsync(x0, emb, (size_t) rows * E * 4, cudaMemcpyHostToDevice, strm(c));
            } else {
                cudaMemcpyAsync(b(c).tokd, tokens, (size_t) rows * 4, cudaMemcpyHostToDevice, strm(c));
                bk::iq_embed_rows(m_.embed_type(), m_.embedding(), m_.embed_row_bytes(), b(c).tokd, rows, E, b(c).mix, strm(c));
                bk::rms_norm_rows(b(c).mix, m_.embed_norm(c), rows, E, g_.rms_eps, x0, strm(c));
            }
        }
        if (overlap) {
            // TP2 2-micro-slice compute/DMA P2P overlap: each visit's attention B.o FP32 DMA and MoE FP16 DMA run on
            // the hardware copy engine (copy_st) concurrently with the other micro-slice's GPU compute. Causal KV
            // caches, sliding-window stage_to_ring, and post-conv rings always execute slice 0 before slice 1.
            bool has_pending = false;
            const Step* prev_s = nullptr;
            int prev_parity = 0;
            const auto flush_pending = [&]() {
                if (!has_pending) return;
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    finish(c, *prev_s, off[1], cnt[1], prev_parity, 1);
                    cudaEventRecord(lane_[c].recv_moe_free[prev_parity], strm(c));
                }
                has_pending = false;
            };
            for (const Step& s : walk_.steps) {
                if (const int k = ds_->slot_of_visit(s.visit); k >= 0) {
                    flush_pending();   // all rows of buf(0, s.hid) must be complete before DSpark tap
                    cudaMemcpyAsync(ds_->tap(k), buf(0, s.hid), (size_t) rows * E * 4, cudaMemcpyDeviceToDevice, strm(0));
                }
                const int parity = (int) (exchanges_++ & 1);
                // 1. Slice 0 attention + start Slice 0 B.o DMA on copy_st
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_attn(c, s, off[0], cnt[0], pos0 + off[0], parity, 1, 0, true);
                }
                // 2. Finish previous step's Slice 1 MoE (overlaps with Slice 0 B.o DMA; completes buf(prev_s->out)[r0..rows))
                flush_pending();
                // 3. Slice 1 attention + start Slice 1 B.o DMA (overlaps with Slice 0 B.o DMA)
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_attn(c, s, off[1], cnt[1], pos0 + off[1], parity, 2, 1, true);
                }
                // 4. Slice 0 MLP + experts + start Slice 0 MoE DMA (overlaps with Slice 1 B.o DMA)
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_mlp(c, s, off[0], cnt[0], parity, 1, 0, true);
                }
                // 5. Slice 1 MLP + experts + start Slice 1 MoE DMA (overlaps with Slice 0 MoE DMA)
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_mlp(c, s, off[1], cnt[1], parity, 2, 1, true);
                }
                // 6. Slice 0 finish (overlaps with Slice 1 MoE DMA); Slice 1 finish defers into next step's Slice 0 attn
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    finish(c, s, off[0], cnt[0], parity, 0);
                }
                has_pending = true;
                prev_s = &s;
                prev_parity = parity;
            }
            flush_pending();
            // Ensure own copy_st finished reading B.p2p_loc before a subsequent chunk reuses buffers
            for (int c = 0; c < nc_; ++c) {
                OnCard on(lane_[c].device);
                cudaStreamWaitEvent(strm(c), lane_[c].sent[1], 0);
            }
        } else {
            for (const Step& s : walk_.steps) {
                if (const int k = ds_->slot_of_visit(s.visit); k >= 0)   // the drafter's features: the visit's input rows
                    cudaMemcpyAsync(ds_->tap(k), buf(0, s.hid), (size_t) rows * E * 4, cudaMemcpyDeviceToDevice, strm(0));
                const int parity = (int) (exchanges_++ & 1);
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_attn(c, s, 0, rows, pos0, parity, 0, 0, false);
                }
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    layer_mlp(c, s, 0, rows, parity, 0, 0, false);
                }
                for (int c = 0; c < nc_; ++c) {
                    OnCard on(lane_[c].device);
                    finish(c, s, 0, rows, parity, 0);
                }
            }
        }
        if (ds_->on()) {   // and every row's final normed state, then the rows' context keys and values
            MixStates st;
            for (int i = 0; i < walk_.n_blocks; ++i) st.s[st.n++] = buf(0, walk_.blocks[i]);
            st.s[st.n++] = buf(0, walk_.final_hid);
            attnres_mix(st, m_.output_query(), 1, rows, 1e-6f, b(0).mix, (int64_t) rows * E, strm(0));
            bk::rms_norm_rows(b(0).mix, m_.output_norm(), rows, E, g_.rms_eps, ds_->tap(kDsTargets - 1), strm(0));
            DsparkHook::PromptScratch ps;
            ps.mix = b(0).mix;
            ps.kb = b(0).yk;
            ps.vb = b(0).yv;
            ps.xq = b(0).xqd;
            ps.pos3 = reinterpret_cast<int32_t*>(b(0).rlog);
            ps.bounds = b(0).dbounds;
            ps.ident = b(0).ident;
            ds_->commit_prompt(b(0).pos, rows, *lane_[0].mmq, ps, strm(0));
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    }

    // x (rows of `cols` values) as MMQ's q8_1 rows for products of `type`
    void quant(int c, const float* x, int cols, int rows, int type) { mmq::quantize(x, nullptr, b(c).xqd, type, cols, cols, rows, strm(c)); }
    // y[rows][n_out] = W x for the rows quant() rounded: one Q8_0 matrix [n_out][cols] at w
    void product(int c, const uint8_t* w, int n_out, int cols, int rows, float* y, int bounds_idx = 0) {
        mmq::Product p;
        p.w = w;
        p.type = 8;
        p.w_rows = n_out;
        p.w_cols = cols;
        p.expert_bytes = mmq::matrix_bytes(8, n_out, cols);
        p.n = 1;
        p.xq = b(c).xqd;
        p.bounds = b(c).dbounds + 2 * bounds_idx;
        p.ids = b(c).ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = y;
        p.ld_dst = n_out;
        lane_[c].mmq->run(p, strm(c));
    }

    // A loop controller over the chunk: out = blend(dest, msg) of stream `st` (out may be dest).
    void controller(int c, const LayerWeights& w, int st, const float* dest, const float* msg, int rows, float* out) {
        const int rank = g_.gate_rank;
        ctrl_down_prompt(dest, msg, w.ctrl_down[st], rows, rank, b(c).hid, strm(c));
        ctrl_up_blend_prompt(dest, msg, b(c).hid, w.ctrl_gf[st], rows, rank, out, strm(c));
    }

    // All four streams' messages of the previous pass into dst_msg[0..3] in one read of bank[0..1].
    void messages(int c, const Step& s, const LayerWeights& w, int off, int rows, float* dst_msg) {
        const size_t E = (size_t) g_.n_embd;
        MixStates st;
        st.s[st.n++] = buf(c, s.bank[0]) + (size_t) off * E;
        st.s[st.n++] = buf(c, s.bank[1]) + (size_t) off * E;
        attnres_mix(st, w.msg_query, 4, rows, 1e-6f, dst_msg, (int64_t) ((size_t) rows * E), strm(c));
    }

    // Card c's attention phase of step s for rows [off, off + rows): w.proj stays unsharded (single MMQ launch),
    // while attn_prepare (Q/R/sink), attention_prompt, attn_gate_q8_1_prompt and w.out_part are sharded under TP2
    // (KV caches stay replicated across cards so persistence/cards_agree are untouched).
    void layer_attn(int c, const Step& s, int off, int rows, int64_t pos0, int parity, int bounds_idx, int slice, bool async_p2p) {
        const Visit& v = g_.visits[(size_t) s.visit];
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l, c);
        const bool full = !g_.swa[(size_t) l];
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const size_t offE = (size_t) off * E;
        const VisitCache& vc = (*lane_[c].vc)[(size_t) v.slot];
        Bufs& B = b(c);
        float* const msg_base = B.msg + 4 * offE;
        float* const o_slice = B.o + offE;
        const float* hid = buf(c, s.hid) + offE;
        const float* blocks[kMaxMix];
        for (int i = 0; i < s.n_blocks; ++i) blocks[i] = buf(c, s.blocks[i]) + offE;
        // the attention's inputs
        const float* in[3] = {hid, hid, hid};
        if (w.attn_query) {
            MixStates st;
            for (int i = 0; i < s.n_blocks; ++i) st.s[st.n++] = blocks[i];
            st.s[st.n++] = hid;
            attnres_mix(st, w.attn_query, full ? 3 : 1, rows, 1e-6f, B.mix, (int64_t) RE, strm(c));
            for (int j = 0; j < 3; ++j) in[j] = B.mix + (full ? j : 0) * RE;
        }
        if (s.msgs) {
            messages(c, s, w, off, rows, msg_base);
            // controller j writes mix slot j; a sliding-window layer's one input feeds all three, so slot 0 goes last
            for (const int j : {1, 2, 0}) controller(c, w, j, in[j], msg_base + (size_t) j * RE, rows, B.mix + (size_t) j * RE);
            for (int j = 0; j < 3; ++j) in[j] = B.mix + (size_t) j * RE;
        }
        const bool one = !full && in[0] == in[1] && in[1] == in[2];
        const int ng = kKvHeads / nc_, g0 = c * ng;
        const int nh = ng * (kHeads / kKvHeads);
        const size_t row_bytes = (size_t) E / 32 * 34;
        if (one) {
            rms_norm_q8_1_prompt(in[0], w.attn_norm, rows, g_.rms_eps, B.xqd, strm(c));
            product(c, w.proj, kProj, E, rows, B.yq, bounds_idx);
        } else {
            const float* wn[3] = {full ? w.q_norm : w.attn_norm, full ? w.k_norm : w.attn_norm, full ? w.v_norm : w.attn_norm};
            rms_norm_q8_1_prompt(in[0], wn[0], rows, g_.rms_eps, B.xqd, strm(c));
            product(c, w.proj, kProjQ, E, rows, B.yq, bounds_idx);
            rms_norm_q8_1_prompt(in[1], wn[1], rows, g_.rms_eps, B.xqd, strm(c));
            product(c, w.proj + (size_t) kRowK * row_bytes, kProjKV, E, rows, B.yk, bounds_idx);
            rms_norm_q8_1_prompt(in[2], wn[2], rows, g_.rms_eps, B.xqd, strm(c));
            product(c, w.proj + (size_t) kRowV * row_bytes, kProjKV, E, rows, B.yv, bounds_idx);
        }
        // convolutions, clips and the keys: a sliding-window visit stages the slice's own, a full visit writes its cells
        const bool fp8 = m_.cache_fp8();
        const bool i8 = m_.cache_int8();
        AttnPrep ap;
        ap.yq = B.yq;
        ap.yk = one ? B.yq + kProjQ : B.yk;
        ap.yv = one ? B.yq + kProjQ + kProjKV : B.yv;
        if (one) ap.ldq = ap.ldk = ap.ldv = kProj;
        ap.pos = B.pos + off;
        ap.full = full;
        ap.k_conv = w.k_conv;
        ap.v_conv = w.v_conv;
        ap.q_scale = w.q_clip;
        ap.r_scale = w.r_clip;
        ap.qk_clip = g_.qk_clip;
        ap.rel_clip = g_.rel_clip;
        ap.eps = g_.rms_eps;
        ap.log_floor = g_.log_floor;
        ap.log_alpha = g_.log_alpha;
        ap.k_ring = vc.k_ring;
        ap.v_ring = vc.v_ring;
        ap.k_cache = vc.ring ? B.k_stage : vc.k;
        ap.v_cache = vc.ring ? B.v_stage : vc.v;
        ap.k_scale = i8 ? (vc.ring ? B.ks_stage : vc.ks) : nullptr;
        ap.v_scale = i8 ? (vc.ring ? B.vs_stage : vc.vs) : nullptr;
        ap.cap = vc.ring ? kChunk : vc.cap;
        ap.fp8 = fp8;
        ap.q = B.q;
        ap.r = B.r;
        ap.sink = B.sink;
        ap.g0 = g0;
        ap.ng = ng;
        ap.kv_stride = ng;
        attn_prepare(ap, rows, strm(c));
        AttnPromptArgs aa;
        aa.q = B.q;
        aa.r = B.r;
        aa.sink = B.sink;
        aa.rel_table = w.rel_table;
        aa.pos0 = pos0;
        aa.k_cache = vc.k;
        aa.v_cache = vc.v;
        aa.k_scale = vc.ks;
        aa.v_scale = vc.vs;
        aa.cap = vc.cap;
        aa.k_chunk = ap.k_cache;
        aa.v_chunk = ap.v_cache;
        aa.k_chunk_scale = ap.k_scale;
        aa.v_chunk_scale = ap.v_scale;
        aa.chunk_cap = ap.cap;
        aa.fp8 = fp8;
        aa.window = vc.ring ? g_.window : 0;
        aa.scale = 1.0f / std::sqrt((float) kHeadDim);
        aa.g0 = g0;
        aa.ng = ng;
        aa.kv_stride = ng;
        attention_prompt(aa, rows, B.attn, strm(c));
        if (vc.ring)
            stage_to_ring(B.k_stage, B.v_stage, kChunk, vc.k, vc.v, vc.cap, pos0, rows, g_.window, (fp8 || i8) ? 1 : 2, strm(c),
                          i8 ? B.ks_stage : nullptr, i8 ? B.vs_stage : nullptr, vc.ks, vc.vs, ng);
        attn_gate_q8_1_prompt(B.attn, B.yq, one ? kProj : kProjQ, rows, B.xqd, strm(c), g0, ng);
        if (nc_ > 1) {
            if (async_p2p) cudaStreamWaitEvent(strm(c), lane_[c].sent_attn[slice], 0);
            product(c, w.out_part, E, nh * kHeadDim, rows, o_slice, bounds_idx);
            if (async_p2p) {
                cudaEventRecord(lane_[c].attn_ready[slice], strm(c));
                cudaStreamWaitEvent(lane_[c].copy_st, lane_[c].attn_ready[slice], 0);
                if (slice == 0) cudaStreamWaitEvent(lane_[c].copy_st, lane_[1 - c].recv_attn_free[parity], 0);
                cudaMemcpyPeerAsync(b(1 - c).recv_attn_f32[parity] + offE, lane_[1 - c].device, o_slice, lane_[c].device,
                                    RE * sizeof(float), lane_[c].copy_st);
                cudaEventRecord(lane_[c].sent_attn[slice], lane_[c].copy_st);
            } else {
                cudaMemcpyPeerAsync(b(1 - c).recv_attn_f32[parity] + offE, lane_[1 - c].device, o_slice, lane_[c].device,
                                    RE * sizeof(float), strm(c));
                cudaEventRecord(lane_[c].sent_attn[slice], strm(c));
            }
        } else {
            product(c, w.out, E, kOut, rows, o_slice, bounds_idx);
        }
    }

    // Card c's MLP phase of step s for rows [off, off + rows): sums TP2 attention partial outputs in FP32, runs post-attn conv and experts.
    void layer_mlp(int c, const Step& s, int off, int rows, int parity, int bounds_idx, int slice, bool async_p2p) {
        (void) bounds_idx;
        const Visit& v = g_.visits[(size_t) s.visit];
        const int l = v.layer;
        const LayerWeights& w = m_.layer(l, c);
        const int E = g_.n_embd;
        const size_t RE = (size_t) rows * E;
        const size_t offE = (size_t) off * E;
        const VisitCache& vc = (*lane_[c].vc)[(size_t) v.slot];
        Bufs& B = b(c);
        float* const msg_base = B.msg + 4 * offE;
        float* const o_slice = B.o + offE;
        const float* hid = buf(c, s.hid) + offE;
        const float* blocks[kMaxMix];
        for (int i = 0; i < s.n_blocks; ++i) blocks[i] = buf(c, s.blocks[i]) + offE;
        if (nc_ > 1) {
            cudaStreamWaitEvent(strm(c), lane_[1 - c].sent_attn[slice], 0);
            if (async_p2p) cudaStreamWaitEvent(strm(c), lane_[c].sent_attn[slice], 0);   // own copy_st done reading o_slice before add_inplace
            bk::add_inplace(o_slice, B.recv_attn_f32[parity] + offE, (int64_t) RE, strm(c));
            if (async_p2p && slice == 1) cudaEventRecord(lane_[c].recv_attn_free[parity], strm(c));
        }
        // the attention's convolution and the residual: a block's first layer starts a new sum
        const bool boundary = g_.boundary(l);
        post_conv(o_slice, B.pos + off, w.attn_post_conv, vc.a_ring, boundary ? nullptr : hid, rows, B.resid + offE, strm(c));
        // the MLP's input
        MixStates st;
        for (int i = 0; i < s.n_blocks; ++i) st.s[st.n++] = blocks[i];
        if (boundary) st.s[st.n++] = hid;
        st.s[st.n++] = B.resid + offE;
        attnres_mix(st, w.ffn_query, 1, rows, 1e-6f, B.mix, (int64_t) RE, strm(c));
        if (s.msgs) controller(c, w, 3, B.mix, msg_base + 3 * RE, rows, B.mix);
        bk::rms_norm_rows(B.mix, w.ffn_norm, rows, E, g_.rms_eps, B.nrm, strm(c));
        experts(c, w, l, off, rows, parity, slice, async_p2p);
    }

    // Card c's end of step s's visit for rows [off, off + rows): the two cards' experts' sums added (tensor parallelism),
    // the MLP's convolution and the residual into the visit's output.
    void finish(int c, const Step& s, int off, int rows, int parity, int slice) {
        const Visit& v = g_.visits[(size_t) s.visit];
        const LayerWeights& w = m_.layer(v.layer, c);
        const VisitCache& vc = (*lane_[c].vc)[(size_t) v.slot];
        const size_t offE = (size_t) off * (size_t) g_.n_embd;
        Bufs& B = b(c);
        if (nc_ > 1) {
            cudaStreamWaitEvent(strm(c), lane_[1 - c].sent[slice], 0);
            add_p2p_f16_scaled(B.p2p_loc + offE, B.p2p_loc_sc + off, B.recv_f16[parity] + offE, B.recv_sc[parity] + off,
                               B.moe, rows, g_.n_embd, strm(c));
        }
        post_conv(B.moe, B.pos + off, w.ffn_post_conv, vc.m_ring, B.resid + offE, rows, buf(c, s.out) + offE, strm(c));
    }

    // The routed experts of the slice's normed MLP inputs (nrm) into moe / p2p_loc.
    void experts(int c, const LayerWeights& w, int l, int off, int rows, int parity, int slice, bool async_p2p) {
        const int E = g_.n_embd, F = (int) w.n_ff, k = g_.n_used, pairs = rows * k;
        const size_t offE = (size_t) off * E;
        Bufs& B = b(c);
        const cudaStream_t st = strm(c);
        bk::f32_rows_dot(B.nrm, w.router, B.rlog, rows, g_.n_expert, E, st);
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = k;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router_select(B.rlog, w.router_bias, rows, ra, B.ids, B.rw, st);
        if (w.slot_map) {   // a layer with experts in RAM (spill.hpp): its pairs by slot, its RAM experts copied into the staging
            remap_ids(B.ids, pairs, w.slot_map, st);
            const uint64_t kept = (uint64_t) (g_.n_expert - w.n_spill) * w.expert_stride;
            if (!m_.stage_from_card(l, st))   // the elastic spill's card copies when the layer has them all
                cudaMemcpyAsync(w.experts + kept, w.spill_host, (size_t) w.n_spill * w.expert_stride, cudaMemcpyHostToDevice, st);
            cudaMemsetAsync(w.experts + (uint64_t) g_.n_expert * w.expert_stride, 0, kSlotGuard, st);   // spill.hpp
        }
        moe_sort(B.ids, rows, k, g_.n_expert, B.sort_scr, B.bounds, B.tok, B.dst, B.exp, st);
        mmq::quantize(B.nrm, B.tok, B.xq, w.gu_type, E, E, pairs, st);
        mmq::Product p;
        p.w = w.experts;
        p.type = w.gu_type;
        p.w_rows = 2 * F;
        p.w_cols = E;
        p.expert_bytes = w.expert_stride;
        p.n = g_.n_expert;
        p.xq = B.xq;
        p.bounds = B.bounds;
        p.ids = B.ident;
        p.total_rows = pairs;
        p.max_rows = rows;
        p.opt_rows = (pairs + g_.n_expert - 1) / g_.n_expert;
        p.dst = B.gu;
        p.ld_dst = 2 * F;
        lane_[c].mmq->run(p, st);
        // an NVFP4 expert's {s_gate, s_up, s_down, 0} tail is read in its blob
        const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, F);
        mmq::SwigluForm form;
        form.gate_softcap = g_.gate_softcap;
        form.up_softcap = g_.up_softcap;
        form.row_expert = B.exp;
        form.tail_stride = w.expert_stride;
        const bool nvfp4 = w.gu_type == kTypeNvfp4;
        mmq::swiglu(B.gu, B.h, pairs, F, false, st, nvfp4 ? (const float*) (w.experts + L.bytes - 16) : nullptr, nullptr, 0, form);
        mmq::quantize(B.h, nullptr, B.xq, w.d_type, F, F, pairs, st);
        mmq::Product d = p;
        d.w = w.experts + L.down_off;
        d.type = w.d_type;
        d.w_rows = E;
        d.w_cols = F;
        d.ids = B.dst;
        d.dst = B.gu;
        d.ld_dst = E;
        lane_[c].mmq->run(d, st);
        if (nc_ > 1) {
            if (async_p2p) {
                cudaStreamWaitEvent(st, lane_[c].sent[slice], 0);   // own copy_st done reading p2p_loc[slice] from previous step
                moe_combine_p2p_f16(B.gu, B.rw, rows, k, B.p2p_loc + offE, B.p2p_loc_sc + off, nullptr, nullptr, st);
                cudaEventRecord(lane_[c].moe_ready[slice], st);
                cudaStreamWaitEvent(lane_[c].copy_st, lane_[c].moe_ready[slice], 0);
                if (slice == 0) cudaStreamWaitEvent(lane_[c].copy_st, lane_[1 - c].recv_moe_free[parity], 0);
                cudaMemcpyPeerAsync(b(1 - c).recv_f16[parity] + offE, lane_[1 - c].device, B.p2p_loc + offE, lane_[c].device,
                                    (size_t) rows * E * sizeof(uint16_t), lane_[c].copy_st);
                cudaMemcpyPeerAsync(b(1 - c).recv_sc[parity] + off, lane_[1 - c].device, B.p2p_loc_sc + off, lane_[c].device,
                                    (size_t) rows * sizeof(float), lane_[c].copy_st);
                cudaEventRecord(lane_[c].sent[slice], lane_[c].copy_st);
            } else {
                moe_combine_p2p_f16(B.gu, B.rw, rows, k, B.p2p_loc + offE, B.p2p_loc_sc + off,
                                    b(1 - c).recv_f16[parity] + offE, b(1 - c).recv_sc[parity] + off, st);
                cudaEventRecord(lane_[c].sent[slice], st);
            }
        } else {
            moe_combine(B.gu, B.rw, rows, k, B.moe, st, nullptr);
        }
    }

    InklingModel& m_;
    const Geometry& g_;
    const int nc_;                // the cards (tensor parallelism: 2)
    DsparkHook* ds_ = nullptr;    // the reading session's drafter
    Walk walk_;
    Lane lane_[2];                // each card's buffers, products and stream
    cudaEvent_t done_[2] = {};
    cudaEvent_t fork_ = nullptr;  // the second card's stream after the first's
    uint64_t exchanges_ = 0;      // the receive buffers alternate: a card writes one while the other may still read the other
};

}  // namespace

std::unique_ptr<PromptRunner> make_prompt(InklingModel& m, void* const streams[2], std::string& err) {
    auto p = std::make_unique<Prompt>(m, streams);
    if (!p->init(err)) return nullptr;
    return p;
}

uint64_t prompt_bytes(const Geometry& g) {
    // the whole experts' width bounds a card's half with its receive buffers (P F / 2 == 2 R E at n_used 8)
    Carve c;
    Bufs b;
    carve(c, b, g, g.n_ff_exp, false);
    Carve h;
    carve(h, b, g, g.n_ff_exp / 2, true);
    return (uint64_t) std::max(c.at, h.at) + kMmqSpare;
}

}  // namespace granite::models::inkling2
