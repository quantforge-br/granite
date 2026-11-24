// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of four warps, at most
// kMaxSplits slices), so a row's bits do not depend on the other rows of its call or on the launch's bound. A block takes
// kRows in {1, 2, 4} rows, one key/value head (its three query heads) and one slice, four warps a row. Rows that share the
// same cache and start within kSlack keys share a single KV stage pass; otherwise rows partition into separate stage groups.
// Each warp walks its tiles of 32 keys with a lane a key, skipping the kRel=16 relative bias uniformly on far tiles
// (dist >= kExtent), and then sums values with four channels a lane. In FP8 (E4M3), K and V stay raw in shared memory via
// cp.async (staggered K/V overlap for kRows=1,2 at 2 blocks/SM; 2-stage K+V double buffering for kRows=4 at 1 block/SM)
// and unpack branchlessly via bit-shift + non-FTZ HMUL2 by 2^8.
#include "e4m3_fast.cuh"
#include "granite/kernels/pdl.hpp"
#include "granite/kernels/q8_1_finite.hpp"
#include "granite/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace granite::models::inkling2 {
namespace {

namespace bk = granite::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
constexpr int kMaxSplits = 64;
constexpr int kSlack = 8;                    // how far apart rows' slices may start and still share a stage
constexpr int kStage = kUnit + kSlack;       // key rows a stage holds
constexpr int kChunks = kHeadDim * 2 / 16;   // 16-byte chunks of a row in shared memory (fp16)
constexpr int kFp8Loads = kHeadDim / 16;     // 8 x 16-byte chunks of a row in shared memory (fp8)

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// The keys a slice of a row with n keys covers: whole passes of the block, at most kMaxSplits slices.
__host__ __device__ __forceinline__ int span_of(int n) {
    const int units = (n + kUnit * kMaxSplits - 1) / (kUnit * kMaxSplits);
    return (units > 1 ? units : 1) * kUnit;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& n) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    n = p + 1 - lo;
}

// 16 bytes global -> shared, asynchronously (zero-filled when !ok).
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cp16_z(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(ok ? 16 : 0));
}

// 8 cache values as fp32 from 16 bytes of fp16.
__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

// 16 cache values as fp32 from 16 bytes of E4M3 via branchless bit-shift + non-FTZ HMUL2.
__device__ __forceinline__ void unpack16_fp8(const uint4 u, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __half2 h0, h1;
        e4m3x4_to_h2x2_fast(w[i], h0, h1);
        const float2 a = __half22float2(h0), b = __half22float2(h1);
        f[4 * i] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}

template <int kRows, bool FP8, int kStages>
struct SmemPayload;

template <int kRows>
struct SmemPayload<kRows, false, 1> {
    uint4 kv[(kSlack + kStage) * kChunks];
};

template <int kRows, int kStages>
struct SmemPayload<kRows, true, kStages> {
    uint4 k[kStages][kStage * kFp8Loads];
    uint4 v[kStages][kStage * kFp8Loads];
};

template <int kRows, bool FP8>
constexpr int kFp8Stages = (FP8 && kRows == 4) ? 2 : 1;

// A row's slice: its keys [lo, hi) and caches; on when the row has this slice.
struct RowSlice {
    bool on = false;
    int p = 0, lo = 0, hi = 0;
    const uint8_t* k = nullptr;
    const uint8_t* v = nullptr;
};

struct StageGroup {
    int lo = 0, hi = 0, dl = 0, passes = 0;
    const uint8_t* kb = nullptr;
    const uint8_t* vb = nullptr;
};

template <int kRows, bool FP8>
struct Smem {
    float q[kRows][kGroup][kHeadDim];
    float rs[kRows][kGroup][kRel];
    float m[kRows][kWarps][kGroup], l[kRows][kWarps][kGroup];
    RowSlice rk[kRows];
    StageGroup sg[kRows];
    int grp[kRows];
    int n_stages;
    SmemPayload<kRows, FP8, kFp8Stages<kRows, FP8>> st;
};

template <int kRows>
__device__ __forceinline__ int partition_stages(const RowSlice* rk, int* grp, StageGroup* sg) {
    int n_stages = 0;
    for (int k = 0; k < kRows; ++k) {
        grp[k] = -1;
        if (!rk[k].on) continue;
        int found = -1;
        for (int g = 0; g < n_stages; ++g) {
            if (sg[g].kb == rk[k].k && sg[g].vb == rk[k].v) {
                const int nlo = min(sg[g].lo, rk[k].lo);
                const int mlo = max(sg[g].lo + sg[g].dl, rk[k].lo);
                if (mlo - nlo <= kSlack) {
                    found = g;
                    sg[g].lo = nlo;
                    sg[g].dl = mlo - nlo;
                    sg[g].hi = max(sg[g].hi, rk[k].hi);
                    break;
                }
            }
        }
        if (found < 0) {
            found = n_stages++;
            sg[found].lo = rk[k].lo;
            sg[found].hi = rk[k].hi;
            sg[found].dl = 0;
            sg[found].kb = rk[k].k;
            sg[found].vb = rk[k].v;
        }
        grp[k] = found;
    }
    for (int g = 0; g < n_stages; ++g) {
        int max_len = 0;
        for (int k = 0; k < kRows; ++k)
            if (grp[k] == g) max_len = max(max_len, rk[k].hi - rk[k].lo);
        sg[g].passes = (max_len + kUnit - 1) / kUnit;
    }
    return n_stages;
}

template <int kRows, bool FP8, int kMinBlocks>
__global__ void __launch_bounds__(kRows * kWarps * 32, kMinBlocks)
    attn_split_kernel(AttnArgs a, int rows, int z, float* __restrict__ part_ml, float* __restrict__ part_acc) {
    constexpr int kThreads = kRows * kWarps * 32;
    constexpr int kStages = kFp8Stages<kRows, FP8>;
    using SmemT = Smem<kRows, FP8>;
    static_assert(sizeof(float) * kRows * kWarps * kGroup * kHeadDim <= sizeof(SmemPayload<kRows, FP8, kStages>),
                  "the sums fit the stage");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    SmemT& sm = *reinterpret_cast<SmemT*>(smem_raw);
    const int r0 = blockIdx.x * kRows, g = a.g0 + blockIdx.y, s = blockIdx.z;
    const int slot = kRows == 1 ? 0 : (threadIdx.x >> 7);
    const int w = (threadIdx.x >> 5) & (kWarps - 1), lane = threadIdx.x & 31;
    bk::pdl_wait();
    bk::pdl_launch_dependents();

    constexpr int kRowBytes = FP8 ? kHeadDim : kHeadDim * 2;
    constexpr size_t kCell = (size_t) kKvHeads * kRowBytes;
    constexpr int kLoads = kRowBytes / 16;   // 8 for FP8, 16 for FP16
    constexpr int kStep = kThreads / kLoads;

    if (threadIdx.x == 0) {
        for (int k = 0; k < kRows; ++k) {
            sm.rk[k] = RowSlice{};
            const int r = r0 + k;
            if (r >= rows) continue;
            int p, lo, n;
            row_keys(a, r, p, lo, n);
            const int span = span_of(n);
            if (s * span >= n) continue;
            sm.rk[k].on = true;
            sm.rk[k].p = p;
            sm.rk[k].lo = lo + s * span;
            sm.rk[k].hi = min(p + 1, sm.rk[k].lo + span);
            const SegCaches& sc = a.seg.c[bk::seg_of(a.seg.at, r)];
            sm.rk[k].k = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
            sm.rk[k].v = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
        }
        sm.n_stages = partition_stages<kRows>(sm.rk, sm.grp, sm.sg);
    }
    __syncthreads();
    const int n_stages = sm.n_stages;
    if (n_stages == 0) return;

    const int j0 = threadIdx.x / kLoads, ch = threadIdx.x % kLoads;
    const int cap = (int) a.cap;

    // Load queries and relative states into smem (for all active rows).
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
        if (!sm.rk[k].on) continue;
        const int r = r0 + k;
        for (int i = threadIdx.x; i < kGroup * kHeadDim; i += kThreads)
            sm.q[k][i / kHeadDim][i % kHeadDim] = a.q[((size_t) r * kHeads + g * kGroup) * kHeadDim + i];
        for (int i = threadIdx.x; i < kGroup * kRel; i += kThreads)
            sm.rs[k][i / kRel][i % kRel] = a.r[((size_t) r * kHeads + g * kGroup) * kRel + i];
    }

    const RowSlice me = sm.rk[slot];
    const int my_grp = sm.grp[slot];
    float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
    }

    if constexpr (!FP8) {
        // ---- FP16 path (bit-exact kUnit=128 layout, kRows in {1, 2, 4}, far-tile kRel skip) ----
        constexpr int kKeyAt = kSlack * kChunks;
        constexpr int kVEarly = kSlack;
        constexpr int kPerK = (kStage + kStep - 1) / kStep;
        constexpr int kPerV = (kStage - kVEarly + kStep - 1) / kStep;
        static_assert(kVEarly <= kStep, "a thread copies at most one early value chunk");

        uint4 vr[kPerV];
        const auto load_f16 = [&](const StageGroup& st, int b, int e) {
            const int c0 = (b + j0) % cap;
            const auto cell = [&](int j) { const int c = c0 + j; return (size_t) (c >= cap ? c - cap : c) * kCell + ch * 16; };
            if (j0 < kVEarly && b + j0 < e) cp16(&sm.st.kv[j0 * kChunks + ch], st.vb + cell(0));
#pragma unroll
            for (int i = 0; i < kPerK || i < kPerV; ++i) {
                const int j = kStep * i;
                if (i < kPerK && j0 + j < kStage) {
                    const int jj = j0 + j;
                    if (b + jj < e) cp16(&sm.st.kv[kKeyAt + jj * kLoads + (ch ^ (jj & (kLoads - 1)))], st.kb + cell(j));
                }
                if (i < kPerV && j0 + kVEarly + j < kStage)
                    vr[i] = b + j0 + kVEarly + j < e ? *reinterpret_cast<const uint4*>(st.vb + cell(kVEarly + j)) : make_uint4(0, 0, 0, 0);
            }
        };

        for (int q = 0; q < n_stages; ++q) {
            const StageGroup st = sm.sg[q];
            const bool mine = me.on && my_grp == q;
            for (int t = 0; t < st.passes; ++t) {
                if (q > 0 || t > 0) __syncthreads();
                const int stage_lo = st.lo + t * kUnit;
                load_f16(st, stage_lo, min(st.hi, stage_lo + kUnit + st.dl));
                asm volatile("cp.async.wait_all;\n" ::: "memory");
                __syncthreads();

                const int t0 = me.lo + t * kUnit + w * kTile, key = t0 + lane, jt = t0 - stage_lo + lane;
                const bool tile = mine && t0 < me.hi, valid = key < me.hi;
                float sc[kGroup] = {0.f, 0.f, 0.f};
                if (tile) {
#pragma unroll
                    for (int i = 0; i < kLoads; ++i) {
                        const uint4 kc = sm.st.kv[kKeyAt + jt * kLoads + (i ^ (jt & (kLoads - 1)))];
                        float f[8];
                        unpack8_f16(kc, f);
#pragma unroll
                        for (int d = 0; d < 8; ++d)
#pragma unroll
                            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[slot][h][8 * i + d], f[d], sc[h]);
                    }
                    if (me.p - (t0 + kTile - 1) < kExtent) {
                        const int dist = me.p - key;
                        float tv[kRel];
#pragma unroll
                        for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) {
                            float b = 0.f;
#pragma unroll
                            for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[slot][h][i], tv[i], b);
                            sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
                        }
                    } else {
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) sc[h] = valid ? sc[h] * a.scale : -INFINITY;
                    }
                }
                __syncthreads();
#pragma unroll
                for (int i = 0; i < kPerV; ++i) {
                    const int j = j0 + kVEarly + kStep * i;
                    if (j < kStage) sm.st.kv[j * kChunks + ch] = vr[i];
                }
                __syncthreads();
                if (!tile) continue;

                float pr[kGroup];
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float mt = warp_max(sc[h]);
                    const float mn = fmaxf(m[h], mt);
                    const float corr = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
                    pr[h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
                    l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
                    for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
                    m[h] = mn;
                }
                const int jn = me.hi - t0;
                const uint32_t* vw = reinterpret_cast<const uint32_t*>(sm.st.kv) + (size_t) (t0 - stage_lo) * (kHeadDim / 2) + lane * 2;
#pragma unroll
                for (int j = 0; j < kTile; ++j) {
                    uint2 u = *reinterpret_cast<const uint2*>(vw + j * (kHeadDim / 2));
                    if (j >= jn) u = make_uint2(0, 0);
                    const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                    const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) {
                        const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                        acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                        acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                        acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                        acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
                    }
                }
            }
        }
    } else {
        // ---- FP8 (E4M3) path: raw FP8 K & V via cp.async + branchless HMUL2 unpack + far-tile kRel skip ----
        constexpr int kPer = (kStage + kStep - 1) / kStep;
        const auto load_k_fp8 = [&](int buf_idx, const StageGroup& st, int b, int e) {
            const int c0 = (b + j0) % cap;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int jj = j0 + kStep * i;
                if (jj < kStage) {
                    const bool ok = b + jj < e;
                    const int c = c0 + kStep * i;
                    const size_t off = (size_t) (c >= cap ? c - cap : c) * kCell + ch * 16;
                    cp16_z(&sm.st.k[buf_idx][jj * kFp8Loads + (ch ^ (jj & (kFp8Loads - 1)))], ok ? st.kb + off : st.kb, ok);
                }
            }
        };
        const auto load_v_fp8 = [&](int buf_idx, const StageGroup& st, int b, int e) {
            const int c0 = (b + j0) % cap;
#pragma unroll
            for (int i = 0; i < kPer; ++i) {
                const int jj = j0 + kStep * i;
                if (jj < kStage) {
                    const bool ok = b + jj < e;
                    const int c = c0 + kStep * i;
                    const size_t off = (size_t) (c >= cap ? c - cap : c) * kCell + ch * 16;
                    cp16_z(&sm.st.v[buf_idx][jj * kFp8Loads + ch], ok ? st.vb + off : st.vb, ok);
                }
            }
        };

        const auto score_k_fp8 = [&](int buf_idx, int stage_lo, int t, bool mine, float (&sc)[kGroup], bool& tile, int& t0) {
            t0 = me.lo + t * kUnit + w * kTile;
            const int key = t0 + lane, jt = t0 - stage_lo + lane;
            tile = mine && t0 < me.hi;
            const bool valid = key < me.hi;
            sc[0] = sc[1] = sc[2] = 0.f;
            if (!tile) return;
#pragma unroll
            for (int i = 0; i < kFp8Loads; ++i) {
                const uint4 kc = sm.st.k[buf_idx][jt * kFp8Loads + (i ^ (jt & (kFp8Loads - 1)))];
                float f[16];
                unpack16_fp8(kc, f);
#pragma unroll
                for (int d = 0; d < 16; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[slot][h][16 * i + d], f[d], sc[h]);
            }
            if (me.p - (t0 + kTile - 1) < kExtent) {
                const int dist = me.p - key;
                float tv[kRel];
#pragma unroll
                for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    float b = 0.f;
#pragma unroll
                    for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[slot][h][i], tv[i], b);
                    sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
                }
            } else {
#pragma unroll
                for (int h = 0; h < kGroup; ++h) sc[h] = valid ? sc[h] * a.scale : -INFINITY;
            }
        };

        const auto accum_v_fp8 = [&](int buf_idx, int stage_lo, int t0, const float (&sc)[kGroup]) {
            float pr[kGroup];
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float mt = warp_max(sc[h]);
                const float mn = fmaxf(m[h], mt);
                const float corr = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
                pr[h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
                l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
                for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
                m[h] = mn;
            }
            const int jn = me.hi - t0;
            // Raw FP8 V: 128 bytes/row = 32 uint32_t/row -> lane reads 4 channels with zero bank conflicts!
            const uint32_t* vw = reinterpret_cast<const uint32_t*>(sm.st.v[buf_idx]) + (size_t) (t0 - stage_lo) * (kHeadDim / 4) + lane;
#pragma unroll
            for (int j = 0; j < kTile; ++j) {
                const uint32_t u = j < jn ? vw[j * (kHeadDim / 4)] : 0u;
                __half2 h0, h1;
                e4m3x4_to_h2x2_fast(u, h0, h1);
                const float2 v01 = __half22float2(h0);
                const float2 v23 = __half22float2(h1);
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                    acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                    acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                    acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                    acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
                }
            }
        };

        if constexpr (kStages == 2) {
            // Full 2-stage K+V double buffering (kRows=4, 1 block/SM = 16 warps/SM)
            for (int q = 0; q < n_stages; ++q) {
                const StageGroup st = sm.sg[q];
                const bool mine = me.on && my_grp == q;
                if (q > 0) __syncthreads();
                load_k_fp8(0, st, st.lo, min(st.hi, st.lo + kUnit + st.dl));
                load_v_fp8(0, st, st.lo, min(st.hi, st.lo + kUnit + st.dl));
                asm volatile("cp.async.commit_group;\n" ::: "memory");

                for (int t = 0; t < st.passes; ++t) {
                    const int cur = t & 1, nxt = cur ^ 1;
                    if (t > 0) __syncthreads();
                    if (t + 1 < st.passes) {
                        const int nlo = st.lo + (t + 1) * kUnit;
                        load_k_fp8(nxt, st, nlo, min(st.hi, nlo + kUnit + st.dl));
                        load_v_fp8(nxt, st, nlo, min(st.hi, nlo + kUnit + st.dl));
                        asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");
                    } else {
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    }
                    __syncthreads();
                    float sc[kGroup];
                    bool tile = false;
                    int t0 = 0;
                    const int stage_lo = st.lo + t * kUnit;
                    score_k_fp8(cur, stage_lo, t, mine, sc, tile, t0);
                    if (tile) accum_v_fp8(cur, stage_lo, t0, sc);
                }
            }
        } else {
            // Staggered 1-stage K+V cp.async overlap (kRows=1,2 at 2 blocks/SM)
            for (int q = 0; q < n_stages; ++q) {
                const StageGroup st = sm.sg[q];
                const bool mine = me.on && my_grp == q;
                if (q > 0) __syncthreads();
                load_k_fp8(0, st, st.lo, min(st.hi, st.lo + kUnit + st.dl));
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                load_v_fp8(0, st, st.lo, min(st.hi, st.lo + kUnit + st.dl));
                asm volatile("cp.async.commit_group;\n" ::: "memory");

                for (int t = 0; t < st.passes; ++t) {
                    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                    __syncthreads();
                    float sc[kGroup];
                    bool tile = false;
                    int t0 = 0;
                    const int stage_lo = st.lo + t * kUnit;
                    score_k_fp8(0, stage_lo, t, mine, sc, tile, t0);
                    __syncthreads();
                    if (t + 1 < st.passes) {
                        const int nlo = st.lo + (t + 1) * kUnit;
                        load_k_fp8(0, st, nlo, min(st.hi, nlo + kUnit + st.dl));
                        asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");
                    } else {
                        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    }
                    __syncthreads();
                    if (tile) accum_v_fp8(0, stage_lo, t0, sc);
                    if (t + 1 < st.passes) {
                        __syncthreads();
                        const int nlo = st.lo + (t + 1) * kUnit;
                        load_v_fp8(0, st, nlo, min(st.hi, nlo + kUnit + st.dl));
                        asm volatile("cp.async.commit_group;\n" ::: "memory");
                    }
                }
            }
        }
    }

    // each row's four warps' parts of this slice, merged in warp order
    __syncthreads();   // the values read: the stage takes the sums
    float* sacc = reinterpret_cast<float*>(&sm.st);   // [kRows][kWarps][kGroup][kHeadDim]
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        if (lane == 0) {
            sm.m[slot][w][h] = m[h];
            sm.l[slot][w][h] = l[h];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) sacc[((slot * kWarps + w) * kGroup + h) * kHeadDim + lane * 4 + u] = acc[h][u];
    }
    __syncthreads();
    if (!me.on) return;
    const int r = r0 + slot;
    for (int i = threadIdx.x & (kWarps * 32 - 1); i < kGroup * kHeadDim; i += kWarps * 32) {
        const int h = i / kHeadDim, d = i % kHeadDim;
        float mm = -INFINITY;
        for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[slot][ww][h]);
        float ll = 0.f, aa = 0.f;
        for (int ww = 0; ww < kWarps; ++ww) {
            if (sm.m[slot][ww][h] == -INFINITY) continue;
            const float f = expf(sm.m[slot][ww][h] - mm);
            ll = fmaf(sm.l[slot][ww][h], f, ll);
            aa = fmaf(sacc[((slot * kWarps + ww) * kGroup + h) * kHeadDim + d], f, aa);
        }
        const size_t sl = ((size_t) r * kHeads + g * kGroup + h) * z + s;
        part_acc[sl * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[sl * 2] = mm;
            part_ml[sl * 2 + 1] = ll;
        }
    }
}

// A block per (row, head), a thread a channel: the row's slices merged with the sink, gated, written in fp32 and/or as
// q8_1 (a warp a block of 32 channels, the range's heads' blocks only).
__global__ void __launch_bounds__(kHeadDim) attn_combine_kernel(AttnArgs a, int z, const float* __restrict__ part_ml,
                                                                const float* __restrict__ part_acc, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x, h = a.g0 * kGroup + blockIdx.y, d = threadIdx.x;
    if (a.sent && blockIdx.x == 0 && blockIdx.y == 0 && d == 0) *a.sent += 1;   // read after the output projection
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n), splits = (n + span - 1) / span;
    const size_t base = ((size_t) r * kHeads + h) * z;
    const float t = a.sink[(size_t) r * kHeads + h];
    float mm = t;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, part_ml[(base + s) * 2]);
    float ll = expf(t - mm), aa = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float f = expf(part_ml[(base + s) * 2] - mm);
        ll = fmaf(part_ml[(base + s) * 2 + 1], f, ll);
        aa = fmaf(part_acc[(base + s) * kHeadDim + d], f, aa);
    }
    float o = aa / ll;
    const int c = h * kHeadDim + d;
    if (a.gate) o *= 1.f / (1.f + expf(-a.gate[(size_t) r * a.gate_ld + c]));
    if (out) out[(size_t) r * kHeads * kHeadDim + c] = o;
    if (a.q8) {
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        const int rc = c - a.g0 * kGroup * kHeadDim;   // the channel in the range's heads
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kGroup * a.ng * kHeadDim / 32) + rc / 32];
        b.qs[rc % 32] = bk::q8_1_quant(o, dq, amax);
        if ((rc & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
}

__global__ void e4m3_fast_unpack_all_256_kernel(uint16_t* out_h16) {
    const int i = threadIdx.x;   // 0..127 -> 2 bytes each = 256 codes
    const uint16_t pair = (uint16_t) ((2 * i) | ((2 * i + 1) << 8));
    const uint32_t h2 = e4m3x2_to_h2_u32(pair);
    out_h16[2 * i] = (uint16_t) (h2 & 0xFFFFu);
    out_h16[2 * i + 1] = (uint16_t) (h2 >> 16);
}

int launch_splits(int64_t max_keys) {
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit));
}

template <int kRows, bool FP8, int kMinBlocks>
void launch_split(const AttnArgs& a, int rows, int z, float* part_ml, float* part_acc, cudaStream_t s) {
    const dim3 grid((unsigned) ((rows + kRows - 1) / kRows), (unsigned) a.ng, (unsigned) z);
    bk::launch_pdl(attn_split_kernel<kRows, FP8, kMinBlocks>, grid, dim3(kRows * kWarps * 32), sizeof(Smem<kRows, FP8>), s,
                   a, rows, z, part_ml, part_acc);
}

}  // namespace

int opt_in_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kOptInDevices) throw std::runtime_error("inkling2: device out of range");
    return dev;
}

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t slots = (size_t) rows * kHeads * (size_t) launch_splits(max_keys);
    return slots * (2 + kHeadDim) * sizeof(float);
}

void e4m3_fast_unpack_all_256(uint16_t* out_h16, void* stream) {
    e4m3_fast_unpack_all_256_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(out_h16);
    check("e4m3_fast_unpack_all_256");
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const int z = launch_splits(max_keys);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * z * 2;
    static bool attr[kOptInDevices] = {};   // the opt-in is per device
    if (bool& set = attr[opt_in_device()]; !set) {
        cudaFuncSetAttribute(attn_split_kernel<1, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<1, false>));
        cudaFuncSetAttribute(attn_split_kernel<2, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<2, false>));
        cudaFuncSetAttribute(attn_split_kernel<4, false, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<4, false>));
        cudaFuncSetAttribute(attn_split_kernel<1, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<1, true>));
        cudaFuncSetAttribute(attn_split_kernel<2, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<2, true>));
        cudaFuncSetAttribute(attn_split_kernel<4, true, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<4, true>));
        set = true;
    }
    if (a.cap < kStage) throw std::runtime_error("inkling2 attention_decode: the cache holds fewer cells than a stage");
    if (a.g0 < 0 || a.ng < 1 || a.g0 + a.ng > kKvHeads) throw std::runtime_error("inkling2 attention_decode: groups");
    if (a.fp8) {
        if (rows == 1) launch_split<1, true, 2>(a, rows, z, part_ml, part_acc, s);
        else if (rows == 2 || max_keys <= 2048) launch_split<2, true, 2>(a, rows, z, part_ml, part_acc, s);
        else launch_split<4, true, 1>(a, rows, z, part_ml, part_acc, s);
    } else {
        if (rows == 1) launch_split<1, false, 2>(a, rows, z, part_ml, part_acc, s);
        else if (rows == 2 || max_keys <= 2048) launch_split<2, false, 2>(a, rows, z, part_ml, part_acc, s);
        else launch_split<4, false, 1>(a, rows, z, part_ml, part_acc, s);
    }
    check("attention_decode");
    bk::launch_pdl(attn_combine_kernel, dim3((unsigned) rows, (unsigned) (kGroup * a.ng)), dim3(kHeadDim), 0, s, a, z, (const float*) part_ml,
                   (const float*) part_acc, out);
    check("attention_decode/combine");
}

void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream) {
    if (rows <= 0) return;
    decode_prepare(pa, rows, stream);
    attention_decode(a, rows, max_keys, scratch, nullptr, stream);
}

}  // namespace granite::models::inkling2
