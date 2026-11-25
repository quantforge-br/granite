// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention on Ampere Tensor Cores (mma.sync.m16n8k16):
// a query head's scores take a per-query relative bias over the last kExtent positions (uniformly skipped on far tiles),
// and a per-query sink logit sits in the softmax's denominator.
//
// Each row's visible keys [lo, p] are covered by 32-key aligned chunks [c_lo, c_hi), split into at most kMaxSplits
// contiguous slices whose size comes from the row's own chunk count alone — so a row's bits are strictly invariant to
// window width (T=1..8), launch max_keys bound, far-apart rows, and batch-window grouping.
//
// A block takes up to 5 rows (kMTiles=1, M=16 MMA rows) or up to 8 rows (kMTiles=2, M=32 MMA rows reusing the same
// shared K/V tile once per chunk), one KV head (its 3 query heads) and one split, using 4 warps (128 threads):
//   - Register prefetch of next 32-key chunk overlapped with current chunk's HMMA compute
//   - S = Q K^T on mma.sync.m16n8k16 (FP16 in, FP32 acc), O = O * alpha + P V on mma.sync.m16n8k16 (FP16 P/V, FP32 acc)
//   - Branchless E4M3 FP8 unpack via bit-shift + non-FTZ HMUL2 (2^8)
//   - KvInt8 g64 dequant (int8 * per-64-dim FP16 scale) ready for --kv int8
//   - Combine kernel merges slices with per-query sink, sigmoid gate, and q8_1 epilogue.
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
#include <type_traits>

namespace granite::models::inkling2 {
namespace {

namespace bk = granite::kernels;

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;        // 128 threads/block
constexpr int kChunk = 32;                   // 32 keys per HMMA tile
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads per KV head
constexpr int kMaxSplits = 64;
constexpr int kLd = kHeadDim + 8;            // 136 halves: ldmatrix without bank conflicts
constexpr float kLog2e = 1.4426950408889634f;

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

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Chunks of kChunk=32 keys per slice, determined strictly by the row's own chunk count nc.
__host__ __device__ __forceinline__ int chunk_span_of(int nc) {
    const int sp = (nc + kMaxSplits - 1) / kMaxSplits;
    return sp > 1 ? sp : 1;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& c_lo, int& c_hi) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    c_lo = lo / kChunk;
    c_hi = (p + kChunk) / kChunk;
}

template <int kMTiles>
constexpr int kMaxRowsBlock = (kMTiles == 1) ? 5 : 8;

struct RowSlice {
    bool on;
    int p, lo, sc_lo, sc_hi;
    const uint8_t* k;
    const uint8_t* v;
    const uint16_t* ks;
    const uint16_t* vs;
};

struct StageGroup {
    int sc_lo, passes, p_min, c_hi_max;
    const uint8_t* kb;
    const uint8_t* vb;
    const uint16_t* ksb;
    const uint16_t* vsb;
};

template <int kMTiles>
struct SmemMma {
    static constexpr int M = kMTiles * 16;
    static constexpr int R = kMaxRowsBlock<kMTiles>;
    __half qs[M][kLd];
    __half ks[kChunk][kLd];
    __half vs[kChunk][kLd];
    float ss[M][kChunk + 1];
    __half ps[M][kChunk + 8];
    float rs[M][kRel];
    float rm[M], rl[M], ra[M];
    RowSlice rk[R];
    StageGroup sg[R];
    int grp[R];
    int n_stages;
};

template <int R>
__device__ __forceinline__ int partition_stages_mma(const RowSlice* rk, int n_rows, int* grp, StageGroup* sg) {
    int n_stages = 0;
    for (int k = 0; k < n_rows; ++k) {
        grp[k] = -1;
        if (!rk[k].on) continue;
        int found = -1;
        for (int g = 0; g < n_stages; ++g) {
            if (sg[g].kb == rk[k].k && sg[g].vb == rk[k].v && sg[g].ksb == rk[k].ks && sg[g].vsb == rk[k].vs &&
                sg[g].sc_lo == rk[k].sc_lo) {
                found = g;
                sg[g].p_min = min(sg[g].p_min, rk[k].p);
                sg[g].c_hi_max = max(sg[g].c_hi_max, rk[k].sc_hi);
                break;
            }
        }
        if (found < 0) {
            found = n_stages++;
            sg[found].sc_lo = rk[k].sc_lo;
            sg[found].p_min = rk[k].p;
            sg[found].c_hi_max = rk[k].sc_hi;
            sg[found].kb = rk[k].k;
            sg[found].vb = rk[k].v;
            sg[found].ksb = rk[k].ks;
            sg[found].vsb = rk[k].vs;
        }
        grp[k] = found;
    }
    for (int g = 0; g < n_stages; ++g) sg[g].passes = sg[g].c_hi_max - sg[g].sc_lo;
    return n_stages;
}

// Convert 8 E4M3 bytes (uint2) to 8 fp16 values (uint4) via branchless shift + non-FTZ HMUL2.
__device__ __forceinline__ uint4 e4m3x8_to_h2x4(uint2 u) {
    __half2 h0, h1, h2, h3;
    e4m3x4_to_h2x2_fast(u.x, h0, h1);
    e4m3x4_to_h2x2_fast(u.y, h2, h3);
    uint4 out;
    out.x = *reinterpret_cast<const uint32_t*>(&h0);
    out.y = *reinterpret_cast<const uint32_t*>(&h1);
    out.z = *reinterpret_cast<const uint32_t*>(&h2);
    out.w = *reinterpret_cast<const uint32_t*>(&h3);
    return out;
}

// Dequantize 8 int8 codes (uint2) * fp16 scale into 8 fp16 values (uint4).
__device__ __forceinline__ uint4 i8x8_to_h2x4(uint2 u, uint16_t sc_h16) {
    const float sc = __half2float(__ushort_as_half(sc_h16));
    int8_t b[8];
    *reinterpret_cast<uint2*>(b) = u;
    __half2 h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) h[e] = __floats2half2_rn((float) b[2 * e] * sc, (float) b[2 * e + 1] * sc);
    return *reinterpret_cast<const uint4*>(h);
}

// kMode: 0 = FP16, 1 = FP8 E4M3, 2 = KvInt8 g64.
template <int kMode, int kMTiles, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    attn_split_mma_kernel(AttnArgs a, int rows, int z, float* __restrict__ part_ml, float* __restrict__ part_acc) {
    constexpr bool kEightBit = (kMode != 0);
    constexpr int R = kMaxRowsBlock<kMTiles>;
    constexpr int M = kMTiles * 16;
    constexpr int NT = kHeadDim / 32;   // 4 N-tiles of 8 dims per warp
    constexpr int NI = (kChunk * kHeadDim / 8) / kThreads;   // 4 x 8-element groups per thread
    static_assert(NI == 4, "kChunk=32, kHeadDim=128, kThreads=128 -> 4 loads per thread");

    __shared__ __align__(16) SmemMma<kMTiles> sm;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int r0 = (int) blockIdx.x * R;
    const int n_rows = min(R, rows - r0);
    const int kvh = a.g0 + (int) blockIdx.y;
    const int s = (int) blockIdx.z;

    bk::pdl_wait();
    bk::pdl_launch_dependents();

    constexpr int kRowBytes = kEightBit ? kHeadDim : kHeadDim * 2;
    constexpr size_t kCell = (size_t) kKvHeads * kRowBytes;
    constexpr size_t kScaleCell = (size_t) kKvHeads * (kHeadDim / 64);
    const int cap = (int) a.cap;

    if (tid == 0) {
        for (int k = 0; k < n_rows; ++k) {
            sm.rk[k] = RowSlice{};
            const int r = r0 + k;
            int p, lo, c_lo, c_hi;
            row_keys(a, r, p, lo, c_lo, c_hi);
            const int nc = c_hi - c_lo;
            const int c_span = chunk_span_of(nc);
            if (s * c_span >= nc) continue;
            sm.rk[k].on = true;
            sm.rk[k].p = p;
            sm.rk[k].lo = lo;
            sm.rk[k].sc_lo = c_lo + s * c_span;
            sm.rk[k].sc_hi = min(c_hi, sm.rk[k].sc_lo + c_span);
            const SegCaches& sc = a.seg.c[bk::seg_of(a.seg.at, r)];
            sm.rk[k].k = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) kvh * kRowBytes;
            sm.rk[k].v = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) kvh * kRowBytes;
            if constexpr (kMode == 2) {
                sm.rk[k].ks = (a.seg.at.n > 0 ? sc.k_scale : a.k_scale) + (size_t) kvh * (kHeadDim / 64);
                sm.rk[k].vs = (a.seg.at.n > 0 ? sc.v_scale : a.v_scale) + (size_t) kvh * (kHeadDim / 64);
            } else {
                sm.rk[k].ks = nullptr;
                sm.rk[k].vs = nullptr;
            }
        }
        sm.n_stages = partition_stages_mma<R>(sm.rk, n_rows, sm.grp, sm.sg);
    }
    __syncthreads();
    const int n_stages = sm.n_stages;
    if (n_stages == 0) return;

    // Load queries (as fp16) and relative states (as fp32) for all active (row, head) pairs into smem.
    const int M_act = n_rows * kGroup;
    for (int idx = tid; idx < M * kHeadDim; idx += kThreads) {
        const int m = idx / kHeadDim, d = idx - m * kHeadDim;
        const int slot = m / kGroup, h = m - slot * kGroup;
        float v = 0.f;
        if (m < M_act && sm.rk[slot].on) v = a.q[((size_t) (r0 + slot) * kHeads + kvh * kGroup + h) * kHeadDim + d];
        sm.qs[m][d] = __float2half_rn(v);
    }
    for (int idx = tid; idx < M_act * kRel; idx += kThreads) {
        const int m = idx / kRel, ri = idx - m * kRel;
        const int slot = m / kGroup, h = m - slot * kGroup;
        sm.rs[m][ri] = sm.rk[slot].on ? a.r[((size_t) (r0 + slot) * kHeads + kvh * kGroup + h) * kRel + ri] : 0.f;
    }
    if (tid < M) {
        sm.rm[tid] = -INFINITY;
        sm.rl[tid] = 0.f;
        sm.ra[tid] = 1.f;
    }
    for (int idx = tid; idx < (M - M_act) * kChunk; idx += kThreads) {
        const int m = M_act + idx / kChunk, cc = idx - (m - M_act) * kChunk;
        sm.ps[m][cc] = __float2half_rn(0.f);
    }

    float acc[kMTiles][NT][4];
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[mt][n][0] = acc[mt][n][1] = acc[mt][n][2] = acc[mt][n][3] = 0.f;

    using Word = std::conditional_t<kEightBit, uint2, uint4>;
    Word pk[NI], pv[NI];
    uint16_t psk[NI], psv[NI];

    const auto fetch = [&](const StageGroup& st, int c0, int key_hi) {
        const int cell0 = c0 % cap;
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int idx = tid + j * kThreads;
            const int cell = idx >> 4;          // idx / (128 / 8) = idx / 16
            const int d8 = (idx & 15) * 8;
            if (c0 + cell < key_hi) {
                int ce = cell0 + cell;
                if (ce >= cap) ce -= cap;
                const size_t off = (size_t) ce * kCell + (size_t) d8 * (kEightBit ? 1 : 2);
                pk[j] = *reinterpret_cast<const Word*>(st.kb + off);
                pv[j] = *reinterpret_cast<const Word*>(st.vb + off);
                if constexpr (kMode == 2) {
                    const size_t soff = (size_t) ce * kScaleCell + (size_t) (d8 >> 6);
                    psk[j] = st.ksb[soff];
                    psv[j] = st.vsb[soff];
                }
            } else {
                if constexpr (kEightBit) pk[j] = pv[j] = make_uint2(0u, 0u);
                else pk[j] = pv[j] = make_uint4(0u, 0u, 0u, 0u);
                if constexpr (kMode == 2) psk[j] = psv[j] = 0u;
            }
        }
    };

    const auto commit = [&]() {
#pragma unroll
        for (int j = 0; j < NI; ++j) {
            const int idx = tid + j * kThreads;
            const int cell = idx >> 4;
            const int d8 = (idx & 15) * 8;
            uint4 kw, vw;
            if constexpr (kMode == 1) {
                kw = e4m3x8_to_h2x4(pk[j]);
                vw = e4m3x8_to_h2x4(pv[j]);
            } else if constexpr (kMode == 2) {
                kw = i8x8_to_h2x4(pk[j], psk[j]);
                vw = i8x8_to_h2x4(pv[j], psv[j]);
            } else {
                kw = pk[j];
                vw = pv[j];
            }
            *reinterpret_cast<uint4*>(&sm.ks[cell][d8]) = kw;
            *reinterpret_cast<uint4*>(&sm.vs[cell][d8]) = vw;
        }
    };

    for (int qg = 0; qg < n_stages; ++qg) {
        const StageGroup st = sm.sg[qg];
        const int key_hi_g = st.c_hi_max * kChunk;
        if (qg > 0) __syncthreads();
        fetch(st, st.sc_lo * kChunk, key_hi_g);
        for (int t = 0; t < st.passes; ++t) {
            const int chunk_id = st.sc_lo + t;
            const int c0 = chunk_id * kChunk;
            if (t > 0) __syncthreads();
            commit();
            __syncthreads();
            if (t + 1 < st.passes) fetch(st, (chunk_id + 1) * kChunk, key_hi_g);

            // S = Q K^T: warp w computes keys [8*w .. 8*w + 7] for all kMTiles.
#pragma unroll
            for (int mt = 0; mt < kMTiles; ++mt) {
                float s4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int k = 0; k < kHeadDim; k += 16) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldsm_x4(a0, a1, a2, a3, &sm.qs[mt * 16 + (lane & 15)][k + (lane >> 4) * 8]);
                    ldsm_x2(b0, b1, &sm.ks[warp * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                    mma16816(s4, a0, a1, a2, a3, b0, b1);
                }
                sm.ss[mt * 16 + g][warp * 8 + 2 * c] = s4[0];
                sm.ss[mt * 16 + g][warp * 8 + 2 * c + 1] = s4[1];
                sm.ss[mt * 16 + g + 8][warp * 8 + 2 * c] = s4[2];
                sm.ss[mt * 16 + g + 8][warp * 8 + 2 * c + 1] = s4[3];
            }
            __syncthreads();

            // Uniform far-tile check: skip kRel=16 relative bias on ~99.5% of chunks at 243k.
            const bool near = (st.p_min - (c0 + kChunk - 1)) < kExtent;
            for (int m = warp; m < M_act; m += kWarps) {
                const int slot = m / kGroup;
                const bool active = sm.grp[slot] == qg && chunk_id < sm.rk[slot].sc_hi;
                if (!active) {
                    sm.ps[m][lane] = __float2half_rn(0.f);
                    if (lane == 0) sm.ra[m] = 1.f;
                    continue;
                }
                const int rp = sm.rk[slot].p;
                const int rlo = sm.rk[slot].lo;
                const int pos = c0 + lane;
                const bool valid = pos >= rlo && pos <= rp;
                float sc = -INFINITY;
                if (valid) {
                    sc = sm.ss[m][lane] * a.scale;
                    if (near) {
                        const int dist = rp - pos;
                        if (dist < kExtent) {
                            float b = 0.f;
#pragma unroll
                            for (int ri = 0; ri < kRel; ++ri) b = fmaf(sm.rs[m][ri], a.rel_table[ri * kExtent + dist], b);
                            sc += b;
                        }
                    }
                }
                const float mx = warp_max(sc);
                const float mold = sm.rm[m];
                const float mnew = fmaxf(mold, mx);
                const float p = (sc == -INFINITY) ? 0.f : exp2f((sc - mnew) * kLog2e);
                sm.ps[m][lane] = __float2half_rn(p);
                const float lsum = warp_sum(p);
                if (lane == 0) {
                    const float alpha = (mold == -INFINITY) ? 0.f : exp2f((mold - mnew) * kLog2e);
                    sm.ra[m] = alpha;
                    sm.rl[m] = fmaf(sm.rl[m], alpha, lsum);
                    sm.rm[m] = mnew;
                }
            }
            __syncthreads();

            // O = O * alpha + P V: warp w owns dims [32*w .. 32*w + 31].
#pragma unroll
            for (int mt = 0; mt < kMTiles; ++mt) {
                const float ag = sm.ra[mt * 16 + g], ag8 = sm.ra[mt * 16 + g + 8];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    acc[mt][n][0] *= ag;
                    acc[mt][n][1] *= ag;
                    acc[mt][n][2] *= ag8;
                    acc[mt][n][3] *= ag8;
                }
#pragma unroll
                for (int kk = 0; kk < kChunk; kk += 16) {
                    unsigned a0, a1, a2, a3;
                    ldsm_x4(a0, a1, a2, a3, &sm.ps[mt * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        unsigned b0, b1;
                        ldsm_x2_t(b0, b1, &sm.vs[kk + (lane & 15)][warp * 32 + n * 8]);
                        mma16816(acc[mt][n], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }

    // Write per-slice partials directly from registers (no intra-block reduction needed).
#pragma unroll
    for (int mt = 0; mt < kMTiles; ++mt) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int m = mt * 16 + g + 8 * hr;
            if (m >= M_act) continue;
            const int slot = m / kGroup, h = m - slot * kGroup;
            if (!sm.rk[slot].on) continue;
            const int r = r0 + slot;
            const size_t sl = ((size_t) r * kHeads + kvh * kGroup + h) * z + s;
            if (warp == 0 && c == 0) {
                part_ml[sl * 2] = sm.rm[m];
                part_ml[sl * 2 + 1] = sm.rl[m];
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int d = warp * 32 + n * 8 + 2 * c;
                part_acc[sl * kHeadDim + d] = acc[mt][n][2 * hr];
                part_acc[sl * kHeadDim + d + 1] = acc[mt][n][2 * hr + 1];
            }
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
    int p, lo, c_lo, c_hi;
    row_keys(a, r, p, lo, c_lo, c_hi);
    const int nc = c_hi - c_lo;
    const int c_span = chunk_span_of(nc), splits = (nc + c_span - 1) / c_span;
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
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kChunk - 1) / kChunk + 1));
}

template <int kMode, int kMTiles, int kMinBlocks>
void launch_split_mma(const AttnArgs& a, int rows, int z, float* part_ml, float* part_acc, cudaStream_t s) {
    constexpr int R = kMaxRowsBlock<kMTiles>;
    const dim3 grid((unsigned) ((rows + R - 1) / R), (unsigned) a.ng, (unsigned) z);
    bk::launch_pdl(attn_split_mma_kernel<kMode, kMTiles, kMinBlocks>, grid, dim3(kThreads), 0, s,
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
    if (a.cap < kChunk) throw std::runtime_error("inkling2 attention_decode: the cache holds fewer cells than a chunk");
    if (a.g0 < 0 || a.ng < 1 || a.g0 + a.ng > kKvHeads) throw std::runtime_error("inkling2 attention_decode: groups");
    if (a.seg.at.n > 0 && rows > kDecodeRows)
        throw std::runtime_error("inkling2 attention_decode: rows > kDecodeRows with batch segments");
    const bool i8 = a.seg.at.n > 0 ? (a.seg.c[0].k_scale != nullptr) : (a.k_scale != nullptr);
    if (i8) {
        if (rows <= 5) launch_split_mma<2, 1, 3>(a, rows, z, part_ml, part_acc, s);
        else launch_split_mma<2, 2, 2>(a, rows, z, part_ml, part_acc, s);
    } else if (a.fp8) {
        if (rows <= 5) launch_split_mma<1, 1, 3>(a, rows, z, part_ml, part_acc, s);
        else launch_split_mma<1, 2, 2>(a, rows, z, part_ml, part_acc, s);
    } else {
        if (rows <= 5) launch_split_mma<0, 1, 3>(a, rows, z, part_ml, part_acc, s);
        else launch_split_mma<0, 2, 2>(a, rows, z, part_ml, part_acc, s);
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
