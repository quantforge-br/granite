// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of four warps, at most
// kMaxSplits slices), so a row's bits do not depend on the other rows of its call or on the launch's bound. A block takes
// two rows, one key/value head (its three query heads) and one slice, four warps a row. The block stages a pass of the
// slice's keys in shared memory once for both rows (consecutive rows' slices start at most kSlack keys apart), then the
// pass's values over them; rows that cannot share (another conversation's cache, or slices further apart) stage in turn.
// Each warp walks its tiles of 32 keys with a lane a key, the lane scoring its key's row from the stage, and the lanes
// then sum the values with four channels a lane. A second kernel merges a row's slices with the sink, gates the heads and rounds
// them to the q8_1 blocks of the output projection.
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
constexpr int kMaxSplits = 64;
constexpr int kRows = 2;                     // rows a block takes
constexpr int kThreads = kRows * kWarps * 32;
constexpr int kSlack = 8;                    // how far apart two rows' slices may start and still share a stage
constexpr int kStage = kUnit + kSlack;       // key rows a stage holds
constexpr int kChunks = kHeadDim * 2 / 16;   // 16-byte chunks of a row in shared memory (fp16)

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

// 16 bytes global -> shared, asynchronously
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src));
}

// 8 cache values as fp32 from 16 bytes of fp16 (or the first 8 of 16 bytes of E4M3, `hi` picking the other 8)
__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void unpack16_fp8(const uint4 u, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = (__nv_fp8x2_storage_t) (w[i] & 0xFFFF);
        hi.__x = (__nv_fp8x2_storage_t) (w[i] >> 16);
        const float2 a = (float2) lo, b = (float2) hi;
        f[4 * i] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}

struct Smem {
    float q[kRows][kGroup][kHeadDim];
    float rs[kRows][kGroup][kRel];
    float m[kRows][kWarps][kGroup], l[kRows][kWarps][kGroup];
    // a pass: its key rows from row kSlack on (chunks swizzled), then its value rows as fp16 from row 0 (the first kSlack
    // copied with the keys, which no key row covers); then the warps' sums
    uint4 kv[(kSlack + kStage) * kChunks];
};
static_assert(kRows == 2, "a block's rows share one stage or take two");
static_assert(sizeof(float) * kRows * kWarps * kGroup * kHeadDim <= sizeof(uint4) * (kSlack + kStage) * kChunks,
              "the sums fit the stage");

// A row's slice: its keys [lo, hi) and caches; on when the row has this slice.
struct RowSlice {
    bool on = false;
    int p = 0, lo = 0, hi = 0;
    const uint8_t* k = nullptr;
    const uint8_t* v = nullptr;
};

template <bool FP8>
__global__ void __launch_bounds__(kThreads, 2) attn_split_kernel(AttnArgs a, int rows, int z, float* __restrict__ part_ml,
                                                                 float* __restrict__ part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int r0 = blockIdx.x * kRows, g = a.g0 + blockIdx.y, s = blockIdx.z;
    const int slot = threadIdx.x >> 7, w = (threadIdx.x >> 5) & (kWarps - 1), lane = threadIdx.x & 31;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    constexpr int kRowBytes = FP8 ? kHeadDim : kHeadDim * 2;
    constexpr size_t kCell = (size_t) kKvHeads * kRowBytes;                // bytes of a cache cell
    constexpr int kLoads = kRowBytes / 16;                                  // 16-byte chunks of a cache row
    constexpr int kKeyAt = kSlack * kChunks;                                // the stage's first key chunk
    constexpr int kStep = kThreads / kLoads;                                // stage rows between a thread's chunks
    constexpr int kVEarly = FP8 ? 0 : kSlack;                               // value rows copied with the keys
    constexpr int kPerK = (kStage + kStep - 1) / kStep;                     // a thread's key chunks of a pass
    constexpr int kPerV = (kStage - kVEarly + kStep - 1) / kStep;           // a thread's value chunks in registers
    static_assert(kVEarly <= kStep, "a thread copies at most one early value chunk");
    RowSlice rk[kRows];   // the block's rows, by slot
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
        const int r = r0 + k;
        if (r >= rows) continue;
        int p, lo, n;
        row_keys(a, r, p, lo, n);
        const int span = span_of(n);
        if (s * span >= n) continue;   // past this row's slices
        rk[k].on = true;
        rk[k].p = p;
        rk[k].lo = lo + s * span;
        rk[k].hi = min(p + 1, rk[k].lo + span);
        const SegCaches& sc = a.seg.c[bk::seg_of(a.seg.at, r)];   // a batch window's row: its conversation's caches
        rk[k].k = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
        rk[k].v = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
    }
    if (!rk[0].on && !rk[1].on) return;
    const bool share = rk[0].on && rk[1].on && rk[0].k == rk[1].k && rk[0].v == rk[1].v && abs(rk[0].lo - rk[1].lo) <= kSlack;
    // the stages: both rows' slices at once when they share, else a row's at a time; a stage's pass t holds the keys
    // [lo + t * kUnit, + kUnit + dl) below hi, pass t of each of its rows (kUnit keys from the row's own start, dl the
    // rows' starts apart)
    const int n_stages = share ? 1 : (int) rk[0].on + (int) rk[1].on;
    int lo, hi, dl, passes;
    const uint8_t *kb, *vb;
    const auto stage = [&](int q) {
        const bool one = !share && (q == 0 && rk[0].on);   // the stage is row 0 alone
        const RowSlice x = one ? rk[0] : rk[1], y = share ? rk[0] : x;
        lo = min(x.lo, y.lo);
        hi = max(x.hi, y.hi);
        dl = abs(x.lo - y.lo);
        passes = max(x.hi - x.lo, y.hi - y.lo);
        passes = (passes + kUnit - 1) / kUnit;
        kb = x.k;
        vb = x.v;
        return one ? 0 : share ? -1 : 1;   // the stage's row, -1 both
    };
    // a pass, the keys [b, e) of (kb, vb): the key rows and the early value rows copied to the stage, the other value rows
    // into registers (zero past e). A thread takes one chunk of every kStep-th row. The stage's rows past e keep what they
    // held: a lane whose key is past its row's end is masked, and a value past it read as zero.
    const int j0 = threadIdx.x / kLoads, ch = threadIdx.x % kLoads;
    const int cap = (int) a.cap;
    uint4 vr[kPerV];
    const auto load = [&](int b, int e) {
        const int c0 = (b + j0) % cap;
        const auto cell = [&](int j) { const int c = c0 + j; return (size_t) (c >= cap ? c - cap : c) * kCell + ch * 16; };
        if (j0 < kVEarly && b + j0 < e) cp16(&sm.kv[j0 * kChunks + ch], vb + cell(0));
#pragma unroll
        for (int i = 0; i < kPerK || i < kPerV; ++i) {
            const int j = kStep * i;   // past j0
            if (i < kPerK && j0 + j < kStage) {
                const int jj = j0 + j;
                if (b + jj < e) cp16(&sm.kv[kKeyAt + jj * kLoads + (ch ^ (jj & (kLoads - 1)))], kb + cell(j));
            }
            if (i < kPerV && j0 + kVEarly + j < kStage)
                vr[i] = b + j0 + kVEarly + j < e ? *reinterpret_cast<const uint4*>(vb + cell(kVEarly + j)) : make_uint4(0, 0, 0, 0);
        }
    };
    int st_row = stage(0);
    load(lo, min(hi, lo + kUnit + dl));
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
        if (!rk[k].on) continue;
        const int r = r0 + k;
        for (int i = threadIdx.x; i < kGroup * kHeadDim; i += kThreads)
            sm.q[k][i / kHeadDim][i % kHeadDim] = a.q[((size_t) r * kHeads + g * kGroup) * kHeadDim + i];
        for (int i = threadIdx.x; i < kGroup * kRel; i += kThreads)
            sm.rs[k][i / kRel][i % kRel] = a.r[((size_t) r * kHeads + g * kGroup) * kRel + i];
    }
    const RowSlice me = slot ? rk[1] : rk[0];
    float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
    }
    for (int q = 0; q < n_stages; ++q) {
        if (q > 0) st_row = stage(q);
        const bool mine = me.on && (st_row < 0 || st_row == slot);
        for (int t = 0; t < passes; ++t) {
            if (q > 0 || t > 0) {
                __syncthreads();   // the last pass's values read
                load(lo + t * kUnit, min(hi, lo + t * kUnit + kUnit + dl));
            }
            const int stage_lo = lo + t * kUnit;
            asm volatile("cp.async.wait_all;\n" ::: "memory");
            __syncthreads();
            const int t0 = me.lo + t * kUnit + w * kTile, key = t0 + lane, jt = t0 - stage_lo + lane;
            const bool tile = mine && t0 < me.hi, valid = key < me.hi;
            float sc[kGroup] = {0.f, 0.f, 0.f};
            if (tile) {
                // the lane's key scored for the three heads
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    const uint4 kc = sm.kv[kKeyAt + jt * kLoads + (i ^ (jt & (kLoads - 1)))];
                    if constexpr (FP8) {
                        float f[16];
                        unpack16_fp8(kc, f);
#pragma unroll
                        for (int d = 0; d < 16; ++d)
#pragma unroll
                            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[slot][h][16 * i + d], f[d], sc[h]);
                    } else {
                        float f[8];
                        unpack8_f16(kc, f);
#pragma unroll
                        for (int d = 0; d < 8; ++d)
#pragma unroll
                            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[slot][h][8 * i + d], f[d], sc[h]);
                    }
                }
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
            }
            __syncthreads();   // the keys read: the stage takes the values as fp16 (E4M3 is exact in fp16)
#pragma unroll
            for (int i = 0; i < kPerV; ++i) {
                const int j = j0 + kVEarly + kStep * i;
                if (j >= kStage) continue;
                if constexpr (FP8) {
                    float f[16];
                    unpack16_fp8(vr[i], f);
                    uint32_t hw[8];
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const __half2 hv = __floats2half2_rn(f[2 * d], f[2 * d + 1]);
                        hw[d] = *reinterpret_cast<const uint32_t*>(&hv);
                    }
                    sm.kv[j * kChunks + 2 * ch] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                    sm.kv[j * kChunks + 2 * ch + 1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
                } else {
                    sm.kv[j * kChunks + ch] = vr[i];
                }
            }
            __syncthreads();
            if (!tile) continue;
            // the online softmax over the tile
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
            // the tile's values; past the row's end a key's weight is zero and its value read as zero
            const int jn = me.hi - t0;
            const uint32_t* vw = reinterpret_cast<const uint32_t*>(sm.kv) + (size_t) (t0 - stage_lo) * (kHeadDim / 2) + lane * 2;
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
    // each row's four warps' parts of this slice, merged in warp order
    __syncthreads();   // the values read: the stage takes the sums
    float* sacc = reinterpret_cast<float*>(sm.kv);   // [kRows][kWarps][kGroup][kHeadDim]
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

int launch_splits(int64_t max_keys) {
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit));
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

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const int z = launch_splits(max_keys);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * z * 2;
    static bool attr[kOptInDevices] = {};   // the opt-in is per device
    if (bool& set = attr[opt_in_device()]; !set) {
        cudaFuncSetAttribute(attn_split_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_split_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        set = true;
    }
    if (a.cap < kStage) throw std::runtime_error("inkling2 attention_decode: the cache holds fewer cells than a stage");
    if (a.g0 < 0 || a.ng < 1 || a.g0 + a.ng > kKvHeads) throw std::runtime_error("inkling2 attention_decode: groups");
    const dim3 grid((unsigned) ((rows + kRows - 1) / kRows), (unsigned) a.ng, (unsigned) z);
    if (a.fp8) bk::launch_pdl(attn_split_kernel<true>, grid, dim3(kThreads), sizeof(Smem), s, a, rows, z, part_ml, part_acc);
    else bk::launch_pdl(attn_split_kernel<false>, grid, dim3(kThreads), sizeof(Smem), s, a, rows, z, part_ml, part_acc);
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

}  // namespace basalt::models::inkling2
