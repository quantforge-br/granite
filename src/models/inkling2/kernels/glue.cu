// src/models/inkling2/kernels/glue.cu - Inkling 2's fused glue for a decode window: the block-residual mixes with the norm
// and the q8_1 rounding of the products' inputs, the loop controllers (down, up, blend, norm and rounding), the attention
// preparation a key/value head a block, the attention output's convolution with the residual and the MLP's mix, and the
// experts' sum with its convolution and the residual.
//
// A block takes one row (or one row and head, or one stream and row), and no row's arithmetic reads another row's partial
// sums, so a row's bits do not depend on the rows beside it. A window's convolutions read the rows before them in the
// window and the ring for the positions before the window, and each row writes its own ring slot: a window of at most
// kDecodeRows rows touches distinct slots (kDecodeRows + kConv - 1 <= kConvRing).
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/prefetch_l2.cuh"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "prep.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

static_assert(kDecodeRows + kConv - 1 <= kConvRing, "a window's rows and the positions they read fit the ring");

constexpr int kThreads = 768;
constexpr int kWarpsB = kThreads / 32;
constexpr int kPer = kWidth / kThreads;   // 3 values a thread: thread t holds t, t + 768, t + 1536
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "a q8_1 block is 36 bytes");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// The warp's totals of the 12 values each lane holds, each summed by warp_sum's tree (lanes 16, 8, 4, 2, then 1 apart
// added pairwise): at each step a lane adds, from its partner, only the values its side of the pair goes on with, so the
// warp shuffles 13 times for the 12 totals. Every lane ends with the total of value 6 b4 + 3 b3 + (b2 ? 2 : b1), b the bits of
// its lane index; every lane holding a value holds warp_sum's bits of it.
__device__ __forceinline__ float warp_totals12(const float (&p)[12], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
    float q[6], r[3];
#pragma unroll
    for (int k = 0; k < 6; ++k) q[k] = (b4 ? p[6 + k] : p[k]) + __shfl_xor_sync(0xffffffffu, b4 ? p[k] : p[6 + k], 16);
#pragma unroll
    for (int k = 0; k < 3; ++k) r[k] = (b3 ? q[3 + k] : q[k]) + __shfl_xor_sync(0xffffffffu, b3 ? q[k] : q[3 + k], 8);
    // lanes 4 apart: the side with b2 clear goes on with values 0 and 1 of its three, the other side with value 2
    const float g1 = __shfl_xor_sync(0xffffffffu, b2 ? r[0] : r[2], 4), g2 = __shfl_xor_sync(0xffffffffu, r[1], 4);
    const float t0 = (b2 ? r[2] : r[0]) + g1, t1 = r[1] + g2;
    // lanes 2 apart: with b2 clear each side goes on with value b1 of its two
    const float u = (b2 || !b1 ? t0 : t1) + __shfl_xor_sync(0xffffffffu, b2 || b1 ? t0 : t1, 2);
    return u + __shfl_xor_sync(0xffffffffu, u, 1);
}

// The q8_1 blocks w, w + 24 and w + 48 of a row (warp w: lane l holds value l of each, y[k] for block w + 24 k): each
// block's amax and sum reduced together across the warp as warp_totals12 does (the lanes with b4 set take the maxima,
// exact in any order; the others the sums, by warp_sum's tree), 8 shuffles, then the maxima handed to every lane (3); d =
// amax / 127, the sum stored by the lane that holds it.
__device__ __forceinline__ void q8_store3(const float (&y)[kPer], Q81* row, int w, int lane) {
    static_assert(kPer == 3, "a lane holds one value of each of three blocks");
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float a[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float o = __shfl_xor_sync(0xffffffffu, b4 ? y[k] : fabsf(y[k]), 16);
        a[k] = b4 ? fmaxf(fabsf(y[k]), o) : y[k] + o;
    }
    // lanes 8 apart: the side with b3 clear goes on with blocks 0 and 1, the other side with block 2
    const float g1 = __shfl_xor_sync(0xffffffffu, b3 ? a[0] : a[2], 8), g2 = __shfl_xor_sync(0xffffffffu, a[1], 8);
    const float s0 = b3 ? a[2] : a[0];
    const float t0 = b4 ? fmaxf(s0, g1) : s0 + g1, t1 = b4 ? fmaxf(a[1], g2) : a[1] + g2;
    // lanes 4 apart: with b3 clear each side goes on with block b2 of its two
    const float k3 = b3 || !b2 ? t0 : t1, o3 = __shfl_xor_sync(0xffffffffu, b3 || b2 ? t0 : t1, 4);
    float u = b4 ? fmaxf(k3, o3) : k3 + o3;
    const float o4 = __shfl_xor_sync(0xffffffffu, u, 2);
    u = b4 ? fmaxf(u, o4) : u + o4;
    const float o5 = __shfl_xor_sync(0xffffffffu, u, 1);
    u = b4 ? fmaxf(u, o5) : u + o5;
    // block k's amax at lane 16 + 4 k (k < 2) or 24, its sum at lane 4 k or 8
    const float amax[3] = {__shfl_sync(0xffffffffu, u, 16), __shfl_sync(0xffffffffu, u, 20), __shfl_sync(0xffffffffu, u, 24)};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float d = bk::q8_1_finite(amax[k] / 127.0f);
        Q81& b = row[w + kWarpsB * k];
        b.qs[lane] = bk::q8_1_quant(y[k], d, amax[k]);
        if (lane == (k < 2 ? 4 * k : 8)) b.ds = bk::q8_1_ds(d, u);
    }
}

// The block-residual mix of the n states v[i] (thread-held values) scored by the query qv (thread-held values, loaded
// before the kernel's pdl_wait): scores q . s / sqrt(kWidth) over the RMS-normed states (eps 1e-6), softmax, the weighted
// sum of the normed states. A state's sum of squares and score are each warp's total, then the warps' added in warp order
// by warp 0, which takes the norms and the softmax once for the block and hands them over in shared memory (`red`:
// 2 kMaxMix (kWarpsB + 1) floats).
__device__ __forceinline__ void mix_rows(const float (&v)[kMaxMix][kPer], int n, const float (&qv)[kPer], float (&mixed)[kPer],
                                         float* red) {
    static_assert(2 * kMaxMix == 12, "a state's two sums, every state's in one warp_totals12");
    float part[2 * kMaxMix];
#pragma unroll
    for (int i = 0; i < 2 * kMaxMix; ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const float x = v[i][j];
            part[2 * i] = fmaf(x, x, part[2 * i]);
            part[2 * i + 1] = fmaf(qv[j], x, part[2 * i + 1]);
        }
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float tot = warp_totals12(part, lane);
    if (!(lane & 1) && (!(lane & 4) || !(lane & 2)))   // one lane of each value
        red[(6 * ((lane >> 4) & 1) + 3 * ((lane >> 3) & 1) + ((lane & 4) ? 2 : (lane >> 1) & 1)) * kWarpsB + w] = tot;
    __syncthreads();
    float* hand = red + 2 * kMaxMix * kWarpsB;   // [kMaxMix] the softmax weights, then [kMaxMix] the norms
    if (w == 0) {
        float s = 0.f;   // lane 2 i: state i's sum of squares, lane 2 i + 1: its score
        if (lane < 2 * kMaxMix)
#pragma unroll
            for (int j = 0; j < kWarpsB; ++j) s += red[lane * kWarpsB + j];
        const float inv_l = rsqrtf(s / (float) kWidth + 1e-6f);
        // lane i (< n) takes state i: its norm and score, then the softmax over the lanes (the maximum is exact in any
        // order; the sum in state order)
        const int i2 = 2 * (lane % kMaxMix);
        const float ii = __shfl_sync(0xffffffffu, inv_l, i2), dot = __shfl_sync(0xffffffffu, s, i2 + 1);
        const bool mine = lane < n;
        const float inv = mine ? ii : 0.f;
        const float sc = mine ? dot * inv * rsqrtf((float) kWidth) : -INFINITY;
        float m = sc;
#pragma unroll
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));   // lanes 0..7: states, then -inf
        m = __shfl_sync(0xffffffffu, m, 0);
        const float e = mine ? expf(sc - m) : 0.f;
        float den = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) den += __shfl_sync(0xffffffffu, e, i);
        if (lane < kMaxMix) {
            hand[lane] = e / den;
            hand[kMaxMix + lane] = inv;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i)
            if (i < n) acc = fmaf(hand[i], v[i][j] * hand[kMaxMix + i], acc);
        mixed[j] = acc;
    }
}

// Thread t's kPer values of a kWidth vector (t, t + 768, t + 1536): a kernel loads its weights so before its pdl_wait.
__device__ __forceinline__ void load_row(const float* __restrict__ p, float (&x)[kPer]) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) x[j] = p[threadIdx.x + j * kThreads];
}

// RMS norm of the thread-held row x with the thread-held weights w, into q8_1 (and fp32 when `f32` is set): the sum of
// squares each warp's, then the warps' in warp order. The kernel's last use of `red` (kWarpsB + 1 floats).
__device__ __forceinline__ void norm_q8(const float (&x)[kPer], const float (&w)[kPer], float eps, void* q8, float* f32,
                                        int r, float* red) {
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) ss = fmaf(x[j], x[j], ss);
    ss = warp_sum(ss);
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kWarpsB; ++j) s += red[j];
        red[kWarpsB] = s;
    }
    __syncthreads();
    const float inv = rsqrtf(red[kWarpsB] / (float) kWidth + eps);
    Q81* row = reinterpret_cast<Q81*>(q8) + (size_t) r * (kWidth / 32);
    float y[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        y[j] = x[j] * inv * w[j];
        if (f32) f32[(size_t) r * kWidth + threadIdx.x + j * kThreads] = y[j];
    }
    q8_store3(y, row, wp, lane);
}

using prep::conv_in;

// ---- mixes -------------------------------------------------------------------------------------------------------------

// Block (row, output q): the mix scored by query q (or the last state), then its norm into q8_1, or the mix as it is.
// The older key/value cells of a visit's attention into L2 (MixNorm's kv_*), by a block of its own after the rows' that
// exits before the wait (so the rows' blocks reach their wait at once): its threads request the 128-byte lines of the one
// or two runs (the ring may wrap) of the keys and of the values. Up to kKvAheadCells positions: a longer run does not stay
// in L2 through the projections and is read twice.
constexpr int kKvAheadCells = 256;
__device__ __forceinline__ void kv_read_ahead(const MixNorm& a) {
    const int p0 = a.kv_pos[0];
    if (a.ring_k && threadIdx.x < 3 * 3) {   // ring entry j = 1..3 positions back of the key, value and output rings
        const int j = threadIdx.x % 3 + 1, which = threadIdx.x / 3;
        const int pj = p0 - j;
        if (pj >= 0) {
            const int w = which == 2 ? kWidth : kKvHeads * kHeadDim;
            const float* ring = which == 0 ? a.ring_k : which == 1 ? a.ring_v : a.ring_a;
            bk::prefetch_l2(ring + (size_t) (pj % kConvRing) * w, (uint64_t) w * 4);
        }
    }
    const int lo = a.kv_window > 0 ? max(0, p0 - a.kv_window + 1) : 0;
    const int n = p0 - lo;
    if (n <= 0 || n > kKvAheadCells) return;
    const int64_t c0 = lo % a.kv_cap, run1 = min((int64_t) n, a.kv_cap - c0), run2 = n - run1;
    const uint64_t b1 = (uint64_t) run1 * a.kv_cell, b2 = (uint64_t) run2 * a.kv_cell;
    constexpr uint64_t kPiece = 128;
    const uint64_t n1 = (b1 + kPiece - 1) / kPiece, n2 = (b2 + kPiece - 1) / kPiece;
    for (uint64_t j = threadIdx.x; j < 2 * (n1 + n2); j += blockDim.x) {
        const char* base = (const char*) (j < n1 + n2 ? a.kv_k : a.kv_v);
        const uint64_t i = j % (n1 + n2);
        const uint64_t start = i < n1 ? (uint64_t) c0 * a.kv_cell + i * kPiece : (i - n1) * kPiece;
        const uint64_t end = i < n1 ? (uint64_t) c0 * a.kv_cell + b1 : b2;
        bk::prefetch_l2(base + start, min(kPiece, end - start));
    }
}

__global__ void __launch_bounds__(kThreads) mix_norm_kernel(MixNorm a) {
    const int r = blockIdx.x, q = blockIdx.y, n = a.st.n;
    if (a.kv_k && r == (int) gridDim.x - 1) {   // the read-ahead's block: constant through the window, no wait
        if (q == 0) kv_read_ahead(a);
        return;
    }
    float qv[kPer], nw[kPer];   // the weights, read while the kernel before finishes
    if (a.queries) load_row(a.queries + (size_t) q * kWidth, qv);
    if (!a.mix_out) load_row(a.norm[q], nw);
    // and every state but the last: only the last one may be the kernel before's
    float v[kMaxMix][kPer];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n - 1) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = a.st.s[i][(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == n - 1)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = a.st.s[i][(size_t) r * kWidth + threadIdx.x + j * kThreads];
    float mixed[kPer];
    if (a.queries == nullptr) {   // the last state as it is (one output)
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i)
            if (i == n - 1)
#pragma unroll
                for (int j = 0; j < kPer; ++j) mixed[j] = v[i][j];
    } else {
        mix_rows(v, n, qv, mixed, red);
    }
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            a.mix_out[(size_t) q * a.out_stride + (size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[j];
        return;
    }
    norm_q8(mixed, nw, a.eps, a.q8[q], q == 0 ? a.norm_out : nullptr, r, red);
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

// hid[s][r][o] = down_s[o] . [rms(dest_s[r]) | rms(msg_s[r])]. The 2 kWidth inputs are kDownSlices slices of kSlice (the
// first half dest's, the second msg's). Each output's dot and sum of squares over a slice is lane l's sum of the pairs
// l + 32 b in order of b, then warp_sum; the slices' are added in order in each half, and each half's dot is scaled by its
// row's norm. Both kernels below add every sum in that order whatever the rows, so a row's bits do not depend on the
// others or on the kernel that took it.
constexpr int kDownSlices = 8;
constexpr int kSlice = 2 * kWidth / kDownSlices;   // 576 inputs
constexpr int kSlicePairs = kSlice / 64;           // a lane's input pairs in a slice
static_assert(kWidth % kSlice == 0 && kSlice % 64 == 0, "a slice is whole pairs of every lane, inside one half");

// A few rows: block (o0 / O, s) takes the O outputs o0.. whole, warp w slice w, and adds the slices from shared memory.
// A lane loads its input pairs of every row before their first sum, so the loads overlap; its weights before pdl_wait.
template<int O, int RMAX>
__global__ void __launch_bounds__(kDownSlices * 32) ctrl_down_whole_kernel(CtrlArgs a, int rows) {
    static_assert(RMAX <= 4, "a lane holds the input pairs of at most 4 rows (registers)");
    __shared__ float part[kDownSlices][O][RMAX][2];
    const int o0 = blockIdx.x * O, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool second = w >= kDownSlices / 2;
    const int c0 = (w % (kDownSlices / 2)) * kSlice;   // the slice's first channel in its half
    const float* in = (second ? a.msg[s] : a.dest[s]) + c0;
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(a.down[s] + (size_t) o0 * 2 * kWidth + (second ? kWidth : 0) + c0);
    uint32_t wu[kSlicePairs][O];   // output o0 + j: 2 kWidth bf16, kWidth words
#pragma unroll
    for (int b = 0; b < kSlicePairs; ++b)
#pragma unroll
        for (int j = 0; j < O; ++j) wu[b][j] = wr[(size_t) j * kWidth + lane + 32 * b];
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    float2 x[kSlicePairs][RMAX];
#pragma unroll
    for (int b = 0; b < kSlicePairs; ++b)
#pragma unroll
        for (int t = 0; t < RMAX; ++t)
            if (t < rows) x[b][t] = reinterpret_cast<const float2*>(in + (size_t) t * kWidth)[lane + 32 * b];
    float dot[O][RMAX], sq[RMAX];
#pragma unroll
    for (int t = 0; t < RMAX; ++t) {
        sq[t] = 0.f;
#pragma unroll
        for (int j = 0; j < O; ++j) dot[j][t] = 0.f;
    }
#pragma unroll
    for (int b = 0; b < kSlicePairs; ++b) {
        float w0[O], w1[O];
#pragma unroll
        for (int j = 0; j < O; ++j) {
            w0[j] = __uint_as_float(wu[b][j] << 16);
            w1[j] = __uint_as_float(wu[b][j] & 0xffff0000u);
        }
#pragma unroll
        for (int t = 0; t < RMAX; ++t) {
            if (t < rows) {
#pragma unroll
                for (int j = 0; j < O; ++j) dot[j][t] = fmaf(w1[j], x[b][t].y, fmaf(w0[j], x[b][t].x, dot[j][t]));
                sq[t] = fmaf(x[b][t].y, x[b][t].y, fmaf(x[b][t].x, x[b][t].x, sq[t]));
            }
        }
    }
#pragma unroll
    for (int t = 0; t < RMAX; ++t) {
        if (t >= rows) break;
        const float q = warp_sum(sq[t]);
#pragma unroll
        for (int j = 0; j < O; ++j) {
            const float d = warp_sum(dot[j][t]);
            if (lane == 0) {
                part[w][j][t][0] = d;
                part[w][j][t][1] = q;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < rows * O) {
        const int t = threadIdx.x % rows, j = threadIdx.x / rows;
        float dd = 0.f, sd = 0.f, dm = 0.f, sm = 0.f;
#pragma unroll
        for (int k = 0; k < kDownSlices / 2; ++k) {
            dd += part[k][j][t][0];
            sd += part[k][j][t][1];
            dm += part[kDownSlices / 2 + k][j][t][0];
            sm += part[kDownSlices / 2 + k][j][t][1];
        }
        a.scratch[((size_t) s * kDecodeRows + t) * kCtrlRank + o0 + j] =
            dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
    }
}

// More rows: block (g, w, s) takes slice w of stream s for the kDownOut outputs from g kDownOut, so a row's slice is read
// once for them (staged in shared memory) instead of once for every few outputs. Each warp takes kDownOut / WARPS
// outputs; the slices' partials go to the scratch, and the last of a group's kDownSlices blocks (counted in the group's
// counter, which it resets) adds them.
constexpr int kDownOut = 8;   // outputs a block
static_assert(kCtrlRank % kDownOut == 0, "the rank is whole groups");

// The controllers' scratch after hid [kMaxStreams][kDecodeRows][kCtrlRank] and gf [kMaxStreams][kDecodeRows][2 kWidth]:
// the slices' dots [kMaxStreams][kDownSlices][kDecodeRows][kCtrlRank], their sums of squares [kMaxStreams][kGroups]
// [kDownSlices][kDecodeRows] (each group's own copy) and the groups' counters [kMaxStreams][kGroups]: their
// offsets in floats, and the scratch's size.
struct CtrlLayout {
    size_t dot, sq, done, size;
};
constexpr int kGroups = kCtrlRank / kDownOut;   // a stream's groups
constexpr CtrlLayout kCtrl = [] {
    const size_t dot = (size_t) kMaxStreams * kDecodeRows * (kCtrlRank + 2 * kWidth);
    const size_t sq = dot + (size_t) kMaxStreams * kDownSlices * kDecodeRows * kCtrlRank;
    const size_t done = sq + (size_t) kMaxStreams * kGroups * kDownSlices * kDecodeRows;
    return CtrlLayout{dot, sq, done, done + (size_t) kMaxStreams * kGroups};
}();

template<int WARPS>
__global__ void __launch_bounds__(WARPS * 32) ctrl_down_slice_kernel(CtrlArgs a, int rows) {
    constexpr int kWarpOut = kDownOut / WARPS;   // outputs a warp
    static_assert(kDownOut % WARPS == 0 && WARPS * 32 >= kDecodeRows * kDownOut, "whole outputs a warp, a thread a sum");
    __shared__ alignas(16) float xs[kDecodeRows][kSlice];
    __shared__ bool last;
    const int g = blockIdx.x, w = blockIdx.y, s = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o0 = g * kDownOut + warp * kWarpOut;   // the warp's first output
    const bool second = w >= kDownSlices / 2;
    const int c0 = (w % (kDownSlices / 2)) * kSlice;   // the slice's first channel in its half
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(a.down[s] + (size_t) o0 * 2 * kWidth + (second ? kWidth : 0) + c0);
    uint32_t wu[kWarpOut][kSlicePairs];   // output o0 + j: 2 kWidth bf16, kWidth words; constant, so read before pdl_wait
#pragma unroll
    for (int j = 0; j < kWarpOut; ++j)
#pragma unroll
        for (int b = 0; b < kSlicePairs; ++b) wu[j][b] = wr[(size_t) j * kWidth + lane + 32 * b];
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float* in = (second ? a.msg[s] : a.dest[s]) + c0;
    for (int i = threadIdx.x; i < rows * (kSlice / 4); i += WARPS * 32) {   // every copy in flight at once
        const int t = i / (kSlice / 4), c = 4 * (i - t * (kSlice / 4));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(&xs[t][c])),
                     "l"(in + (size_t) t * kWidth + c));
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    float dot[kWarpOut][kDecodeRows];
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t)
#pragma unroll
        for (int j = 0; j < kWarpOut; ++j) dot[j][t] = 0.f;
#pragma unroll
    for (int b = 0; b < kSlicePairs; ++b) {
        float w0[kWarpOut], w1[kWarpOut];
#pragma unroll
        for (int j = 0; j < kWarpOut; ++j) {
            w0[j] = __uint_as_float(wu[j][b] << 16);
            w1[j] = __uint_as_float(wu[j][b] & 0xffff0000u);
        }
#pragma unroll
        for (int t = 0; t < kDecodeRows; ++t) {
            if (t < rows) {
                const float2 x = reinterpret_cast<const float2*>(xs[t])[lane + 32 * b];
#pragma unroll
                for (int j = 0; j < kWarpOut; ++j) dot[j][t] = fmaf(w1[j], x.y, fmaf(w0[j], x.x, dot[j][t]));
            }
        }
    }
    float* const pdot = a.scratch + kCtrl.dot;
    float* const psq = a.scratch + kCtrl.sq;
    for (int t = warp; t < rows; t += WARPS) {   // warp t % WARPS: row t's sum of squares over the slice
        float q = 0.f;
#pragma unroll
        for (int b = 0; b < kSlicePairs; ++b) {
            const float2 x = reinterpret_cast<const float2*>(xs[t])[lane + 32 * b];
            q = fmaf(x.y, x.y, fmaf(x.x, x.x, q));
        }
        q = warp_sum(q);
        if (lane == 0) psq[(((size_t) s * kGroups + g) * kDownSlices + w) * kDecodeRows + t] = q;
    }
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        if (t >= rows) break;
#pragma unroll
        for (int j = 0; j < kWarpOut; ++j) {
            const float d = warp_sum(dot[j][t]);
            if (lane == 0) pdot[(((size_t) s * kDownSlices + w) * kDecodeRows + t) * kCtrlRank + o0 + j] = d;
        }
    }
    __syncthreads();   // the block's partials are thread 0's to release
    if (threadIdx.x == 0) {
        unsigned* done = reinterpret_cast<unsigned*>(a.scratch + kCtrl.done) + (size_t) s * kGroups + g;
        unsigned prior;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(prior) : "l"(done) : "memory");
        last = prior == (unsigned) (kDownSlices - 1);
        if (last) *done = 0u;   // every slice has counted: ready for the next launch
    }
    __syncthreads();   // the other slices' partials, acquired by thread 0, are the block's
    if (!last) return;
    if (threadIdx.x < rows * kDownOut) {
        const int t = threadIdx.x % rows, o = g * kDownOut + threadIdx.x / rows;
        const float* dt = pdot + ((size_t) s * kDownSlices * kDecodeRows + t) * kCtrlRank + o;   // slice 0's, + slice
        const float* qt = psq + ((size_t) s * kGroups + g) * kDownSlices * kDecodeRows + t;
        const size_t dk = (size_t) kDecodeRows * kCtrlRank;
        constexpr int kHalf = kDownSlices / 2;
        float dd = 0.f, sd = 0.f, dm = 0.f, sm = 0.f;
#pragma unroll
        for (int k = 0; k < kHalf; ++k) {
            dd += __ldcg(dt + k * dk);
            sd += __ldcg(qt + k * kDecodeRows);
            dm += __ldcg(dt + (kHalf + k) * dk);
            sm += __ldcg(qt + (kHalf + k) * kDecodeRows);
        }
        a.scratch[((size_t) s * kDecodeRows + t) * kCtrlRank + o] =
            dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
    }
}

// gf[s][r][i] = up_s[i] . hid[s][r] for the 2 kWidth rows of up_s, kUpRows rows a block. A row's sum is eight chains of
// kUpPart values (its parts), then the parts' sums in pairs; each of a row's kUpLanes lanes (8 lanes apart in the warp)
// takes kUpLaneParts parts, adds its pairs, and the shuffles add the rest in the same pairs.
constexpr int kUpLanes = 4, kUpRows = 64;
constexpr int kUpPart = kCtrlRank / 8, kUpLaneParts = 8 / kUpLanes;
static_assert(kUpPart % 8 == 0 && 8 * kUpLanes == 32 && 2 * kWidth % kUpRows == 0 && kUpRows * kUpLanes % 32 == 0,
              "whole 16-byte words a part, a warp's 8 rows, whole blocks and warps");
__global__ void __launch_bounds__(kUpRows * kUpLanes) ctrl_up_kernel(CtrlArgs a, int rows) {
    __shared__ alignas(16) float hs[kDecodeRows][kCtrlRank];
    const int s = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int q = lane >> 3, row = blockIdx.x * kUpRows + warp * 8 + (lane & 7);   // parts q kUpLaneParts.. of the row
    constexpr int kWords = kUpLaneParts * kUpPart / 8;   // the lane's weights: 8 bf16 a word
    const uint4* ur = reinterpret_cast<const uint4*>(a.up[s] + (size_t) row * kCtrlRank + q * kUpLaneParts * kUpPart);
    uint4 u[kWords];   // constant, so read before pdl_wait
#pragma unroll
    for (int k = 0; k < kWords; ++k) u[k] = ur[k];
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float* hid = a.scratch + (size_t) s * kDecodeRows * kCtrlRank;
    for (int i = threadIdx.x; i < rows * (kCtrlRank / 4); i += kUpRows * kUpLanes) {   // every copy in flight at once
        const int t = i / (kCtrlRank / 4), c = 4 * (i - t * (kCtrlRank / 4));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(&hs[t][c])),
                     "l"(hid + (size_t) t * kCtrlRank + c));
    }
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    const uint32_t* uw = reinterpret_cast<const uint32_t*>(u);   // bf16 pairs, the lower first
    float* gf = a.scratch + (size_t) kMaxStreams * kDecodeRows * kCtrlRank;
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        if (t >= rows) break;
        const float4* h4 = reinterpret_cast<const float4*>(&hs[t][q * kUpLaneParts * kUpPart]);
        float p[kUpLaneParts];
#pragma unroll
        for (int k = 0; k < kUpLaneParts; ++k) {
            float acc = 0.f;
#pragma unroll
            for (int c = 0; c < kUpPart / 4; ++c) {
                const float4 h = h4[k * kUpPart / 4 + c];
                const uint32_t w0 = uw[(k * kUpPart + 4 * c) / 2], w1 = uw[(k * kUpPart + 4 * c) / 2 + 1];
                acc = fmaf(__uint_as_float(w0 & 0xffff0000u), h.y, fmaf(__uint_as_float(w0 << 16), h.x, acc));
                acc = fmaf(__uint_as_float(w1 & 0xffff0000u), h.w, fmaf(__uint_as_float(w1 << 16), h.z, acc));
            }
            p[k] = acc;
        }
#pragma unroll
        for (int w = 1; w < kUpLaneParts; w *= 2)
#pragma unroll
            for (int k = 0; k < kUpLaneParts; k += 2 * w) p[k] += p[k + w];
        float v = p[0];
#pragma unroll
        for (int m = 8; m < 32; m *= 2) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (q == 0) gf[((size_t) s * kDecodeRows + t) * 2 * kWidth + row] = v;
    }
}

// The blend of stream s and row r, then its norm into q8_1.
__global__ void __launch_bounds__(kThreads) ctrl_blend_kernel(CtrlArgs a) {
    const int r = blockIdx.x, s = blockIdx.y;
    float nw[kPer], de[kPer], ms[kPer];
    load_row(a.norm[s], nw);
    // and the blend's states: written before the controllers' first kernel, which has finished before this one starts
    load_row(a.dest[s] + (size_t) r * kWidth, de);
    load_row(a.msg[s] + (size_t) r * kWidth, ms);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kWarpsB + 1];
    const float* gf =
        a.scratch + (size_t) kMaxStreams * kDecodeRows * kCtrlRank + ((size_t) s * kDecodeRows + r) * 2 * kWidth;
    float y[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        const float g = tanhf(gf[c]), f = tanhf(gf[kWidth + c]);
        y[j] = (1.f - f) * de[j] + g * ms[j];
    }
    norm_q8(y, nw, a.eps, a.q8[s], s == 0 ? a.norm_out : nullptr, r, red);
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

// Block (row, key/value head g), four warps: warp 0 the head's keys (convolved, clipped, cached; the raw inputs into the
// ring), warp 1 its values (normed), warps 2 and 3 its three query heads, their relative states and sinks (prep.cuh).
__global__ void __launch_bounds__(128) prepare_kernel(AttnPrep a) {
    const int r = blockIdx.x, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (w >= 2 && (g < a.g0 || g >= a.g0 + a.ng)) return;   // another card's query heads
    float wt[4][kConv];   // the weights, read while the kernel before finishes
    prep::load_weights(a, g, w, lane, wt);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int p = a.pos[r];
    if (w < 2) {
        // a batch window's row: its conversation's caches, and its rows of the window from its segment's first
        const int k = bk::seg_of(a.seg.at, r);
        const bool b = a.seg.at.n > 0;
        const SegCaches& sc = a.seg.c[k];
        float* ring = w ? (b ? sc.v_ring : a.v_ring) : (b ? sc.k_ring : a.k_ring);
        void* cache = w ? (b ? sc.v : a.v_cache) : (b ? sc.k : a.k_cache);
        float val[4], raw[4], f;
        prep::kv_row(a, r, p, g, w, lane, wt, val, raw, f, ring, b ? a.seg.at.row0[k] : 0);
        const size_t cell = ((size_t) (p % a.cap) * kKvHeads + g) * kHeadDim + lane * 4;
#pragma unroll
        for (int u = 0; u < 4; ++u) prep::store_kv(cache, a.fp8, cell + u, val[u] * f);
        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
            make_float4(raw[0], raw[1], raw[2], raw[3]);
        return;
    }
    prep::q_heads(a, r, p, g, w, lane, wt, a.q + ((size_t) r * kHeads + g * 3) * kHeadDim, a.r + ((size_t) r * kHeads + g * 3) * kRel,
                  a.sink + (size_t) r * kHeads + g * 3);
}

// ---- after the attention -----------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) post_attn_kernel(PostAttn a) {
    float taps[kPer][kConv], qv[kPer], nw[kPer];   // the weights, read while the kernel before finishes
#pragma unroll
    for (int j = 0; j < kPer; ++j)
#pragma unroll
        for (int k = 0; k < kConv; ++k) taps[j][k] = a.taps[(threadIdx.x + j * kThreads) * kConv + (kConv - 1 - k)];
    load_row(a.query, qv);
    if (!a.mix_out) load_row(a.norm, nw);
    // also before the wait, what no kernel since the attention's projections writes (the kernels since have passed their
    // waits before this one starts): the finished blocks, the layer's input, the ring's entries of the positions before
    // the window; every input is read before the first store (the ring is one of them)
    const int r = blockIdx.x, pr = a.pos[r];
    // a batch window's row: its conversation's ring, and its rows of the window from its segment's first
    const int sk = bk::seg_of(a.seg.at, r), first = a.seg.at.n > 0 ? a.seg.at.row0[sk] : 0;
    float* const ring = a.seg.at.n > 0 ? a.seg.c[sk].a_ring : a.ring;
    float v[kMaxMix][kPer];
    const int nb = a.n_blocks + (a.boundary ? 1 : 0), n = nb + 1;
#pragma unroll
    for (int i = 0; i < kMaxMix - 1; ++i) {
        if (i >= nb) break;
        const float* s = i < a.n_blocks ? a.blocks[i] : a.hid;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = s[(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    float xin[kPer][kConv], hres[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
#pragma unroll
        for (int k = 1; k < kConv; ++k)
            if (r - k < first) xin[j][k] = prep::conv_ring(ring, kWidth, pr - k, c);
        hres[j] = a.boundary ? 0.f : a.hid[(size_t) r * kWidth + c];
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
    // the convolution's rows of the output projection: this row's and the window's rows before it
#pragma unroll
    for (int j = 0; j < kPer; ++j)
#pragma unroll
        for (int k = 0; k < kConv; ++k)
            if (r - k >= first) xin[j][k] = a.o[(size_t) (r - k) * kWidth + threadIdx.x + j * kThreads];
    const int slot = pr % kConvRing;
    float res[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < kConv; ++k) acc = fmaf(taps[j][k], xin[j][k], acc);
        const float conv = __fmul_rn(acc + xin[j][0], rsqrtf((float) kConv + 1.f));
        res[j] = a.boundary ? conv : conv + hres[j];
        a.resid[(size_t) r * kWidth + c] = res[j];
        ring[(size_t) slot * kWidth + c] = xin[j][0];
    }
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == nb)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = res[j];
    float mixed[kPer];
    mix_rows(v, n, qv, mixed, red);
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) a.mix_out[(size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[j];
        return;
    }
    norm_q8(mixed, nw, a.eps, a.q8, a.norm_out, r, red);
}

// ---- experts -----------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The exchange's wait (Exchange): until the box reaches `want` (counts wrap). It reads the abort word only once it has
// waited kXchPollNs, and gives up past kXchBoundNs, setting the word.
constexpr uint64_t kXchPollNs = 1000000ull, kXchBoundNs = 2000000000ull;
__device__ __noinline__ void exchange_wait(Exchange x, uint32_t want) {
    const uint64_t t0 = global_ns();
    while ((int32_t) (ld_acquire_sys(x.box) - want) < 0) {
        const uint64_t waited = global_ns() - t0;
        if (waited < kXchPollNs) continue;
        volatile uint32_t* const word = x.abort;
        if (*word != 0) return;
        if (waited > kXchBoundNs) {
            *word = 1;
            __threadfence_system();
            return;
        }
    }
}

// One thread: this card's count of the attention's exchange to the other card, then the wait for the other's.
__global__ void attn_exchange_kernel(Exchange x) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    if (threadIdx.x != 0) return;
    const uint32_t want = *x.sent;
    st_release_sys(x.peer_box, want);
    exchange_wait(x, want);
}

// Block (row, 256 channels), a thread a channel: the experts' weighted sums of this row and the rows before it in the
// window that its convolution reads, the ring for the positions before the window, plus the residual; the ring takes this
// row's sum.
__global__ void __launch_bounds__(256) moe_finish_kernel(MoeFinish a) {
    const int r = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    // the exchange: block 0 tells the other card this card's count first (its sum went out in the kernel before)
    uint32_t want = 0;
    if (a.xch.sent && threadIdx.x == 0) {
        want = *a.xch.sent;
        if (r == 0 && blockIdx.y == 0) st_release_sys(a.xch.peer_box, want);
    }
    // a batch window's row: its conversation's ring, and its rows of the window from its segment's first
    const int sk = bk::seg_of(a.seg.at, r), first = a.seg.at.n > 0 ? a.seg.at.row0[sk] : 0;
    float* const ring = a.seg.at.n > 0 ? a.seg.c[sk].m_ring : a.ring;
    float taps[kConv];   // read while the kernel before finishes
#pragma unroll
    for (int j = 0; j < kConv; ++j) taps[j] = a.taps[c * kConv + (kConv - 1 - j)];
    if (threadIdx.x < kConv - 1) {   // the block's ring entries of the three positions before this row into L2
        const int pj = a.pos[r] - 1 - (int) threadIdx.x;
        if (pj >= 0)
            bk::prefetch_l2(ring + (size_t) (pj % kConvRing) * kWidth + blockIdx.y * 256, 256 * 4);
    }
    if (a.xch.sent) {   // the other card's sum of this visit is in once the box holds this card's count
        if (threadIdx.x == 0) exchange_wait(a.xch, want);
        __syncthreads();
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    {
        float acc = 0.f, x0 = 0.f;
#pragma unroll
        for (int j = 0; j < kConv; ++j) {
            float x;
            if (r - j >= first && a.part[0]) {   // two cards: card 0's half experts' sum, then card 1's
                const size_t i = (size_t) (r - j) * kWidth + c;
                x = a.part[0][i] + a.part[1][i];
            } else if (r - j >= first) {
                x = 0.f;
                for (int s = 0; s < a.k; ++s) x = fmaf(a.w[(r - j) * a.k + s], a.rows[((size_t) (r - j) * a.k + s) * kWidth + c], x);
            } else {
                const int p = a.pos[r] - j;
                x = p < 0 ? 0.f : ring[(size_t) (p % kConvRing) * kWidth + c];
            }
            if (j == 0) x0 = x;
            acc = fmaf(taps[j], x, acc);
        }
        const size_t i = (size_t) r * kWidth + c;
        const float y = __fmul_rn(acc + x0, rsqrtf((float) kConv + 1.f)) + a.resid[i];
        a.out[i] = y;
        if (a.tap) a.tap[i] = y;
        ring[(size_t) (a.pos[r] % kConvRing) * kWidth + c] = x0;
    }
}

void check_rows(int rows, const char* what) {
    if (rows > kDecodeRows) throw std::runtime_error(std::string("inkling2 ") + what + ": more rows than a decode window");
}

}  // namespace

void mix_norm(const MixNorm& a, int n_out, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "mix_norm");
    if (a.st.n < 1 || a.st.n > kMaxMix || n_out < 1 || (!a.mix_out && n_out > 3))
        throw std::runtime_error("inkling2 mix_norm: states or outputs");
    bk::launch_pdl(mix_norm_kernel, dim3((unsigned) (rows + (a.kv_k ? 1 : 0)), (unsigned) n_out), dim3(kThreads), 0, (cudaStream_t) stream,
                   a);
    check("mix_norm");
}

size_t ctrl_scratch_floats() { return kCtrl.size; }

void controllers(const CtrlArgs& a, int streams, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "controllers");
    if (streams < 1 || streams > kMaxStreams) throw std::runtime_error("inkling2 controllers: streams");
    cudaStream_t s = (cudaStream_t) stream;
    // The pick for the rows and streams measured fastest: a few rows read each input once for a block's outputs and need
    // no merge, more rows read it once for a group's; one stream's launches have a third of the blocks, so their blocks
    // take fewer outputs, and their slices more warps
    const dim3 whole(kDownSlices * 32), slices((unsigned) kGroups, kDownSlices, (unsigned) streams);
    if (rows <= 2 && streams == 1)
        bk::launch_pdl(ctrl_down_whole_kernel<1, 2>, dim3((unsigned) kCtrlRank, 1u), whole, 0, s, a, rows);
    else if (rows <= 4 && streams > 1)
        bk::launch_pdl(ctrl_down_whole_kernel<4, 4>, dim3((unsigned) kCtrlRank / 4, (unsigned) streams), whole, 0, s, a,
                       rows);
    else if (streams == 1) bk::launch_pdl(ctrl_down_slice_kernel<8>, slices, dim3(8 * 32), 0, s, a, rows);
    else bk::launch_pdl(ctrl_down_slice_kernel<4>, slices, dim3(4 * 32), 0, s, a, rows);
    check("controllers/down");
    bk::launch_pdl(ctrl_up_kernel, dim3((unsigned) (2 * kWidth / kUpRows), (unsigned) streams), dim3(kUpRows * kUpLanes), 0,
                   s, a, rows);
    check("controllers/up");
    bk::launch_pdl(ctrl_blend_kernel, dim3((unsigned) rows, (unsigned) streams), dim3(kThreads), 0, s, a);
    check("controllers/blend");
}

void decode_prepare(const AttnPrep& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "decode_prepare");
    bk::launch_pdl(prepare_kernel, dim3((unsigned) rows, kKvHeads), dim3(128), 0, (cudaStream_t) stream, a);
    check("decode_prepare");
}

void post_attn(const PostAttn& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "post_attn");
    if (a.n_blocks + (a.boundary ? 1 : 0) + 1 > kMaxMix) throw std::runtime_error("inkling2 post_attn: blocks");
    bk::launch_pdl(post_attn_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, a);
    check("post_attn");
}

void attn_exchange(const Exchange& x, void* stream) {
    if (!x.sent || !x.box || !x.peer_box || !x.abort) throw std::runtime_error("inkling2 attn_exchange: words");
    bk::launch_pdl(attn_exchange_kernel, dim3(1), dim3(32), 0, (cudaStream_t) stream, x);
    check("attn_exchange");
}

void moe_finish(const MoeFinish& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "moe_finish");
    bk::launch_pdl(moe_finish_kernel, dim3((unsigned) rows, kWidth / 256), dim3(256), 0, (cudaStream_t) stream, a);
    check("moe_finish");
}

}  // namespace basalt::models::inkling2
