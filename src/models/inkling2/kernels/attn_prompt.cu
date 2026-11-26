// src/models/inkling2/kernels/attn_prompt.cu - Inkling 2's prompt attention: a chunk's rows against every key before them,
// flash style on the tensor cores, with each query's relative bias over the last kExtent positions and its sink.
//
// A block takes kPos positions of the chunk and one key/value head. Its rows are the group's three query heads over those
// positions, head-major: warp w owns query head w / kWarpsPerHead of the group at 16 consecutive positions. Keys stream through shared
// memory in tiles of kKeys. S = Q K^T runs on fp16 tensor cores with fp32 accumulation. The relative bias of a warp's rows
// against a tile is one more product, R (the rows' relative states) times the band of the bias table the tile's distances
// span, moved through shared memory into the scores' layout. The softmax is online in fp32 (base 2), O += P V runs on fp16
// tensor cores with fp32 accumulation, and the sink joins the denominator at the end. A key at or after the chunk's first
// position comes from the chunk's own buffer (a full layer's cache, a sliding-window layer's stage), an earlier one from
// the cache (a window ring or the full cache).
#include "e4m3_fast.cuh"
#include "granite/models/inkling2/prompt_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace granite::models::inkling2 {
namespace {

constexpr int kPos = 64;                  // positions a block
constexpr int kWarpsPerHead = kPos / 16;  // 16 positions a warp
constexpr int kWarps = 3 * kWarpsPerHead;
constexpr int kThreads = kWarps * 32;
constexpr int kKeys = 32;                 // keys a tile
constexpr int kLd = kHeadDim + 8;         // padded fp16 row of a key tile (halves): ldmatrix without bank conflicts
constexpr int kBand = 96;                 // band rows: the distances a tile spans for the block (kPos + kKeys - 1)
constexpr int kBandLd = kRel + 8;         // padded band row (halves)
constexpr int kGCols = 48;                // distances a warp's rows span against a tile (16 + kKeys - 1), whole n-tiles
constexpr int kGLd = 55;                  // padded row of the exchanged bias (floats): two-way bank conflicts at most
constexpr float kLog2e = 1.4426950408889634f;
static_assert(kBand >= kPos + kKeys - 1 && kGCols >= 16 + kKeys - 1 && kGCols % 8 == 0 && kGLd >= kGCols, "band sizes");

struct KvTile {
    __half k[kKeys][kLd];
    __half v[kKeys][kLd];
};
struct Raw8Tile {
    uint8_t k[kKeys][kHeadDim];
    uint8_t v[kKeys][kHeadDim];
    uint16_t ks[kKeys][kHeadDim / 64];
    uint16_t vs[kKeys][kHeadDim / 64];
};
// A tile beyond the bias's reach needs neither the band nor the exchange, so the next such tile streams into the
// exchange's room while the block works on the current one (in FP8/KvInt8, raw_8 stages 8.25 KiB via cp.async on far
// tiles only; near tiles keep gx untouched and unpack synchronously into t0).
struct Smem {
    KvTile t0;
    __half band[kBand][kBandLd];
    union {
        struct {
            KvTile t1;
            Raw8Tile raw_8;
        } far_buf;
        float gx[kWarps][16][kGLd];
    } u;
};
static_assert(sizeof(Smem::u.far_buf) <= sizeof(float) * kWarps * 16 * kGLd,
              "the second tile and raw 8-bit staging fit in the exchange");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}
// Stage raw 8-bit tile (E4M3 or KvInt8 g64) into smem via cp.async (far tiles only, where gx is unused).
template <int kMode>
__device__ __forceinline__ void stage_raw_8(const AttnPromptArgs& a, Raw8Tile& raw, int lkvh, int64_t k0, int64_t lo,
                                            int64_t hi, int tid) {
    constexpr int kPieces = kKeys * kHeadDim / 16;   // 256 x 16-byte pieces per K or V tile
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem / (kHeadDim / 16), col = (rem % (kHeadDim / 16)) * 16;
        const int64_t key = k0 + j;
        const bool ok = key >= lo && key <= hi;
        const bool fresh = key >= a.pos0;
        const int64_t cap = fresh ? a.chunk_cap : a.cap;
        const uint8_t* base = static_cast<const uint8_t*>(fresh ? (isv ? a.v_chunk : a.k_chunk) : (isv ? a.v_cache : a.k_cache));
        const uint8_t* src = base + ((size_t) (ok ? key % cap : 0) * a.kv_stride + lkvh) * kHeadDim + col;
        cp16(isv ? &raw.v[j][col] : &raw.k[j][col], src, ok);
    }
    if constexpr (kMode == 2) {
        if (tid < 2 * kKeys) {
            const bool isv = tid >= kKeys;
            const int j = isv ? tid - kKeys : tid;
            const int64_t key = k0 + j;
            const bool ok = key >= lo && key <= hi;
            uint32_t sv = 0u;
            if (ok) {
                const bool fresh = key >= a.pos0;
                const int64_t cap = fresh ? a.chunk_cap : a.cap;
                const uint16_t* sbase = fresh ? (isv ? a.v_chunk_scale : a.k_chunk_scale) : (isv ? a.v_scale : a.k_scale);
                sv = *reinterpret_cast<const uint32_t*>(sbase + ((size_t) (key % cap) * a.kv_stride + lkvh) * (kHeadDim / 64));
            }
            *reinterpret_cast<uint32_t*>(isv ? raw.vs[j] : raw.ks[j]) = sv;
        }
    }
}

__device__ __forceinline__ void unpack16_i8_to_h(uint4 u, float sc, __half* dst) {
    int8_t b[16];
    *reinterpret_cast<uint4*>(b) = u;
    uint32_t h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half2 h2 = __floats2half2_rn((float) b[2 * i] * sc, (float) b[2 * i + 1] * sc);
        h[i] = *reinterpret_cast<const uint32_t*>(&h2);
    }
    *reinterpret_cast<uint4*>(dst) = make_uint4(h[0], h[1], h[2], h[3]);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(h[4], h[5], h[6], h[7]);
}

// Convert staged raw 8-bit tile in smem into padded FP16 KvTile.
template <int kMode>
__device__ __forceinline__ void unpack_raw_8_to_tile(const Raw8Tile& raw, KvTile& dst_tile, int tid) {
    constexpr int kPieces = kKeys * kHeadDim / 16;
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem / (kHeadDim / 16), col = (rem % (kHeadDim / 16)) * 16;
        const uint4 u = *reinterpret_cast<const uint4*>(isv ? &raw.v[j][col] : &raw.k[j][col]);
        __half* dst = isv ? &dst_tile.v[j][col] : &dst_tile.k[j][col];
        if constexpr (kMode == 1) {
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
            uint32_t h[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                __half2 h0, h1;
                e4m3x4_to_h2x2_fast(w[i], h0, h1);
                h[2 * i] = *reinterpret_cast<const uint32_t*>(&h0);
                h[2 * i + 1] = *reinterpret_cast<const uint32_t*>(&h1);
            }
            *reinterpret_cast<uint4*>(dst) = make_uint4(h[0], h[1], h[2], h[3]);
            *reinterpret_cast<uint4*>(dst + 8) = make_uint4(h[4], h[5], h[6], h[7]);
        } else {
            const float sc = __half2float(__ushort_as_half(isv ? raw.vs[j][col >> 6] : raw.ks[j][col >> 6]));
            unpack16_i8_to_h(u, sc, dst);
        }
    }
}

// One tile of keys and values into shared memory: key k0 + j at row j, zero outside [lo, hi].
// kMode: 0 = FP16, 1 = FP8 E4M3, 2 = KvInt8 g64.
template <int kMode>
__device__ __forceinline__ void load_tile(const AttnPromptArgs& a, KvTile& sm, int lkvh, int64_t k0, int64_t lo, int64_t hi, int tid) {
    if constexpr (kMode == 0) {
        constexpr int kPieces = kKeys * kHeadDim / 8;   // 16-byte pieces of one tile
        for (int c = tid; c < 2 * kPieces; c += kThreads) {
            const bool isv = c >= kPieces;
            const int rem = isv ? c - kPieces : c;
            const int j = rem / (kHeadDim / 8), col = (rem % (kHeadDim / 8)) * 8;
            const int64_t key = k0 + j;
            const bool ok = key >= lo && key <= hi;
            const bool fresh = key >= a.pos0;
            const int64_t cap = fresh ? a.chunk_cap : a.cap;
            const __half* base = static_cast<const __half*>(fresh ? (isv ? a.v_chunk : a.k_chunk) : (isv ? a.v_cache : a.k_cache));
            const __half* src = base + ((size_t) (ok ? key % cap : 0) * a.kv_stride + lkvh) * kHeadDim + col;
            cp16(isv ? &sm.v[j][col] : &sm.k[j][col], src, ok);
        }
    } else {
        constexpr int kPieces = kKeys * kHeadDim / 16;  // 16 codes a piece
        for (int c = tid; c < 2 * kPieces; c += kThreads) {
            const bool isv = c >= kPieces;
            const int rem = isv ? c - kPieces : c;
            const int j = rem / (kHeadDim / 16), col = (rem % (kHeadDim / 16)) * 16;
            const int64_t key = k0 + j;
            const bool ok = key >= lo && key <= hi;
            uint4 u = make_uint4(0u, 0u, 0u, 0u);
            float sc = 0.f;
            if (ok) {
                const bool fresh = key >= a.pos0;
                const int64_t cap = fresh ? a.chunk_cap : a.cap;
                const size_t row = (size_t) (key % cap) * a.kv_stride + lkvh;
                const uint8_t* base = static_cast<const uint8_t*>(fresh ? (isv ? a.v_chunk : a.k_chunk) : (isv ? a.v_cache : a.k_cache));
                u = *reinterpret_cast<const uint4*>(base + row * kHeadDim + col);
                if constexpr (kMode == 2) {
                    const uint16_t* sbase = fresh ? (isv ? a.v_chunk_scale : a.k_chunk_scale) : (isv ? a.v_scale : a.k_scale);
                    sc = __half2float(__ushort_as_half(sbase[row * (kHeadDim / 64) + (col >> 6)]));
                }
            }
            __half* dst = isv ? &sm.v[j][col] : &sm.k[j][col];
            if constexpr (kMode == 1) {
                const uint32_t w[4] = {u.x, u.y, u.z, u.w};
                uint32_t h[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    __half2 h0, h1;
                    e4m3x4_to_h2x2_fast(w[i], h0, h1);
                    h[2 * i] = *reinterpret_cast<const uint32_t*>(&h0);
                    h[2 * i + 1] = *reinterpret_cast<const uint32_t*>(&h1);
                }
                *reinterpret_cast<uint4*>(dst) = make_uint4(h[0], h[1], h[2], h[3]);
                *reinterpret_cast<uint4*>(dst + 8) = make_uint4(h[4], h[5], h[6], h[7]);
            } else {
                unpack16_i8_to_h(u, sc, dst);
            }
        }
    }
}

template <int kMode>
__global__ void __launch_bounds__(kThreads, 1) attn_prompt_kernel(const AttnPromptArgs a, const int rows, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int kvh = a.g0 + (int) blockIdx.y;
    const int lkvh = a.kv_stride < kKvHeads ? (int) blockIdx.y : kvh;
    const int i0 = (int) (gridDim.x - 1 - blockIdx.x) * kPos;   // the block's first row: the blocks with the most keys first
    const int64_t q0 = a.pos0 + i0;
    const int head = kvh * 3 + w / kWarpsPerHead;
    const int jw = (w % kWarpsPerHead) * 16;                     // the warp's first position in the block
    const int64_t pw = q0 + jw;
    const int ia = i0 + jw + g, ib = ia + 8;                     // the thread's two rows
    const bool va = ia < rows, vb = ib < rows;
    const int64_t hi = a.pos0 + min(rows, i0 + kPos) - 1;        // the block's last position: its last key
    const int64_t lo = a.window > 0 ? max((int64_t) 0, q0 - a.window + 1) : 0;   // its first key

    // the rows' queries and relative states as fp16 fragments (rows past the chunk read row 0 and are never stored)
    uint32_t qf[kHeadDim / 16][4], rf[4];
    {
        const float* qa = a.q + ((size_t) (va ? ia : 0) * kHeads + head) * kHeadDim + 2 * tq;
        const float* qb = a.q + ((size_t) (vb ? ib : 0) * kHeads + head) * kHeadDim + 2 * tq;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(qa + 16 * kk);
            const float2 x1 = *reinterpret_cast<const float2*>(qb + 16 * kk);
            const float2 x2 = *reinterpret_cast<const float2*>(qa + 16 * kk + 8);
            const float2 x3 = *reinterpret_cast<const float2*>(qb + 16 * kk + 8);
            qf[kk][0] = pack_h2(x0.x, x0.y);
            qf[kk][1] = pack_h2(x1.x, x1.y);
            qf[kk][2] = pack_h2(x2.x, x2.y);
            qf[kk][3] = pack_h2(x3.x, x3.y);
        }
        const float* ra = a.r + ((size_t) (va ? ia : 0) * kHeads + head) * kRel + 2 * tq;
        const float* rb = a.r + ((size_t) (vb ? ib : 0) * kHeads + head) * kRel + 2 * tq;
        rf[0] = pack_h2(ra[0], ra[1]);
        rf[1] = pack_h2(rb[0], rb[1]);
        rf[2] = pack_h2(ra[8], ra[9]);
        rf[3] = pack_h2(rb[8], rb[9]);
    }

    float o[kHeadDim / 8][4];
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    const float scale2 = a.scale * kLog2e;
    float (*gx)[kGLd] = sm.u.gx[w];

    // a tile is far when no distance of the block's rows to its keys reaches below kExtent: no bias, no band, no exchange
    const auto far = [&](int64_t k0) { return q0 - k0 - (kKeys - 1) >= kExtent; };
    int cur = 0;          // the room the current tile is in (1: the exchange's, FP16 only)
    bool ready = false;   // FP16: current tile prefetched; FP8: current far tile staged in u.far_buf.raw_fp8
    for (int64_t k0 = lo / kKeys * kKeys; k0 <= hi; k0 += kKeys) {
        const KvTile* kt_ptr = &sm.t0;
        if constexpr (kMode == 0) {
            __syncthreads();   // every warp is done with the previous tile
            if (!ready) {
                cur = 0;
                load_tile<0>(a, sm.t0, lkvh, k0, lo, hi, tid);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            // a far tile followed by a far tile: the next one is requested into the other room before this one is used
            const bool next = far(k0) && k0 + kKeys <= hi && far(k0 + kKeys);
            if (next) {
                load_tile<0>(a, cur ? sm.t0 : sm.u.far_buf.t1, lkvh, k0 + kKeys, lo, hi, tid);
                asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;\n" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            }
            kt_ptr = cur ? &sm.u.far_buf.t1 : &sm.t0;
            ready = next;
            if (next) cur ^= 1;
        } else {
            if (far(k0)) {
                if (!ready) {
                    __syncthreads();
                    stage_raw_8<kMode>(a, sm.u.far_buf.raw_8, lkvh, k0, lo, hi, tid);
                    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
                    __syncthreads();
                } else {
                    // wait for previous iteration's async prefetch of k0 into raw_8, and for all warps to finish reading t0
                    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    __syncthreads();
                }
                unpack_raw_8_to_tile<kMode>(sm.u.far_buf.raw_8, sm.t0, tid);
                __syncthreads();   // t0 populated and raw_8 completely read
                const bool next_far = k0 + kKeys <= hi && far(k0 + kKeys);
                if (next_far) {
                    stage_raw_8<kMode>(a, sm.u.far_buf.raw_8, lkvh, k0 + kKeys, lo, hi, tid);
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                }
                ready = next_far;
            } else {
                // Near tile: gx is active in sm.u, so never touch sm.u.far_buf; ready is guaranteed false here.
                __syncthreads();
                load_tile<kMode>(a, sm.t0, lkvh, k0, lo, hi, tid);
                ready = false;
            }
        }
        const KvTile& kt = *kt_ptr;
        // the bias band: row b holds the table's column d = q0 - k0 - (kKeys - 1) + b, zero outside [0, kExtent)
        if (!far(k0))
            for (int idx = tid; idx < kBand * kRel; idx += kThreads) {
                const int t = idx / kBand, b = idx % kBand;
                const int64_t d = q0 - k0 - (kKeys - 1) + b;
                sm.band[b][t] = __float2half_rn(d >= 0 && d < kExtent ? a.rel_table[(size_t) t * kExtent + d] : 0.f);
            }
        if constexpr (kMode == 0) __syncthreads();
        else if (!far(k0)) __syncthreads();

        // a warp none of whose rows sees a key of the tile skips it; one whose rows see every key needs no mask
        const int64_t p_first = pw, p_last = pw + 15;
        if (k0 > p_last || (a.window > 0 && k0 + kKeys - 1 <= p_first - a.window)) continue;
        const bool whole = k0 + kKeys - 1 <= p_first && (a.window <= 0 || k0 > p_last - a.window);

        // S = Q K^T
        float s[kKeys / 8][4];
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < kKeys / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &kt.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                mma16816(s[n + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
            }
        }
        // the scores in base 2 with the bias: G[j][c] = r_j . table[:, pw - k0 - (kKeys - 1) + c] for the warp's rows j,
        // and key k0 + kk of row j sits at c = j - kk + kKeys - 1
        if (p_first - (k0 + kKeys - 1) < kExtent) {
            float ga[kGCols / 8][4];
#pragma unroll
            for (int nt = 0; nt < kGCols / 8; ++nt) {
                ga[nt][0] = ga[nt][1] = ga[nt][2] = ga[nt][3] = 0.f;
                unsigned b0, b1;
                ldsm_x2(b0, b1, &sm.band[jw + nt * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
                mma16816(ga[nt], rf[0], rf[1], rf[2], rf[3], b0, b1);
            }
#pragma unroll
            for (int nt = 0; nt < kGCols / 8; ++nt) {
                gx[g][nt * 8 + 2 * tq] = ga[nt][0];
                gx[g][nt * 8 + 2 * tq + 1] = ga[nt][1];
                gx[g + 8][nt * 8 + 2 * tq] = ga[nt][2];
                gx[g + 8][nt * 8 + 2 * tq + 1] = ga[nt][3];
            }
            __syncwarp();
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int j = g + (e >> 1) * 8, kk = n * 8 + 2 * tq + (e & 1);
                    s[n][e] = fmaf(s[n][e], scale2, gx[j][j - kk + kKeys - 1] * kLog2e);
                }
            __syncwarp();
        } else {
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[n][e] *= scale2;
        }
        if (!whole) {
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int64_t key = k0 + n * 8 + 2 * tq + (e & 1), p = pw + g + (e >> 1) * 8;
                    if (key > p || (a.window > 0 && key <= p - a.window)) s[n][e] = -INFINITY;
                }
        }
        // the online softmax: rows a (s[n][0..1]) and b (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) {
            mx_a = fmaxf(mx_a, fmaxf(s[n][0], s[n][1]));
            mx_b = fmaxf(mx_b, fmaxf(s[n][2], s[n][3]));
        }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float ca = mx_a == -INFINITY ? 1.f : exp2f(m_a - mx_a), cb = mx_b == -INFINITY ? 1.f : exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.f, ps_b = 0.f;
#pragma unroll
        for (int n = 0; n < kKeys / 8; ++n) {
            s[n][0] = s[n][0] == -INFINITY ? 0.f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] == -INFINITY ? 0.f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] == -INFINITY ? 0.f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] == -INFINITY ? 0.f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * ca + ps_a;
        l_b = l_b * cb + ps_b;
#pragma unroll
        for (int n = 0; n < kHeadDim / 8; ++n) {
            o[n][0] *= ca;
            o[n][1] *= ca;
            o[n][2] *= cb;
            o[n][3] *= cb;
        }
        // O += P V
        unsigned pf[kKeys / 16][4];
#pragma unroll
        for (int j = 0; j < kKeys / 16; ++j) {
            pf[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pf[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pf[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pf[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < kHeadDim / 8; n += 2)
#pragma unroll
            for (int j = 0; j < kKeys / 16; ++j) {
                unsigned b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, &kt.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816(o[n], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                mma16816(o[n + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
            }
    }

    // the sink beside the keys, then the rows out
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float sa = va ? a.sink[(size_t) ia * kHeads + head] * kLog2e : 0.f;
    const float sb = vb ? a.sink[(size_t) ib * kHeads + head] * kLog2e : 0.f;
    const float ma = fmaxf(m_a, sa), mb = fmaxf(m_b, sb);
    const float fa = exp2f(m_a - ma), fb = exp2f(m_b - mb);
    const float inv_a = fa / (l_a * fa + exp2f(sa - ma)), inv_b = fb / (l_b * fb + exp2f(sb - mb));
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) {
        const int col = n * 8 + 2 * tq;
        if (va) *reinterpret_cast<float2*>(out + ((size_t) ia * kHeads + head) * kHeadDim + col) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        if (vb) *reinterpret_cast<float2*>(out + ((size_t) ib * kHeads + head) * kHeadDim + col) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
    }
}

}  // namespace

void attention_prompt(const AttnPromptArgs& a, int rows, float* out, void* stream) {
    if (rows <= 0) return;
    if (a.g0 < 0 || a.ng < 1 || a.g0 + a.ng > kKvHeads) throw std::runtime_error("inkling2 attention_prompt: groups");
    if (a.kv_stride != kKvHeads && a.kv_stride != a.ng) throw std::runtime_error("inkling2 attention_prompt: kv_stride");
    const dim3 grid((unsigned) ((rows + kPos - 1) / kPos), (unsigned) a.ng);
    cudaStream_t s = (cudaStream_t) stream;
    static bool sized[kOptInDevices] = {};   // the opt-in is per device
    if (bool& z = sized[opt_in_device()]; !z) {
        cudaFuncSetAttribute(attn_prompt_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_prompt_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_prompt_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        z = true;
    }
    if (a.k_scale) attn_prompt_kernel<2><<<grid, kThreads, sizeof(Smem), s>>>(a, rows, out);
    else if (a.fp8) attn_prompt_kernel<1><<<grid, kThreads, sizeof(Smem), s>>>(a, rows, out);
    else attn_prompt_kernel<0><<<grid, kThreads, sizeof(Smem), s>>>(a, rows, out);
    check("attention_prompt");
}

}  // namespace granite::models::inkling2
