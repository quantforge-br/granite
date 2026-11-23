// src/kernels/cuda/window_gemm.cuh - the decode window products' kernel (see include/basalt/kernels/window_gemm.hpp).  No
// kernel is instantiated here.
//
// A work item is RS rows of each of NM matrices (NM 2: a SwiGLU pair, gate rows in the first warps, up rows in the
// others) over one slice of K; a warp owns 16 rows.  The grid is at most one wave of blocks, each taking items
// blockIdx.x, + gridDim.x, ...: their chunks of CH values stream in order through shared memory as one sequence
// (cp.async, ST stages, the first ones issued before the PDL wait since weights are constant), and each warp runs its rows
// over them on mma m16n8k16 with the window's rows as the 8 columns.  Thread (g, c) holds weight rows g and g + 8 of its
// warp and activation column g; in every 64-value block of K it takes values 16c .. 16c + 15 (exactly one UE4M3 group
// per row), step s's k-slots {2c, 2c+1} values 16c + 4s, +1 and {2c+8, 2c+9} values 16c + 4s + 2, +3, and multiplies the
// same values of its activation column, so the sums are plain dot products.  With several slices each block writes its
// sums to the workspace, and the last block of a row tile to finish (a ticket) adds them in slice order and applies the
// epilogue.  The slice count depends on the call's shapes and the device only, never on T: a column's bits are the same
// at any T.
//
// Adapts the launch protocol of Infernix's decode products (Apache-2.0, by Ian Ranson and the Infernix contributors;
// see THIRD_PARTY_LICENSES): weights staged before the PDL wait, dependents started once the weights are read.
#pragma once

#include "basalt/kernels/narrow_bf16.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace basalt::kernels::wgemm::detail {

struct Args {
    const uint8_t* w[kMaxMatrices];
    const uint8_t* sp[kMaxMatrices];   // scale planes
    int64_t rows[kMaxMatrices];
    const float* scale[kMaxMatrices];
    float* y[kMaxMatrices];
    int64_t ld_y[kMaxMatrices];
    int64_t row_off[kMaxMatrices];   // the matrix's first row in the workspace's rows
    int tile0[kMaxMatrices + 1];     // the matrix's first tile (tile0[n]: all tiles)
    int n;
    int64_t K, part_rows;
    int T, tiles, ks, cps, nchunks, epi;
    const uint16_t* act;   // bf16 rows
    uint16_t* act_out;     // SwiGLU: h as the next product's bf16 rows ([kRows][rows]), or null
    float* partial;
    int32_t* tickets;
};

template <int RS, int NM, int CH, int ST>
struct Cfg {
    static constexpr int kThreads = 32 * NM * RS / 16;   // with one 16-row tile per warp
    static constexpr int CB = CH / 2;                    // weight bytes per row per chunk
    static constexpr int P = CB / 16;                    // 16-byte pieces per row per chunk
    static constexpr int SB = (CH / 64) * RS * 4;        // scale bytes per matrix per chunk
    static constexpr int MB = RS * CB + SB;              // bytes per matrix per stage
    static constexpr int kStage = NM * MB;
    static constexpr int kSmem = ST * kStage;
    static constexpr int kScalePiece = RS / 8;           // a scale tile line's bytes for RS rows
    static_assert(CH % 64 == 0 && (RS == 32 || RS == 64 || RS == 128), "window gemm: tile shape");
};

// the piece a row's 16-byte piece q lands on in shared memory (bank spread for the reads below)
template <int P> __device__ __forceinline__ int swz(int r) {
    if constexpr (P >= 8) return (r & 3) << 1;
    else if constexpr (P == 4) return ((r >> 1) & 1) << 1;
    else return 0;
}

// N bytes from gmem to the shared-window address s
template <int N> __device__ __forceinline__ void cp_async(unsigned s, const void* gmem) {
    if constexpr (N == 16) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
    else asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(gmem), "n"(N));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ld_nc16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <int RS, int NM, int CH, int ST>
__global__ void __launch_bounds__(Cfg<RS, NM, CH, ST>::kThreads, 1) wgemm_kernel(const Args a) {
    using C = Cfg<RS, NM, CH, ST>;
    constexpr int KB = CH / 64;                // 64-value blocks per chunk
    constexpr int kWarpsPerM = RS / 16;        // warps per matrix
    constexpr int kThreads = C::kThreads;
    extern __shared__ __align__(16) uint8_t smem[];
    __shared__ int last;
    __shared__ float xchg[NM == 2 ? RS * kRows : 1];   // SwiGLU: the up rows for the gate warps
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int ks = a.ks, items = a.tiles * ks, G = gridDim.x;
    const int64_t row_bytes = a.K / 2;

    // a work item is (row tile, K slice); a block takes items blockIdx.x, + G, ... and streams their chunks as one
    // sequence through the stages
    auto matrix_of = [&](int tile) {
        int mi = 0;
        if constexpr (NM == 1)
            while (mi + 1 < a.n && tile >= a.tile0[mi + 1]) ++mi;
        return mi;
    };
    auto chunks_of = [&](int item) { return min(a.cps, a.nchunks - (item % ks) * a.cps); };

    // A chunk's copies.  Of each matrix slot, thread t copies the 16-byte weight pieces t + j kThreads (rows
    // t / P + j kRowStep, all with piece t % P, so all with one swizzle) and the scale lines t + j kThreads: fixed offsets
    // in a stage.  The sources of the item being issued are set when it starts and advance a chunk at a time.
    constexpr int NW = RS * C::P / kThreads, NS = KB * 32 / kThreads;   // weight pieces, scale lines a thread copies
    constexpr int kRowStep = kThreads / C::P, kLineStep = kThreads / 32;
    static_assert(NW * kThreads == RS * C::P && kRowStep * C::P == kThreads && kRowStep % 4 == 0, "window gemm: copies");
    static_assert(NS * kThreads == KB * 32 && kLineStep * 32 == kThreads, "window gemm: scale copies");
    const int wr = threadIdx.x / C::P, wq = threadIdx.x % C::P;   // the first piece's row, the piece
    const int skb = threadIdx.x >> 5, sl = threadIdx.x & 31;      // the first scale line's 64-value block, the line
    const unsigned smem_s = (unsigned) __cvta_generic_to_shared(smem);
    const unsigned w_dst = smem_s + wr * C::CB + ((wq ^ swz<C::P>(wr)) << 4);
    const unsigned s_dst = smem_s + RS * C::CB + skb * RS * 4 + sl * C::kScalePiece;
    const uint8_t* wsrc[NM][NW];
    const uint8_t* ssrc[NM];
    auto start = [&](int item) {   // the sources of the item's first chunk (rows are whole tiles: multiples of 128)
        const int tile = item / ks, mi = matrix_of(tile);
        const int64_t row0 = (int64_t) (tile - a.tile0[mi]) * RS, kc = (int64_t) (item - tile * ks) * a.cps;
#pragma unroll
        for (int m = 0; m < NM; ++m) {
            const int mat = NM == 1 ? mi : m;
#pragma unroll
            for (int j = 0; j < NW; ++j)
                wsrc[m][j] = a.w[mat] + (row0 + wr + j * kRowStep) * row_bytes + kc * C::CB + wq * 16;
            ssrc[m] = a.sp[mat] + ((row0 >> 7) * (a.K / 64) + kc * KB + skb) * 512 + sl * 16 + ((row0 & 127) >> 5) * 4;
        }
    };
    auto issue = [&](unsigned st) {   // the next chunk into the stage at byte st
#pragma unroll
        for (int m = 0; m < NM; ++m) {
#pragma unroll
            for (int j = 0; j < NW; ++j) {
                cp_async<16>(w_dst + st + m * C::MB + j * kRowStep * C::CB, wsrc[m][j]);
                wsrc[m][j] += C::CB;
            }
#pragma unroll
            for (int j = 0; j < NS; ++j)
                cp_async<C::kScalePiece>(s_dst + st + m * C::MB + j * kLineStep * RS * 4, ssrc[m] + j * kLineStep * 512);
            ssrc[m] += KB * 512;
        }
    };
    int iss_item = blockIdx.x, iss_c = 0, iss_n = 0;
    unsigned iss_st = 0;
    auto issue_next = [&]() {
        if (iss_item < items) {
            if (iss_c == 0) {
                start(iss_item);
                iss_n = chunks_of(iss_item);
            }
            issue(iss_st);
            iss_st = iss_st == (ST - 1) * C::kStage ? 0u : iss_st + C::kStage;
            if (++iss_c == iss_n) {
                iss_c = 0;
                iss_item += G;
            }
        }
        cp_commit();
    };

#pragma unroll
    for (int i = 0; i < ST - 1; ++i) issue_next();
    pdl_wait();   // the activations and the workspace come from the kernels before this one
    // the next kernel may take SM room as blocks finish and stage its weights meanwhile; only after the wait, since a
    // dependent may read what the kernel before this one wrote before its own wait
    pdl_launch_dependents();

    const int wm = NM == 1 ? 0 : warp / kWarpsPerM;   // this warp's matrix slot in the stage
    const int rw = warp % kWarpsPerM;
    const int r0l = rw * 16 + g, r1l = r0l + 8;        // this thread's rows in the item
    const bool live = g < a.T;
    const uint16_t* xcol = a.act + (int64_t) g * a.K + 16 * c;
    // T * (thread + 1) < 2^31: a zero the compiler cannot prove
    const E2m1Table e2m1 = e2m1_table((unsigned) (a.T * (int) (threadIdx.x + 1)) >> 31);
    int q = 0;   // chunks consumed
    for (int item = blockIdx.x; item < items; item += G) {
        const int tile = item / ks, slice = item % ks, mi = matrix_of(tile);
        const int64_t row0 = (int64_t) (tile - a.tile0[mi]) * RS;
        const int n = chunks_of(item);
        const int64_t k0 = (int64_t) slice * a.cps * CH;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // the activations of chunk i + 1 load while chunk i computes
        uint4 na[KB], nb[KB];
        auto load_x = [&](int i) {
#pragma unroll
            for (int b = 0; b < KB; ++b) {
                const uint16_t* xp = xcol + k0 + i * CH + b * 64;
                na[b] = live ? ld_nc16(xp) : make_uint4(0, 0, 0, 0);
                nb[b] = live ? ld_nc16(xp + 8) : make_uint4(0, 0, 0, 0);
            }
        };
        load_x(0);
        for (int i = 0; i < n; ++i, ++q) {
            issue_next();
            uint4 xa[KB], xb[KB];
#pragma unroll
            for (int b = 0; b < KB; ++b) {
                xa[b] = na[b];
                xb[b] = nb[b];
            }
            if (i + 1 < n) load_x(i + 1);
            cp_wait<ST - 1>();
            __syncthreads();
            const uint8_t* sm = smem + (q % ST) * C::kStage + wm * C::MB;
            const uint8_t* q0 = sm + r0l * C::CB;
            const uint8_t* q1 = sm + r1l * C::CB;
#pragma unroll
            for (int b = 0; b < KB; ++b) {
                const int pc = 2 * b + (c >> 1), half = (c & 1) * 8;
                const uint2 w0 = *(const uint2*) (q0 + ((pc ^ swz<C::P>(r0l)) << 4) + half);
                const uint2 w1 = *(const uint2*) (q1 + ((pc ^ swz<C::P>(r1l)) << 4) + half);
                const uint8_t* ss = sm + RS * C::CB + b * RS * 4;
                const unsigned s0 = ss[(r0l & 31) * C::kScalePiece + (r0l >> 5) * 4 + c];
                const unsigned s1 = ss[(r1l & 31) * C::kScalePiece + (r1l >> 5) * 4 + c];
                const unsigned sc0 = ue4m3_bf16x2(s0), sc1 = ue4m3_bf16x2(s1);
                unsigned h0[8], h1[8];
                e2m1x8_bf16x2_scaled(w0.x, sc0, e2m1, *reinterpret_cast<unsigned(*)[4]>(h0));
                e2m1x8_bf16x2_scaled(w0.y, sc0, e2m1, *reinterpret_cast<unsigned(*)[4]>(h0 + 4));
                e2m1x8_bf16x2_scaled(w1.x, sc1, e2m1, *reinterpret_cast<unsigned(*)[4]>(h1));
                e2m1x8_bf16x2_scaled(w1.y, sc1, e2m1, *reinterpret_cast<unsigned(*)[4]>(h1 + 4));
                const unsigned xw[8] = {xa[b].x, xa[b].y, xa[b].z, xa[b].w, xb[b].x, xb[b].y, xb[b].z, xb[b].w};
                // step s: rows g, g + 8 at k-slots 2c.., then 2c + 8..
#pragma unroll
                for (int s = 0; s < 4; ++s) mma(acc, h0[2 * s], h1[2 * s], h0[2 * s + 1], h1[2 * s + 1], xw[2 * s], xw[2 * s + 1]);
            }
            __syncthreads();
        }
        // (dependents were started at the top: see the trigger before the wait)

        // ---- the item's epilogue: its slice's sums to the workspace, the last slice of the tile adds them in order;
        // acc[j] is column 2c + (j & 1) of row r0l + 8 (j >> 1)
        const int mat = NM == 1 ? mi : wm;
        const int64_t rows = a.rows[mat];
        bool final_block = true;
        if (ks > 1) {
            float* part = a.partial + (int64_t) slice * kRows * a.part_rows + a.row_off[mat];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = 2 * c + (j & 1);
                const int64_t r = row0 + r0l + 8 * (j >> 1);
                if (col < a.T && r < rows) part[col * a.part_rows + r] = acc[j];
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(a.tickets + tile, 1) == ks - 1;
            __syncthreads();
            final_block = last;
            if (final_block) {
                __threadfence();
                const float* base = a.partial + a.row_off[mat];
                // slice by slice, every element's load of a slice issued together (each element still sums its
                // slices in order from 0)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[j] = 0.0f;
                for (int k = 0; k < ks; ++k) {
                    const float* bk = base + (int64_t) k * kRows * a.part_rows;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int col = 2 * c + (j & 1);
                        const int64_t r = row0 + r0l + 8 * (j >> 1);
                        if (col < a.T && r < rows) acc[j] += __ldcg(bk + (int64_t) col * a.part_rows + r);
                    }
                }
                if (threadIdx.x == 0) a.tickets[tile] = 0;
            }
        }
        if (!final_block) continue;
        if constexpr (NM == 2) {
            // SwiGLU: the up warps hand their rows to the gate warps
            const float su = a.scale[1] ? *a.scale[1] : 1.0f, sg = a.scale[0] ? *a.scale[0] : 1.0f;
            if (wm == 1)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = 2 * c + (j & 1), rl = r0l + 8 * (j >> 1);
                    if (col < a.T) xchg[rl * kRows + col] = acc[j] * su;
                }
            __syncthreads();
            if (wm == 0)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = 2 * c + (j & 1), rl = r0l + 8 * (j >> 1);
                    if (col >= a.T || row0 + rl >= rows) continue;
                    const float gv = acc[j] * sg, uv = xchg[rl * kRows + col];
                    const float hv = gv / (1.0f + expf(-gv)) * uv;
                    if (a.y[0]) a.y[0][col * a.ld_y[0] + row0 + rl] = hv;
                    if (a.act_out) a.act_out[col * rows + row0 + rl] = __bfloat16_as_ushort(__float2bfloat16_rn(hv));
                }
            __syncthreads();   // xchg is free for the next item
        } else {
            const float sc = a.scale[mat] ? *a.scale[mat] : 1.0f;
            float* y = a.y[mat];
            const int64_t ld = a.ld_y[mat];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = 2 * c + (j & 1);
                const int64_t r = row0 + r0l + 8 * (j >> 1);
                if (col >= a.T || r >= rows) continue;
                const float v = acc[j] * sc;
                float* dst = y + col * ld + r;
                if (a.epi == (int) Epilogue::Add) *dst += v;
                else *dst = v;
            }
        }
    }
    cp_wait<0>();
}

/// A kernel instantiation as the host dispatch sees it.
struct Kernel {
    void (*fn)(Args) = nullptr;
    int threads = 0, smem = 0, rs = 0, ch = 0, nm = 1;
};

}  // namespace basalt::kernels::wgemm::detail
