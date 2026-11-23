// Router: adapted from the gating and top-k selection of build_moe_ffn in llama.cpp (src/llama-graph.cpp): sqrt-softplus or
// sigmoid scores, a bias for the choice, the unbiased scores of the chosen normalised.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "granite/kernels/router.hpp"

#include "granite/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kMaxTok = 8;          // tokens a logits block takes at most
constexpr int kLogitThreads = 256;
constexpr int kMaxVec = 3;          // float4 per thread: n_embd <= 4 * kMaxVec * kLogitThreads
constexpr int kMaxExpert = 512;
constexpr int kGateRows = 4;        // gate rows a logits block: a token's float4 is read once for them
constexpr int kTokStep = 2;         // tokens a step of a thread's loop takes, their reads in flight together
static_assert(32 % kGateRows == 0 && kMaxTok % kTokStep == 0 && kGateRows * kMaxTok <= kLogitThreads,
              "whole blocks of a multiple of 32 experts, whole steps, a thread a (row, token) sum");

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// kGateRows gate rows against `tb` (at most kMaxTok) tokens a block. A thread owns fixed columns of the rows, loaded once
// into registers; each (row, token) sum runs over the thread's columns in order, then the warp tree, then the warps in
// order, so its bits do not depend on the block's rows or tokens.
__global__ void __launch_bounds__(kLogitThreads)
router_logits_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_embd, int n_expert, int n_tokens,
                     int tb, float* __restrict__ logits) {
    __shared__ float red[kGateRows][kMaxTok * (kLogitThreads / 32)];
    const int r0 = blockIdx.x * kGateRows, t0 = blockIdx.y * tb, nt = min(tb, n_tokens - t0), n4 = n_embd >> 2;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float4 w[kGateRows][kMaxVec];
#pragma unroll
    for (int g = 0; g < kGateRows; ++g)
#pragma unroll
        for (int c = 0; c < kMaxVec; ++c) {
            const int i = threadIdx.x + c * kLogitThreads;
            if (i < n4) w[g][c] = reinterpret_cast<const float4*>(gate + (size_t) (r0 + g) * n_embd)[i];
        }
    // the gate rows are constant: they are read while the kernel before finishes (inside a PdlScope)
    pdl_wait();
    pdl_launch_dependents();
    for (int t = 0; t < nt; t += kTokStep) {
        float4 v[kTokStep][kMaxVec];
#pragma unroll
        for (int u = 0; u < kTokStep; ++u)
#pragma unroll
            for (int c = 0; c < kMaxVec; ++c) {
                const int i = threadIdx.x + c * kLogitThreads;
                if (t + u < nt && i < n4) v[u][c] = reinterpret_cast<const float4*>(x + (size_t) (t0 + t + u) * n_embd)[i];
            }
        float acc[kTokStep][kGateRows];
#pragma unroll
        for (int u = 0; u < kTokStep; ++u)
#pragma unroll
            for (int g = 0; g < kGateRows; ++g) acc[u][g] = 0.f;
#pragma unroll
        for (int c = 0; c < kMaxVec; ++c) {
            if (threadIdx.x + c * kLogitThreads >= n4) break;
#pragma unroll
            for (int u = 0; u < kTokStep; ++u)
#pragma unroll
                for (int g = 0; g < kGateRows; ++g) {
                    acc[u][g] = fmaf(w[g][c].x, v[u][c].x, acc[u][g]);
                    acc[u][g] = fmaf(w[g][c].y, v[u][c].y, acc[u][g]);
                    acc[u][g] = fmaf(w[g][c].z, v[u][c].z, acc[u][g]);
                    acc[u][g] = fmaf(w[g][c].w, v[u][c].w, acc[u][g]);
                }
        }
#pragma unroll
        for (int u = 0; u < kTokStep; ++u)
#pragma unroll
            for (int g = 0; g < kGateRows; ++g) {
                const float sum = warp_sum(acc[u][g]);
                if (lane == 0 && t + u < nt) red[g][(t + u) * (kLogitThreads / 32) + warp] = sum;
            }
    }
    __syncthreads();
    const int g = threadIdx.x / kMaxTok, t = threadIdx.x % kMaxTok;
    if (g < kGateRows && t < nt) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kLogitThreads / 32; ++i) s += red[g][t * (kLogitThreads / 32) + i];
        logits[(size_t) (t0 + t) * n_expert + r0 + g] = s;
    }
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : logf(1.f + expf(x));
    return sqrtf(sp);
}

__device__ __forceinline__ float score(float x, int gate) { return gate == 0 ? sqrt_softplus(x) : 1.f / (1.f + expf(-x)); }

// Shared memory of one token's selection.
struct SelectScratch {
    float wmax[kMaxExpert / 32];   // each warp's largest biased score
    float lb[kMaxExpert];          // the candidates' biased scores and experts, in no particular order
    int le[kMaxExpert];
    float thr;
    int n;
    float sel[16];
};

// Stage: one token, one thread per expert. An expert's rank counts the experts ahead of it (higher biased score, or
// equal and a lower index); ranks below k are the selection, in rank order. The k-th largest warp maximum bounds the
// k-th score from below (k distinct experts reach it), so only experts at or above it can rank below k, and they are
// ranked among themselves.
__device__ void router_select_token(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                    int k, float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights,
                                    SelectScratch& sh) {
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5, nw = n_expert >> 5;
    const float s = score(logits[e], gate);
    const float b = s + bias[e];
    float m = b;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) sh.wmax[warp] = m;
    if (e == 0) sh.n = 0;
    __syncthreads();
    if (warp == 0) {
        if (nw < k) {
            if (lane == 0) sh.thr = -INFINITY;
        } else if (lane < nw) {
            const float v = sh.wmax[lane];
            int r = 0;
            for (int j = 0; j < nw; ++j) r += (sh.wmax[j] > v) | ((sh.wmax[j] == v) & (j < lane));
            if (r == k - 1) sh.thr = v;
        }
    }
    __syncthreads();
    const bool cand = b >= sh.thr;
    if (cand) {
        const int i = atomicAdd(&sh.n, 1);
        sh.lb[i] = b;
        sh.le[i] = e;
    }
    __syncthreads();
    if (cand) {
        int rank = 0;
        for (int j = 0; j < sh.n; ++j) rank += (sh.lb[j] > b) | ((sh.lb[j] == b) & (sh.le[j] < e));
        if (rank < k) {
            sh.sel[rank] = s;
            ids[rank] = e;
        }
    }
    __syncthreads();
    if (e == 0) {
        float sum = 0.f;
        for (int r = 0; r < k; ++r) sum += sh.sel[r];
        const float denom = sum + 1e-20f;
        for (int r = 0; r < k; ++r) weights[r] = sh.sel[r] / denom * scale;
    }
}

__global__ void router_select_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n_expert,
                                     int k, float scale, int gate, int32_t* __restrict__ ids, float* __restrict__ weights) {
    pdl_wait();
    pdl_launch_dependents();
    __shared__ SelectScratch sh;
    const int t = blockIdx.x;
    router_select_token(logits + (size_t) t * n_expert, bias, n_expert, k, scale, gate, ids + (size_t) t * k,
                        weights + (size_t) t * k, sh);
}

}  // namespace

void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream) {
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.n_embd % 128 || a.n_embd > 4 * kMaxVec * kLogitThreads || a.k < 1 || a.k > 16 || a.k > a.n_expert)
        throw std::invalid_argument("router: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    // tokens a block: fewer for a few tokens, so the launch has more blocks (the pick measured fastest)
    const int tb = n_tokens <= 4 ? 2 : n_tokens <= 8 ? 4 : kMaxTok;
    const int launch_tok = tb * 65535;   // grid.y limit
    for (int t0 = 0; t0 < n_tokens; t0 += launch_tok) {
        const int nt = n_tokens - t0 < launch_tok ? n_tokens - t0 : launch_tok;
        launch_pdl(router_logits_kernel, dim3(a.n_expert / kGateRows, (nt + tb - 1) / tb), dim3(kLogitThreads), 0, st,
                   x + (size_t) t0 * a.n_embd, gate, a.n_embd, a.n_expert, nt, tb, logits + (size_t) t0 * a.n_expert);
    }
    router_select(logits, bias, n_tokens, a, ids, weights, stream);
}

void router_select(const float* logits, const float* bias, int n_tokens, const RouterArgs& a, int32_t* ids, float* weights,
                   void* stream) {
    if (a.n_expert % 32 || a.n_expert > kMaxExpert || a.k < 1 || a.k > 16 || a.k > a.n_expert)
        throw std::invalid_argument("router_select: unsupported geometry");
    cudaStream_t st = (cudaStream_t) stream;
    launch_pdl(router_select_kernel, dim3(n_tokens), dim3(a.n_expert), 0, st, logits, bias, a.n_expert, a.k, a.weight_scale,
               (int) a.gate, ids, weights);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::kernels
