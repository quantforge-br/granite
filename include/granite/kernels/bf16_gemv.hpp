// include/granite/kernels/bf16_gemv.hpp - the GEMV for a BF16 weight and FP32 activations, defined in
// src/kernels/cuda/native_bf16.cu after the pinned llama.cpp CUDA single-token MMVF.
#pragma once

#include <cstdint>

namespace granite::kernels {

/// Single-token, contiguous 2D BF16-weight/F32-activation MMVF from the pinned llama.cpp CUDA path.
/// Preserves pair accumulation, adaptive block size and warp-XOR reduction order.
/// `w[o*n_in+i]` is BF16 bits; x/y are F32. n_in must be positive and even, n_out positive, both <= INT_MAX.
/// x must be 8-byte aligned, w 4-byte aligned, and y 4-byte aligned. Throws on invalid geometry/pointers.
/// The device buffers must not overlap; allocation bounds remain the caller's responsibility.
/// No allocations or synchronization, including when stream is null (the CUDA default stream).
void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream);
/// `ncols` columns at once: x is [ncols][n_in], y is [ncols][n_out]; each column bitwise equal to a
/// `bf16_gemv_fp32_mmvf` call on it.
void bf16_gemv_fp32_mmvf_cols(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int ncols,
                              void* stream);
/// bf16_gemv_fp32_mmvf for n_tok (1..8) activation rows x[t * ldx], outputs y[t * ldy + j]; one launch,
/// the weight read once, every output bit-identical to its own single-row call.
void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream);

}  // namespace granite::kernels
