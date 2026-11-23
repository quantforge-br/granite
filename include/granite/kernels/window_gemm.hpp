// include/granite/kernels/window_gemm.hpp - the weight products of a decode window: up to kRows activation rows against an
// NVFP4 matrix in planes (nvfp4_planes.hpp), the matrix read once at memory speed.
//
// Activations: bf16 rows.  Weights: E2M1 codes times their UE4M3 group scale, converted to bf16 in registers (exact).
// Products: mma m16n8k16 (bf16 in, fp32 accumulate) with the window's rows as the 8 columns.  A column's sums
// never meet another column's, and the order of a row's sums (fixed chunks of K in order within a K slice, the slices
// across blocks added in slice order) depends only on the shapes of the call's matrices: a row's bits do not depend on T
// or on what the other rows hold.
//
// Products start as programmatic dependents inside a PdlScope (pdl.hpp): each stages its first weights before it waits
// for the kernel before it.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::wgemm {

constexpr int kRows = 8;   // rows (columns of the product) a call takes: a drafted block

/// Bytes of a window's activations for K values per row: [kRows][K] bf16.
std::size_t act_bytes(int64_t K);

/// Rows 0..T-1 of x (row t at x + t * ld_x, K values) as bf16 rows in `act`.
void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream);
/// The RMS norm of rows 0..T-1 of x ([T][K], weights w, any offset folded) packed into `act` (rms_norm_rows' values, then
/// rounded to bf16).
void norm_pack(const float* x, const float* w, float eps, int64_t K, int T, void* act, void* stream);
/// The device scratch for split-K sums, sized with workspace_need over every call a model makes.
struct Workspace {
    float* partial = nullptr;    // [slices][kRows][rows] of the product in flight
    int32_t* tickets = nullptr;  // per row tile, back to 0 after each product
    std::size_t partial_floats = 0, tickets_n = 0;
};

/// One matrix of a call: W [rows][K] in planes, its output y[t * ld_y + r], `scale` a device fp32 scalar or null.
struct Matrix {
    const uint8_t* w = nullptr;
    int64_t rows = 0;
    const float* scale = nullptr;
    float* y = nullptr;
    int64_t ld_y = 0;
};

enum class Epilogue {
    Store,    // y = scale * W x
    Add,      // y += scale * W x
    SwiGLU,   // two matrices of the same rows (gate, up): m[0].y = silu(s_g gate x) * (s_u up x)
};

constexpr int kMaxMatrices = 3;

/// One call: up to three matrices of the same K against the same activations, one launch.
struct Call {
    int64_t K = 0;
    Matrix m[kMaxMatrices];
    int n = 1;
    Epilogue epi = Epilogue::Store;
    void* act_out = nullptr;   // SwiGLU: h also as bf16 rows for the next product ([kRows][rows], act_bytes(rows)); m[0].y may be null
};

/// The workspace a call needs (its shape and the current device).
void workspace_need(const Call& c, std::size_t& partial_floats, std::size_t& tickets);

/// Runs the call for rows 0..T-1 (1..kRows) of `act` (packed for c.K): W 16-byte aligned, rows and K multiples of 128.
void gemm(const Call& c, const void* act, int T, const Workspace& ws, void* stream);

}  // namespace basalt::kernels::wgemm
