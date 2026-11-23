// include/basalt/kernels/mhc.hpp - manifold-constrained hyper-connections (mHC): the stream count of the fused collapse
// and norm (row_norm.hpp's hc_pre_rms_norm).
//
// Hyper-connection streams are laid out [T][4][n_embd] (stream-major per token).
#pragma once

namespace basalt::kernels {

constexpr int kHcMult = 4;

}  // namespace basalt::kernels
