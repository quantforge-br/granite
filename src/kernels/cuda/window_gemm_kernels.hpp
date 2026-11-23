// src/kernels/cuda/window_gemm_kernels.hpp - the decode window products' kernel instantiations (window_gemm_nvfp4.cu).
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#pragma once

#include "window_gemm.cuh"

namespace basalt::kernels::wgemm::detail {

/// The kernel for a call of one or several matrices, and for a SwiGLU pair.
const Kernel& nvfp4();
const Kernel& nvfp4_swiglu();

template <int RS, int NM, int CH, int ST> Kernel make_kernel() {
    using C = Cfg<RS, NM, CH, ST>;
    Kernel k;
    k.fn = wgemm_kernel<RS, NM, CH, ST>;
    k.threads = C::kThreads;
    k.smem = C::kSmem;
    k.rs = RS;
    k.ch = CH;
    k.nm = NM;
    return k;
}

}  // namespace basalt::kernels::wgemm::detail
