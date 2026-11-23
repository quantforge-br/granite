// src/kernels/cuda/window_gemm_nvfp4.cu - the decode window products against NVFP4 matrices in planes, alone and as a
// SwiGLU pair (window_gemm.cuh).
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#include "window_gemm_kernels.hpp"

namespace granite::kernels::wgemm::detail {
namespace {
constexpr int CH = 256, ST = 4;
}

const Kernel& nvfp4() {
    static const Kernel k = make_kernel<64, 1, CH, ST>();
    return k;
}

const Kernel& nvfp4_swiglu() {
    static const Kernel k = make_kernel<32, 2, CH, ST>();
    return k;
}

}  // namespace granite::kernels::wgemm::detail
