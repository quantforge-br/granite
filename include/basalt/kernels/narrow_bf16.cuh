// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  sm_86 has no cvt from either format (cvt.*.e4m3x2 needs sm_89,
// cvt.*.e2m1x2 sm_100) and no mul.bf16x2 (sm_90), so a code's magnitude bits go under the bf16 exponent, where they read
// as the code's value times a fixed power of two (2^-120 for E4M3, 2^-126 for E2M1; normal and subnormal codes alike),
// and one bf16x2 fma by the inverse power, with a -0 addend (a plain multiply, -0 products kept), gives the value exactly.
// An E4M3 NaN (magnitude 0x7f) becomes a bf16 NaN.  tests/kernels/narrow_bf16_parity.cu checks every input against the
// host's values.
#pragma once

#include <cuda_runtime.h>

namespace basalt::kernels {

/// a * b on bf16x2, rounded once (fma.rn with a -0 addend: mul.rn.bf16x2 needs sm_90)
__device__ __forceinline__ unsigned bf16x2_mul_rn(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact; a NaN stays a NaN)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
    const unsigned s = (x & 0xffu) | ((x & 0xff00u) << 8);   // byte k in half k
    const unsigned m = (s & 0x007f007fu) << 4;                // exponent and mantissa under bf16's: the value * 2^-120
    const unsigned h = bf16x2_mul_rn(m | ((s & 0x00800080u) << 8), 0x7b807b80u);   // * 2^120, the sign kept
    return h | (__vcmpeq2(m, 0x07f007f0u) & 0x7fff7fffu);
}

/// two NVFP4 sub-block scale bytes (UE4M3: the low 16 bits of x, the low byte the lower half) as bf16x2, as the format
/// reads them elsewhere (basalt/artifact/dequant.hpp, the MMQ tiles): bit 7 ignored, 0x7f (NaN) as 0
__device__ __forceinline__ unsigned ue4m3x2_bf16x2(unsigned x) {
    const unsigned s = (x & 0x7fu) | ((x & 0x7f00u) << 8);   // byte k in half k, bit 7 dropped
    return e4m3x2_bf16x2(x & 0x7f7fu) & ~__vcmpeq2(s, 0x007f007fu);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most six significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned b = (w >> (8 * i)) & 0xffu;
        const unsigned s = (b & 0x0fu) | ((b & 0xf0u) << 12);   // code k in half k
        // exponent and mantissa under bf16's (the value * 2^-126), the sign to bit 15
        const unsigned bits = ((s & 0x00070007u) << 6) | ((s & 0x00080008u) << 12);
        h[i] = bf16x2_mul_rn(bf16x2_mul_rn(bits, 0x7e807e80u), sc);   // * 2^126, then the scale
    }
}

}  // namespace basalt::kernels
