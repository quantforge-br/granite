// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  The supported cards have no cvt from either format and no mul.bf16x2, so
// an E4M3 code's magnitude bits go under the bf16 exponent, where they read as the code's value times 2^-120 (normal and
// subnormal codes alike), and one bf16x2 fma by 2^120, with a -0 addend (a plain multiply, -0 products kept), gives the
// value exactly; an E2M1 code's eight magnitudes come from a byte table.  An E4M3 NaN (magnitude 0x7f) becomes a bf16
// NaN.  tests/kernels/narrow_bf16_parity.cu checks every input against the host's values.
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

/// an NVFP4 sub-block scale byte (UE4M3, x < 256) as bf16 in both halves, as the format reads it elsewhere
/// (basalt/artifact/dequant.hpp, the MMQ tiles): bit 7 ignored, 0x7f (NaN) as 0
__device__ __forceinline__ unsigned ue4m3_bf16x2(unsigned x) {
    const unsigned m = (x * 0x00100010u) & 0x07f007f0u;   // in each half the magnitude under bf16's exponent: * 2^-120
    return m == 0x07f007f0u ? 0u : bf16x2_mul_rn(m, 0x7b807b80u);
}

/// The bf16 bytes of the E2M1 magnitudes 0, 0.5, 1, 1.5 (lo: the low bytes, hi: the high bytes; the other four
/// magnitudes' words are immediates in e2m1x8_bf16x2_scaled).  A kernel makes it once, from a zero that the compiler
/// cannot prove and that varies by lane, so the two words stay in registers instead of being copied from constants
/// before every permute.
struct E2m1Table {
    unsigned lo, hi;
};
__device__ __forceinline__ E2m1Table e2m1_table(unsigned zero) { return {0xc0800000u | zero, 0x3f3f3f00u | zero}; }

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most six significant bits).  A code's magnitude picks the two bytes of its bf16 value
/// from two eight-byte tables (byte permutes indexed by the magnitudes, four codes a permute), its sign goes to the high
/// byte's top bit, and one multiply applies the scale.
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, E2m1Table t, unsigned (&h)[4]) {
    const unsigned lo0 = t.lo, hi0 = t.hi;
    constexpr unsigned kLo1 = 0xc0804000u, kHi1 = 0x40404040u;   // magnitudes 2, 3, 4, 6
    // code k's magnitude in nibble k of the selector, which reads its low 16 bits (bit 3 of a nibble would replicate a
    // sign; __byte_perm masks it again on every call, hence the plain prmt)
    const unsigned mag = w & 0x77777777u, sels[2] = {mag, mag >> 16};
    const unsigned w4 = w << 4;   // code 2i's sign at the top of byte i, as code 2i + 1's is in w
    unsigned lo[2], hi[2];        // codes 4j .. 4j + 3: byte k the low or high byte of code 4j + k
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        unsigned l, h;
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(l) : "r"(lo0), "n"(kLo1), "r"(sels[j]));
        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(hi0), "n"(kHi1), "r"(sels[j]));
        lo[j] = l;
        hi[j] = h | (__byte_perm(w4, w, j ? 0x7362u : 0x5140u) & 0x80808080u);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = bf16x2_mul_rn(__byte_perm(lo[i >> 1], hi[i >> 1], (i & 1) ? 0x7362u : 0x5140u), sc);
}

}  // namespace basalt::kernels
