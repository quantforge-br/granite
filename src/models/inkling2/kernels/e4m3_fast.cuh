// src/models/inkling2/kernels/e4m3_fast.cuh - branchless E4M3 (finite-only, bias 7) -> IEEE FP16 (bias 15) unpack
// for Ampere (sm_86) units compiled without --use_fast_math.
//
// Exact bit-for-bit on all 254 finite E4M3 codes (including -0.0 and all 14 subnormals), because shifting
//   ((b & 0x7F) << 7) | ((b & 0x80) << 8)
// maps E4M3 exponent 0 (subnormals m * 2^-9) to FP16 subnormals (m * 2^-17) and E4M3 normals (1.m * 2^(e-7))
// to FP16 normals (1.m * 2^(e-15)); multiplying by 2^8 (0x5C00 = 256.0h) in non-FTZ HMUL2 restores exact IEEE FP16.
// Note: E4M3 NaN codes (0x7F / 0xFF) map to +/-480.0h; safe for KV caches where store_kv only encodes finite values.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace granite::models::inkling2 {

__device__ __forceinline__ uint32_t e4m3x2_lanes02_to_h2_u32(uint32_t x) {
    // x holds byte0 in bits [0..7], byte1 in bits [16..23], and zeros in [8..15], [24..31].
    const uint32_t mag  = (x & 0x007F007Fu) << 7;
    const uint32_t sign = (x & 0x00800080u) << 8;
    const uint32_t raw  = mag | sign;
    const uint32_t s_u  = 0x5C005C00u;   // (256.0h, 256.0h) = 2^8 exact
    const __half2 h = __hmul2(*reinterpret_cast<const __half2*>(&raw), *reinterpret_cast<const __half2*>(&s_u));
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t e4m3x2_to_h2_u32(uint16_t u) {
    return e4m3x2_lanes02_to_h2_u32(__byte_perm((uint32_t) u, 0u, 0x4140u));
}

__device__ __forceinline__ __half2 e4m3x2_to_h2_fast(uint16_t u) {
    const uint32_t r = e4m3x2_to_h2_u32(u);
    return *reinterpret_cast<const __half2*>(&r);
}

__device__ __forceinline__ void e4m3x4_to_h2x2_fast(uint32_t w, __half2& h0, __half2& h1) {
    const uint32_t r0 = e4m3x2_lanes02_to_h2_u32(__byte_perm(w, 0u, 0x4140u));
    const uint32_t r1 = e4m3x2_lanes02_to_h2_u32(__byte_perm(w, 0u, 0x4342u));
    h0 = *reinterpret_cast<const __half2*>(&r0);
    h1 = *reinterpret_cast<const __half2*>(&r1);
}

}  // namespace granite::models::inkling2
