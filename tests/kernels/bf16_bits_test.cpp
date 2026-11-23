// tests/kernels/bf16_bits_test.cpp - `bf16_bits.hpp` against ggml's rule, over EVERY f32 bit pattern (CPU only).
//
// the rounding add `i + 0x7FFF + bit16` carries out of the mantissa for a NaN, so 0x7FFFFFFF came back as
// -0 and 0x7F800001 as +inf - an upstream NaN was masked instead of propagated.  `ggml_compute_fp32_to_bf16`
// (ggml-impl.h at the pinned llama.cpp) tests the NaN first and returns `(i >> 16) | 64`.  Three checks, all
// exhaustive (2^32 inputs, a few seconds):
//
//   1. the header equals ggml's function, transcribed below, on every input;
//   2. on every NON-NaN input it equals the previous rounding, so the change cannot move a finite value or an
//      infinity by a single bit;
//   3. every NaN input gives a quiet NaN of the same sign.
//
// And the round trip over all 65,536 bf16 patterns: a non-NaN bf16 survives f32 and back unchanged.
//
// Then `f16_bits.hpp`'s edge cases against numpy's float16: a private encoder that tested `if (exp >= 31)` conflated an
// f32 inf/NaN (raw exponent 255) with a FINITE value too large for fp16, so every finite overflow came back as a NaN
// instead of saturating to inf; the matching decoder flushed fp16 SUBNORMALS to zero (0x0001 as 0.0, not 2^-24).  An
// O(1) fixture cannot see either.
#include "granite/kernels/bf16_bits.hpp"
#include "granite/kernels/f16_bits.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

/// ggml_compute_fp32_to_bf16, ggml/src/ggml-impl.h (MIT, the ggml authors), on raw bits.
uint16_t ggml_bf16(uint32_t u) {
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 64);
    return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

/// The rule before this fix, kept only to prove the change is confined to NaN inputs.
uint16_t previous_bf16(uint32_t u) {
    u = (u + ((u >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (u >> 16);
}

bool is_nan_bits(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }

}  // namespace

int main() {
    unsigned long long vs_ggml = 0, vs_previous = 0, nan_lost = 0, nan_inputs = 0, previous_masked = 0;
    uint32_t first_bad = 0;
    bool have_bad = false;
    for (uint64_t k = 0; k <= 0xFFFFFFFFull; ++k) {
        const uint32_t u = (uint32_t) k;
        float f;
        std::memcpy(&f, &u, 4);
        const uint16_t got = basalt::kernels::bf16_from_f32(f);
        if (got != ggml_bf16(u)) {
            ++vs_ggml;
            if (!have_bad) { first_bad = u; have_bad = true; }
        }
        if (is_nan_bits(u)) {
            ++nan_inputs;
            const uint32_t back = (uint32_t) got << 16;
            // a quiet NaN (bit 22 of the f32) with the input's sign
            if (!is_nan_bits(back) || (back & 0x00400000u) == 0 || (back >> 31) != (u >> 31)) ++nan_lost;
            if (!is_nan_bits((uint32_t) previous_bf16(u) << 16)) ++previous_masked;
        } else if (got != previous_bf16(u)) {
            ++vs_previous;
            if (!have_bad) { first_bad = u; have_bad = true; }
        }
    }
    std::printf("  f32 -> bf16 over 2^32 inputs: %llu differ from ggml, %llu non-NaN differ from the previous rule\n",
                vs_ggml, vs_previous);
    std::printf("  NaN inputs: %llu, not a quiet same-sign NaN now: %llu (the previous rule masked %llu of them)\n",
                nan_inputs, nan_lost, previous_masked);
    if (have_bad) std::printf("    first mismatch at input 0x%08x\n", first_bad);

    unsigned long long trip_bad = 0;
    for (uint32_t h = 0; h <= 0xFFFFu; ++h) {
        const float f = basalt::kernels::f32_from_bf16((uint16_t) h);
        const uint16_t back = basalt::kernels::bf16_from_f32(f);
        const bool nan = is_nan_bits(h << 16);
        if (back != (nan ? (uint16_t) (h | 64u) : (uint16_t) h)) ++trip_bad;
    }
    std::printf("  bf16 -> f32 -> bf16 over 65536 patterns: %llu wrong\n", trip_bad);

    // the exact values, asserted against what numpy's float16 says
    struct F16Case { float in; uint16_t want; const char* what; };
    const F16Case cases[] = {
        {1e30f, 0x7C00u, "1e30 -> +inf (finite overflow)"},
        {-1e30f, 0xFC00u, "-1e30 -> -inf"},
        {65536.0f, 0x7C00u, "65536 -> +inf (just past fp16 max)"},
        // FLT_MAX is 3.4028235e38, so `1e45f` is not even a legal literal; 3e38 is the largest finite
        // f32 that still overflows fp16, which is what this case is for.
        {3.0e38f, 0x7C00u, "3e38 (near FLT_MAX) -> +inf"},
        {65504.0f, 0x7BFFu, "65504 -> the largest finite fp16, NOT inf"},
        // The boundary, checked against numpy: 65520.0 is the exact tie between 65504 and the
        // "next" fp16 (65536 = inf) and rounds to EVEN, i.e. to inf; 65519.996 is below it and stays
        // finite.  The first version of this table asserted inf for 65519.996 - the FIXTURE was wrong and
        // the kernel was right, which is why the expected bits are quoted from numpy rather than reasoned.
        {65519.996f, 0x7BFFu, "65519.996 -> just below the boundary, still finite"},
        {65520.0f, 0x7C00u, "65520.0 -> the exact tie, rounds to even = inf"},
        {5.9604645e-8f, 0x0001u, "2^-24 -> the smallest subnormal, NOT zero"},
        {1.0f, 0x3C00u, "1.0"},
        {0.0f, 0x0000u, "0.0"},
    };
    int fp16_bad = 0;
    for (const F16Case& c : cases) {
        const uint16_t got = basalt::kernels::f16_from_f32(c.in);
        if (got != c.want) {
            std::printf("    *** %s: want 0x%04X got 0x%04X\n", c.what, c.want, got);
            ++fp16_bad;
        }
    }
    // and the reverse direction on the pattern that was flushed
    const float sub = basalt::kernels::f32_from_f16(0x0001u);
    const bool sub_ok = sub == 5.9604645e-8f;
    std::printf("  f32 -> f16 overflow saturates, not NaN: %d of %d wrong\n", fp16_bad,
                (int) (sizeof cases / sizeof cases[0]));
    std::printf("  f16 -> f32 keeps subnormals: 0x0001 -> %.9g (want 2^-24)\n", (double) sub);

    const bool ok = vs_ggml == 0 && vs_previous == 0 && nan_lost == 0 && trip_bad == 0 && previous_masked > 0 &&
                    fp16_bad == 0 && sub_ok;
    std::printf("\nbf16_bits_test %s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
