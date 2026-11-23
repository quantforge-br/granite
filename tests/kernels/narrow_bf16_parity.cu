// tests/kernels/narrow_bf16_parity.cu - the FP8 E4M3 and FP4 E2M1 conversions to bf16x2 (basalt/kernels/narrow_bf16.cuh)
// against the host's values: every pair of E4M3 bytes, and every E2M1 byte at every UE4M3 scale byte in each byte of the
// word, bit for bit (a NaN only as a NaN); and bf16x2_mul_rn against the host's rounding of the product on a sweep of
// bf16 pairs.  Exit 77 without a GPU.
#include "basalt/kernels/narrow_bf16.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bk = basalt::kernels;

namespace {

constexpr int kN = 1 << 16;

// i: two E4M3 bytes
__global__ void e4m3(unsigned* got) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    got[i] = bk::e4m3x2_bf16x2(i);
}

// the word for i: byte k is the low byte of i plus 37 k, so every byte position sees all 256 values; the high byte of i
// is the UE4M3 scale
__host__ __device__ unsigned e2m1_word(unsigned i) {
    unsigned w = 0;
    for (int k = 0; k < 4; ++k) w |= ((i + 37u * k) & 0xffu) << (8 * k);
    return w;
}

__global__ void e2m1(unsigned* got) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned s = i >> 8;
    unsigned h[4];
    bk::e2m1x8_bf16x2_scaled(e2m1_word(i), bk::e4m3x2_bf16x2(s | (s << 8)), h);
    for (int k = 0; k < 4; ++k) got[4 * i + k] = h[k];
}

// i: the low halves of a and b (the high halves the two swapped)
__global__ void mul(unsigned* got) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned a = (i & 0xffu) * 0x0101u + 0x3c00u, b = (i >> 8) * 0x0083u + 0x3800u;   // spread over a few binades
    got[i] = bk::bf16x2_mul_rn((a & 0xffffu) | (b << 16), (b & 0xffffu) | (a << 16));
}

float e4m3f(unsigned b) {
    __nv_fp8_e4m3 v;
    const uint8_t u = (uint8_t) b;
    std::memcpy(&v, &u, 1);
    return (float) v;
}

float e2m1f(unsigned n) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (n & 8) ? -mag[n & 7] : mag[n & 7];
}

bool bf16_nan(unsigned h) { return (h & 0x7f80u) == 0x7f80u && (h & 0x7fu); }

float bf16_f(unsigned h) {
    const unsigned u = (h & 0xffffu) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// the bf16 bits of v rounded to nearest even (v finite)
unsigned bf16_rn(float v) {
    unsigned u;
    std::memcpy(&u, &v, 4);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

int fails = 0;

// one half against the host's value (a NaN only as a NaN; signed zeros by their bits)
void check(const char* what, unsigned in, int half, unsigned got, float want) {
    const unsigned g = (got >> (16 * half)) & 0xffffu;
    const bool ok = want != want ? bf16_nan(g) : g == bf16_rn(want);
    if (!ok && fails++ < 8)
        std::printf("FAIL %s input 0x%x half %d: got 0x%04x, host 0x%04x\n", what, in, half, g, bf16_rn(want));
}

bool run(void (*k)(unsigned*), unsigned* d, std::vector<unsigned>& h, size_t n, const char* what) {
    k<<<kN / 256, 256>>>(d);
    if (cudaMemcpy(h.data(), d, n * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: %s kernel: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    unsigned* d = nullptr;
    if (cudaMalloc(&d, 4 * kN * sizeof(unsigned)) != cudaSuccess) return 77;
    std::vector<unsigned> g(4 * kN);

    if (!run(e4m3, d, g, kN, "e4m3")) return 1;
    for (unsigned i = 0; i < kN; ++i)
        for (int half = 0; half < 2; ++half) check("e4m3x2", i, half, g[i], e4m3f((i >> (8 * half)) & 0xffu));
    std::printf("e4m3x2: NaN 0x7f -> 0x%04x, 0xff -> 0x%04x\n", g[0x7f] & 0xffffu, g[0xff] & 0xffffu);

    if (!run(e2m1, d, g, 4 * (size_t) kN, "e2m1")) return 1;
    for (unsigned i = 0; i < kN; ++i) {
        const float sc = e4m3f(i >> 8);
        const unsigned w = e2m1_word(i);
        for (int k = 0; k < 4; ++k)
            for (int half = 0; half < 2; ++half)
                check("e2m1x8 scaled", i, half, g[4 * i + k], e2m1f((w >> (8 * k + 4 * half)) & 0xfu) * sc);
    }

    if (!run(mul, d, g, kN, "mul")) return 1;
    for (unsigned i = 0; i < kN; ++i) {
        const unsigned a = (i & 0xffu) * 0x0101u + 0x3c00u, b = (i >> 8) * 0x0083u + 0x3800u;
        check("bf16x2_mul_rn", i, 0, g[i], bf16_f(a) * bf16_f(b));   // exact in fp32, so one rounding to bf16
        check("bf16x2_mul_rn", i, 1, g[i], bf16_f(b) * bf16_f(a));
    }

    std::printf("%s: %d failures\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
