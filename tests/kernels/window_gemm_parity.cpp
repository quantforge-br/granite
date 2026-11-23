// tests/kernels/window_gemm_parity.cpp - a decode window's products (window_gemm.hpp) at the Inkling 2 drafter's widths,
// NVFP4 planes: against a double-precision reference on the bf16-packed activations, each row of a window bitwise the
// same whatever the window's rows (1..8) and whatever the other rows hold, the Add and SwiGLU epilogues, a call of
// several matrices, and the fused pack against rms_norm_rows followed by pack.
// --bench: us per call and GB/s of weights, the weights rotated through copies larger than L2.  Exit 77 without a GPU.
#include "granite/kernels/dense_ops.hpp"
#include "granite/kernels/nvfp4_planes.hpp"
#include "granite/kernels/pdl.hpp"
#include "granite/kernels/window_gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace wg = basalt::kernels::wgemm;
namespace np = basalt::kernels::nvfp4p;

namespace {

std::mt19937 rng(7);
int fails = 0;
constexpr int R = wg::kRows;

float e4m3(uint8_t b) {
    __nv_fp8_e4m3 v;
    std::memcpy(&v, &b, 1);
    return (float) v;
}
float e2m1(unsigned n) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (n & 8) ? -mag[n & 7] : mag[n & 7];
}
float bf16f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t bf16_of(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

std::vector<float> bf16_rows(const void* act, size_t n) {
    const auto h = down((const uint16_t*) act, n);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = bf16f(h[i]);
    return f;
}

// a matrix's bytes and its values as the reference reads them
struct Mat {
    int64_t rows, K;
    std::vector<uint8_t> bytes;
    float value(int64_t r, int64_t k) const {
        const uint8_t code = bytes[(size_t) r * (K / 2) + k / 2];
        const uint8_t sc = bytes[np::scale_offset(rows, K) + (size_t) np::scale_index(r, k / 16, K)];
        return e2m1((k & 1) ? code >> 4 : code & 15) * e4m3(sc);
    }
};
Mat make_mat(int64_t rows, int64_t K) {
    Mat m{rows, K, {}};
    m.bytes.resize(np::bytes(rows, K));
    const size_t so = np::scale_offset(rows, K);
    for (size_t i = 0; i < so; ++i) m.bytes[i] = (uint8_t) rng();
    for (size_t i = so; i < m.bytes.size(); ++i) m.bytes[i] = (uint8_t) (rng() % 0x7f);   // positive, finite, subnormals too
    return m;
}

struct Shape {
    int64_t rows, K;
    const char* name;
};
// the drafter's q, out and down
const Shape kShapes[] = {{3072, 2304, "FP4 q 3072x2304"}, {2304, 3072, "FP4 out 2304x3072"}, {2304, 9216, "FP4 down 2304x9216"}};

wg::Workspace ws;
void need(const wg::Call& c) {
    size_t p, t;
    wg::workspace_need(c, p, t);
    if (p > ws.partial_floats || t > ws.tickets_n) {
        cudaDeviceSynchronize();
        cudaFree(ws.partial);
        cudaFree(ws.tickets);
        ws.partial_floats = std::max(ws.partial_floats, p);
        ws.tickets_n = std::max(ws.tickets_n, t);
        cudaMalloc(&ws.partial, std::max<size_t>(ws.partial_floats, 1) * 4);
        cudaMalloc(&ws.tickets, ws.tickets_n * 4);
        cudaMemset(ws.tickets, 0, ws.tickets_n * 4);
    }
}

std::vector<float> make_x(int64_t K) {
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) R * K);
    for (int t = 0; t < R; ++t)
        for (int64_t k = 0; k < K; ++k) x[(size_t) t * K + k] = std::ldexp(1.0f, (3 * t) % 20 - 10) * nd(rng);
    x[(size_t) 3 * K + 17] = 3.0e4f;
    return x;
}

// max |y - ref| / sum |w x| over a sample of rows (ref in double on the packed activations)
double check_ref(const Mat& m, const std::vector<float>& hx, const std::vector<float>& fac, const std::vector<float>& y,
                 int64_t ld, double scale, int T) {
    double worst = 0;
    const int64_t step = m.rows > 1024 ? 3 : 1;
    for (int t = 0; t < T; ++t)
        for (int64_t r = 0; r < m.rows; r += step) {
            double s = 0, sa = 0;
            for (int64_t k = 0; k < m.K; ++k) {
                const double p = (double) m.value(r, k) * (double) hx[(size_t) t * m.K + k];
                s += p;
                sa += std::fabs(p);
            }
            s *= fac[t] * scale;
            sa *= fac[t] * scale;
            worst = std::max(worst, std::fabs(y[(size_t) t * ld + r] - s) / std::max(sa, 1e-30));
        }
    return worst;
}

void parity(const Shape& sh) {
    const Mat m = make_mat(sh.rows, sh.K);
    const auto x = make_x(sh.K);
    uint8_t* d_w = up(m.bytes);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, wg::act_bytes(sh.K));
    float *d_y, *d_y2, *d_sc;
    cudaMalloc(&d_y, (size_t) R * sh.rows * 4);
    cudaMalloc(&d_y2, (size_t) R * sh.rows * 4);
    d_sc = up(std::vector<float>{0.03f});
    wg::Call c;
    c.K = sh.K;
    c.m[0] = {d_w, sh.rows, d_sc, d_y, sh.rows};
    need(c);
    wg::pack(d_x, sh.K, sh.K, R, act, nullptr);
    wg::gemm(c, act, R, ws, nullptr);
    const auto yref = down(d_y, (size_t) R * sh.rows);
    const auto hx = bf16_rows(act, (size_t) R * sh.K);
    const std::vector<float> fac(R, 1.0f);
    const double worst = check_ref(m, hx, fac, yref, sh.rows, 0.03, R);
    const bool ok = worst < 5e-5;
    std::printf("%-24s vs double: max |err| / sum|w x| %.3g%s\n", sh.name, worst, ok ? "" : "  FAIL");
    if (!ok) ++fails;
    int diff = 0;
    std::normal_distribution<float> nd;
    for (int T = 1; T <= R; ++T) {
        std::vector<float> x2 = x;
        for (int t = T; t < R; ++t)
            for (int64_t k = 0; k < sh.K; ++k) x2[(size_t) t * sh.K + k] = 1e3f * nd(rng);
        float* d_x2 = up(x2);
        wg::pack(d_x2, sh.K, sh.K, T, act, nullptr);
        cudaMemset(d_y2, 0xFF, (size_t) R * sh.rows * 4);
        c.m[0].y = d_y2;
        wg::gemm(c, act, T, ws, nullptr);
        const auto yt = down(d_y2, (size_t) R * sh.rows);
        diff += std::memcmp(yt.data(), yref.data(), (size_t) T * sh.rows * 4) != 0;
        for (size_t i = (size_t) T * sh.rows; i < yt.size(); ++i) diff += !std::isnan(yt[i]);   // rows past T untouched
        cudaFree(d_x2);
    }
    std::printf("%-24s rows bitwise at T = 1..8: %s\n", sh.name, diff ? "DIFFER  FAIL" : "yes");
    if (diff) ++fails;
    const std::vector<float> base((size_t) R * sh.rows, 0.5f);
    cudaMemcpy(d_y2, base.data(), base.size() * 4, cudaMemcpyHostToDevice);
    wg::pack(d_x, sh.K, sh.K, R, act, nullptr);
    c.epi = wg::Epilogue::Add;
    wg::gemm(c, act, R, ws, nullptr);
    const auto ya = down(d_y2, (size_t) R * sh.rows);
    int bad = 0;
    for (size_t i = 0; i < ya.size(); ++i) bad += std::fabs(ya[i] - (0.5 + (double) yref[i])) > 1e-6 * (1.0 + std::fabs((double) yref[i]));
    std::printf("%-24s Add epilogue: %s\n", sh.name, bad ? "WRONG  FAIL" : "ok");
    if (bad) ++fails;
    cudaFree(d_w); cudaFree(d_x); cudaFree(act); cudaFree(d_y); cudaFree(d_y2); cudaFree(d_sc);
}

// gate and up as one SwiGLU call against the two products and the formula; rows bitwise across T
void parity_swiglu(int64_t rows, int64_t K) {
    const Mat g = make_mat(rows, K), u = make_mat(rows, K);
    const auto x = make_x(K);
    uint8_t *d_g = up(g.bytes), *d_u = up(u.bytes);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, wg::act_bytes(K));
    float *d_yg, *d_yu, *d_h, *d_sg, *d_su;
    cudaMalloc(&d_yg, (size_t) R * rows * 4);
    cudaMalloc(&d_yu, (size_t) R * rows * 4);
    cudaMalloc(&d_h, (size_t) R * rows * 4);
    d_sg = up(std::vector<float>{0.02f});
    d_su = up(std::vector<float>{0.05f});
    wg::pack(d_x, K, K, R, act, nullptr);
    wg::Call c;
    c.K = K;
    c.n = 2;
    c.epi = wg::Epilogue::SwiGLU;
    c.m[0] = {d_g, rows, d_sg, d_h, rows};
    c.m[1] = {d_u, rows, d_su, nullptr, rows};
    void* d_h16;
    cudaMalloc(&d_h16, wg::act_bytes(rows));
    c.act_out = d_h16;
    need(c);
    wg::gemm(c, act, R, ws, nullptr);
    const auto h16 = down(d_h, (size_t) R * rows);
    {   // the bf16 rows the SwiGLU epilogue leaves for the down product: h rounded
        const auto hb = down((const uint16_t*) d_h16, (size_t) R * rows);
        int bad = 0;
        for (size_t i = 0; i < hb.size(); ++i) bad += hb[i] != bf16_of(h16[i]);
        std::printf("%-24s SwiGLU bf16 rows == bf16(h): %s\n", "FP4 gate+up", bad ? "NO  FAIL" : "yes");
        if (bad) ++fails;
    }
    const auto hx = bf16_rows(act, (size_t) R * K);
    const std::vector<float> fac(R, 1.0f);
    double worst = 0;
    for (int t = 0; t < R; ++t)
        for (int64_t r = 0; r < rows; r += 29) {
            double sg = 0, su = 0, ag = 0, au = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double xv = hx[(size_t) t * K + k];
                sg += g.value(r, k) * xv;
                su += u.value(r, k) * xv;
                ag += std::fabs(g.value(r, k) * xv);
                au += std::fabs(u.value(r, k) * xv);
            }
            const double gv = sg * fac[t] * 0.02, uv = su * fac[t] * 0.05;
            const double ref = gv / (1.0 + std::exp(-gv)) * uv;
            const double tol = (ag * fac[t] * 0.02 * std::fabs(uv) + au * fac[t] * 0.05 * std::max(std::fabs(gv), 1.0)) + 1e-30;
            worst = std::max(worst, std::fabs(h16[(size_t) t * rows + r] - ref) / tol);
        }
    const bool ok = worst < 5e-5;
    std::printf("%-24s SwiGLU vs double: %.3g%s\n", "FP4 gate+up", worst, ok ? "" : "  FAIL");
    if (!ok) ++fails;
    int diff = 0;
    for (int T : {1, 3, 5, 8}) {
        cudaMemset(d_h, 0xFF, (size_t) R * rows * 4);
        wg::gemm(c, act, T, ws, nullptr);
        const auto ht = down(d_h, (size_t) R * rows);
        diff += std::memcmp(ht.data(), h16.data(), (size_t) T * rows * 4) != 0;
    }
    std::printf("%-24s SwiGLU rows bitwise across T: %s\n", "FP4 gate+up", diff ? "DIFFER  FAIL" : "yes");
    if (diff) ++fails;
    cudaFree(d_g); cudaFree(d_u); cudaFree(d_x); cudaFree(act); cudaFree(d_yg); cudaFree(d_yu); cudaFree(d_h);
    cudaFree(d_sg); cudaFree(d_su); cudaFree(d_h16);
}

// q, k, v as one call: each matrix against the reference
void parity_multi() {
    const int64_t K = 2304;
    const Mat q = make_mat(3072, K), k = make_mat(1024, K), v = make_mat(1024, K);
    const auto x = make_x(K);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, wg::act_bytes(K));
    wg::pack(d_x, K, K, R, act, nullptr);
    const Mat* ms[3] = {&q, &k, &v};
    wg::Call c;
    c.K = K;
    c.n = 3;
    uint8_t* dw[3];
    float* dy[3];
    for (int i = 0; i < 3; ++i) {
        dw[i] = up(ms[i]->bytes);
        cudaMalloc(&dy[i], (size_t) R * ms[i]->rows * 4);
        c.m[i] = {dw[i], ms[i]->rows, nullptr, dy[i], ms[i]->rows};
    }
    need(c);
    wg::gemm(c, act, R, ws, nullptr);
    const auto hx = bf16_rows(act, (size_t) R * K);
    const std::vector<float> fac(R, 1.0f);
    double worst = 0;
    for (int i = 0; i < 3; ++i) worst = std::max(worst, check_ref(*ms[i], hx, fac, down(dy[i], (size_t) R * ms[i]->rows), ms[i]->rows, 1.0, R));
    const bool ok = worst < 5e-5;
    std::printf("%-24s vs double: %.3g%s\n", "FP4 q,k,v in one call", worst, ok ? "" : "  FAIL");
    if (!ok) ++fails;
    for (int i = 0; i < 3; ++i) { cudaFree(dw[i]); cudaFree(dy[i]); }
    cudaFree(d_x); cudaFree(act);
}

// norm_pack against rms_norm_rows followed by pack, bit for bit
void parity_packs() {
    const int64_t K = 2304;
    const auto x = make_x(K);
    std::vector<float> w(K);
    std::normal_distribution<float> nd;
    for (auto& v : w) v = 1.0f + 0.1f * nd(rng);
    float *d_x = up(x), *d_w = up(w), *d_xn;
    cudaMalloc(&d_xn, (size_t) R * K * 4);
    void *a1, *a2;
    cudaMalloc(&a1, wg::act_bytes(K));
    cudaMalloc(&a2, wg::act_bytes(K));
    basalt::kernels::rms_norm_rows(d_x, d_w, d_xn, R, K, 1e-6f, nullptr);
    wg::pack(d_xn, K, K, R, a1, nullptr);
    wg::norm_pack(d_x, d_w, 1e-6f, K, R, a2, nullptr);
    const bool ok = down((const uint8_t*) a1, wg::act_bytes(K)) == down((const uint8_t*) a2, wg::act_bytes(K));
    std::printf("%-24s norm_pack == rms_norm_rows + pack: %s\n", "packs", ok ? "yes" : "NO  FAIL");
    if (!ok) ++fails;
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_xn); cudaFree(a1); cudaFree(a2);
}

void bench_call(const char* name, int64_t rows, int64_t K, bool swiglu) {
    const size_t wb = np::bytes(rows, K);
    const size_t per = swiglu ? 2 * wb : wb;
    const int copies = (int) std::max<size_t>(2, (512u << 20) / per + 1);
    std::vector<uint8_t*> d_w((size_t) copies * (swiglu ? 2 : 1));
    for (auto& p : d_w) {
        cudaMalloc(&p, wb);
        cudaMemset(p, 0x31, wb);
    }
    std::vector<float> x((size_t) R * K, 0.5f);
    float* d_x = up(x);
    void* act;
    cudaMalloc(&act, wg::act_bytes(K));
    float* d_y;
    cudaMalloc(&d_y, (size_t) R * rows * 4);
    wg::pack(d_x, K, K, R, act, nullptr);
    wg::Call c;
    c.K = K;
    if (swiglu) {
        c.n = 2;
        c.epi = wg::Epilogue::SwiGLU;
    }
    auto set = [&](int i) {
        if (swiglu) {
            c.m[0] = {d_w[2 * i], rows, nullptr, d_y, rows};
            c.m[1] = {d_w[2 * i + 1], rows, nullptr, nullptr, rows};
        } else {
            c.m[0] = {d_w[i], rows, nullptr, d_y, rows};
        }
    };
    set(0);
    need(c);
    {
        size_t p, t;
        wg::workspace_need(c, p, t);
        std::printf("%-24s tiles %zu, slices %zu\n", name, t, p ? p / R / (swiglu ? 2 * rows : rows) : 1);
    }
    basalt::kernels::PdlScope pdl(true);   // as a decode window launches them
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int T : {1, 8}) {
        int it = 0;
        for (int i = 0; i < copies; ++i) {
            set(it++ % copies);
            wg::gemm(c, act, T, ws, nullptr);
        }
        const int n = 6 * copies;
        cudaEventRecord(e0);
        for (int i = 0; i < n; ++i) {
            set(it++ % copies);
            wg::gemm(c, act, T, ws, nullptr);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / n;
        std::printf("%-24s %6.1f MB  T=%2d %7.1f us (%5.0f GB/s)\n", name, per / 1e6, T, us, per / us / 1e3);
    }
    for (auto p : d_w) cudaFree(p);
    cudaFree(d_x); cudaFree(act); cudaFree(d_y);
}

}  // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        // the Inkling 2 drafter's block products (a layer: q, k and v as one call, out, gate and up, down)
        bench_call("FP4 drafter qkv 5120x2304", 5120, 2304, false);
        bench_call("FP4 drafter out 2304x3072", 2304, 3072, false);
        bench_call("FP4 drafter gate+up 9216", 9216, 2304, true);
        bench_call("FP4 drafter down 2304x9216", 2304, 9216, false);
        return 0;
    }
    for (const Shape& sh : kShapes) parity(sh);
    parity_swiglu(9216, 2304);
    parity_multi();
    parity_packs();
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
}
