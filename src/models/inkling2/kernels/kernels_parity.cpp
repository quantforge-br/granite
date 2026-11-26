// src/models/inkling2/kernels/kernels_parity.cpp - the Inkling 2 kernels against plain CPU references (double accumulation)
// at the model's geometry: the block-residual mixes, the loop controllers' norm and blend, the attention preparation
// (short convolutions, clips, cache writes), the attention with its relative bias and sink over a window ring and a full
// cache, the post-sublayer convolutions, the experts with the softcapped SwiGLU and their weighted sum, and the softcap;
// and the decode window's fused kernels against the references and the kernels they replace, their q8_1 blocks against a
// CPU rounding of their own fp32 rows. Every kernel is checked for a row's bits not depending on the rows beside it, and
// timed; and the experts cut in the two halves tensor parallelism gives each card, against the whole experts. Exit 77
// without a GPU.
#include "granite/ggml/ggml.h"
#include "granite/kernels/iq_kernels.hpp"
#include "granite/kernels/native_mmvq.hpp"
#include "granite/kernels/router.hpp"
#include "granite/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace granite::models::inkling2;
namespace bk = granite::kernels;

namespace {

constexpr int E = kWidth;
std::mt19937 rng(23);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.f, float mean = 0.f) {
    std::normal_distribution<float> d(mean, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T) + 16);
    cudaMemset(p, 0, n * sizeof(T) + 16);
    return p;
}

void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-34s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void same_bits(const char* what, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    std::printf("%-34s %s\n", what, ok ? "bitwise" : "FAIL");
    failures += !ok;
}
// microseconds a call, averaged over `n` calls after one warm-up
double time_us(const std::function<void()>& f, int n = 50) {
    f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}
double h2d(uint16_t h) {
    __half_raw r;
    r.x = h;
    return (double) __half2float(__half(r));
}
uint16_t f2h(float f) { return __half_raw(__float2half(f)).x; }
double tanhd(double x) { return std::tanh(x); }
double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The block-residual mix of row r of `st` (each [rows][E]) scored by the query q[0..E).
std::vector<double> mix_ref(const std::vector<const std::vector<float>*>& st, const float* q, int r) {
    const size_t n = st.size();
    std::vector<double> inv(n), sc(n);
    double m = -1e300;
    for (size_t i = 0; i < n; ++i) {
        double ss = 0, d = 0;
        for (int c = 0; c < E; ++c) {
            const double x = (*st[i])[(size_t) r * E + c];
            ss += x * x;
            d += (double) q[c] * x;
        }
        inv[i] = 1.0 / std::sqrt(ss / E + 1e-6);
        sc[i] = d * inv[i] / std::sqrt((double) E);
        m = std::max(m, sc[i]);
    }
    double den = 0;
    for (double& v : sc) den += (v = std::exp(v - m));
    std::vector<double> out(E, 0.0);
    for (int c = 0; c < E; ++c)
        for (size_t i = 0; i < n; ++i) out[c] += sc[i] / den * (*st[i])[(size_t) r * E + c] * inv[i];
    return out;
}
std::vector<double> norm_ref(const std::vector<double>& x, const std::vector<float>& w, double eps = 1e-6) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / (double) x.size() + eps);
    std::vector<double> y(x.size());
    for (size_t c = 0; c < x.size(); ++c) y[c] = x[c] * inv * w[c];
    return y;
}

// The q8_1 blocks of `rows` rows of `width` against a CPU rounding of the kernel's own fp32 rows `x`: the int8 values and
// the scale exactly, the sum to its fp16 rounding.
void check_q8(const char* what, const void* dq8, const std::vector<float>& x, int rows, int width) {
    const size_t nb = (size_t) rows * width / 32;
    std::vector<uint8_t> q = host(static_cast<const uint8_t*>(dq8), nb * 36);
    bool ok = true;
    for (size_t b = 0; b < nb && ok; ++b) {
        const float* v = &x[b * 32];
        float amax = 0.f, sum = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = std::max(amax, std::fabs(v[i]));
            sum += v[i];
        }
        const float d = amax / 127.0f;
        uint16_t dh, sh;
        std::memcpy(&dh, &q[b * 36], 2);
        std::memcpy(&sh, &q[b * 36 + 2], 2);
        ok = dh == f2h(d) && std::fabs(h2d(sh) - (double) sum) <= 1e-3 * std::max(1.0, std::fabs((double) sum));
        for (int i = 0; i < 32 && ok; ++i) {
            const float t = amax == 0.f ? 0.f : std::max(-127.f, std::min(127.f, std::round(v[i] / d)));
            ok = (int8_t) q[b * 36 + 4 + i] == (int8_t) t;
        }
    }
    std::printf("%-34s %s\n", what, ok ? "ok" : "FAIL");
    failures += !ok;
}

// ---- block residuals ---------------------------------------------------------------------------------------------------

void test_mix() {
    for (int n_q : {1, 3, 4}) {
        const int R = 5, n = n_q == 4 ? 2 : 5;
        std::vector<std::vector<float>> st(n);
        std::vector<float*> ds(n);
        for (int i = 0; i < n; ++i) {
            st[i] = randn((size_t) R * E, 0.5f + 0.3f * i);
            ds[i] = dev(st[i]);
        }
        const std::vector<float> q = randn((size_t) n_q * E, 0.05f);
        float* dq = dev(q);
        float* out = zeros<float>((size_t) n_q * R * E);
        MixStates ms;
        for (int i = 0; i < n; ++i) ms.s[ms.n++] = ds[i];
        attnres_mix(ms, dq, n_q, R, 1e-6f, out, (int64_t) R * E, nullptr);
        std::vector<float> got = host(out, (size_t) n_q * R * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r) {
            std::vector<double> inv(n);
            for (int i = 0; i < n; ++i) {
                double ss = 0;
                for (int c = 0; c < E; ++c) ss += (double) st[i][(size_t) r * E + c] * st[i][(size_t) r * E + c];
                inv[i] = 1.0 / std::sqrt(ss / E + 1e-6);
            }
            for (int j = 0; j < n_q; ++j) {
                std::vector<double> sc(n);
                double m = -1e300;
                for (int i = 0; i < n; ++i) {
                    double d = 0;
                    for (int c = 0; c < E; ++c) d += (double) q[(size_t) j * E + c] * st[i][(size_t) r * E + c];
                    sc[i] = d * inv[i] / std::sqrt((double) E);
                    m = std::max(m, sc[i]);
                }
                double den = 0;
                for (double& s : sc) den += (s = std::exp(s - m));
                for (int c = 0; c < E; ++c) {
                    double acc = 0;
                    for (int i = 0; i < n; ++i) acc += sc[i] / den * st[i][(size_t) r * E + c] * inv[i];
                    ref[((size_t) j * R + r) * E + c] = acc;
                }
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "attnres_mix (%d states, %d queries)", n, n_q);
        report(name, got, ref, 2e-6);
        MixStates one;   // row 3 alone
        for (int i = 0; i < n; ++i) one.s[one.n++] = ds[i] + 3 * E;
        float* out1 = zeros<float>((size_t) n_q * E);
        attnres_mix(one, dq, n_q, 1, 1e-6f, out1, E, nullptr);
        std::vector<float> g1 = host(out1, (size_t) n_q * E);
        bool same = true;
        for (int j = 0; j < n_q; ++j) same = same && !std::memcmp(&g1[(size_t) j * E], &got[((size_t) j * R + 3) * E], E * 4);
        std::printf("%-34s %s\n", "  a row alone", same ? "bitwise" : "FAIL");
        failures += !same;
        if (n_q == 1)
            std::printf("  %.1f us a call (1 row, %d states)\n", time_us([&] { attnres_mix(one, dq, 1, 1, 1e-6f, out1, E, nullptr); }), n);
    }
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

void test_ctrl() {
    const int R = 3;
    const std::vector<float> a = randn((size_t) R * E, 0.7f), b = randn((size_t) R * E, 1.3f), gf = randn((size_t) R * 2 * E, 0.8f);
    float *da = dev(a), *db = dev(b), *dgf = dev(gf), *pair = zeros<float>((size_t) R * 2 * E), *out = zeros<float>((size_t) R * E);
    ctrl_pair_norm(da, db, R, pair, nullptr);
    ctrl_blend(da, db, dgf, R, out, nullptr);
    std::vector<float> gp = host(pair, (size_t) R * 2 * E), go = host(out, (size_t) R * E);
    std::vector<double> rp(gp.size()), ro(go.size());
    for (int r = 0; r < R; ++r) {
        double sa = 0, sb = 0;
        for (int c = 0; c < E; ++c) {
            sa += (double) a[(size_t) r * E + c] * a[(size_t) r * E + c];
            sb += (double) b[(size_t) r * E + c] * b[(size_t) r * E + c];
        }
        const double ia = 1 / std::sqrt(sa / E + 1e-6), ib = 1 / std::sqrt(sb / E + 1e-6);
        for (int c = 0; c < E; ++c) {
            rp[(size_t) r * 2 * E + c] = a[(size_t) r * E + c] * ia;
            rp[(size_t) r * 2 * E + E + c] = b[(size_t) r * E + c] * ib;
            const double g = tanhd(gf[(size_t) r * 2 * E + c]), f = tanhd(gf[(size_t) r * 2 * E + E + c]);
            ro[(size_t) r * E + c] = (1 - f) * a[(size_t) r * E + c] + g * b[(size_t) r * E + c];
        }
    }
    report("ctrl_pair_norm", gp, rp, 2e-6);
    report("ctrl_blend", go, ro, 2e-6);
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

// The CPU reference of one row's convolution input at position p - j (window rows, else the ring, else 0).
double conv_input(const std::vector<float>& y, int ld, const std::vector<float>& ring, int width, const std::vector<int>& pos, int r, int j,
                  int c) {
    if (r - j >= 0) return y[(size_t) (r - j) * ld + c];
    const int p = pos[r] - j;
    if (p < 0) return 0.0;
    return ring[(size_t) (p % kConvRing) * width + c];
}

void test_prepare() {
    for (bool full : {false, true}) {
        const int R = 3, p0 = full ? 9000 : 2;   // full: past the log scale's floor; a window row: near the start (zero history)
        const int64_t cap = full ? 9100 : kExtent + 8;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        const std::vector<float> yq = randn((size_t) R * kProjQ, 1.5f), yk = randn((size_t) R * kProjKV, 2.0f), yv = randn((size_t) R * kProjKV, 0.6f);
        const std::vector<float> kc = randn(1024 * kConv, 0.3f), vc = randn(1024 * kConv, 0.3f);
        const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rs = randn(kHeads * kRel, 0.2f, 1.f);
        const std::vector<float> kr = randn((size_t) kConvRing * 1024, 2.0f), vr = randn((size_t) kConvRing * 1024, 0.6f);
        float *dyq = dev(yq), *dyk = dev(yk), *dyv = dev(yv), *dkc = dev(kc), *dvc = dev(vc), *dqs = dev(qs), *drs = dev(rs);
        float *dkr = dev(kr), *dvr = dev(vr);
        int* dpos = dev(pos);
        uint16_t *kcache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim), *vcache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim);
        float *q = zeros<float>((size_t) R * kHeads * kHeadDim), *rr = zeros<float>((size_t) R * kHeads * kRel), *sink = zeros<float>((size_t) R * kHeads);
        AttnPrep a;
        a.yq = dyq; a.yk = dyk; a.yv = dyv; a.pos = dpos; a.full = full; a.k_conv = dkc; a.v_conv = dvc; a.q_scale = dqs; a.r_scale = drs;
        a.k_ring = dkr; a.v_ring = dvr; a.k_cache = kcache; a.v_cache = vcache; a.cap = cap; a.q = q; a.r = rr; a.sink = sink;
        attn_prepare(a, R, nullptr);
        cudaDeviceSynchronize();
        // keys and values
        std::vector<float> gk, gv;
        std::vector<double> rk, rv;
        const std::vector<uint16_t> hk = host(kcache, (size_t) kKvHeads * cap * kHeadDim), hv = host(vcache, (size_t) kKvHeads * cap * kHeadDim);
        for (int r = 0; r < R; ++r)
            for (int which = 0; which < 2; ++which) {
                const std::vector<float>& y = which ? yv : yk;
                const std::vector<float>& taps = which ? vc : kc;
                const std::vector<float>& ring = which ? vr : kr;
                for (int h = 0; h < kKvHeads; ++h) {
                    std::vector<double> val(kHeadDim);
                    double ss = 0;
                    for (int d = 0; d < kHeadDim; ++d) {
                        const int c = h * kHeadDim + d;
                        double acc = 0;
                        for (int j = 0; j < kConv; ++j) {
                            double t = taps[(size_t) c * kConv + (kConv - 1 - j)];
                            if (full) t += y[(size_t) r * kProjKV + kOffTaps + h * kConv + j];
                            acc += t * conv_input(y, kProjKV, ring, 1024, pos, r, j, c);
                        }
                        val[d] = (acc + y[(size_t) r * kProjKV + c]) / std::sqrt(full ? 9.0 : 5.0);
                        ss += val[d] * val[d];
                    }
                    const double f = which == 0 ? std::min(22.627417 / (std::sqrt(ss) + 1e-6), 1.0) : 1.0 / std::sqrt(ss / kHeadDim + 1e-6);
                    for (int d = 0; d < kHeadDim; ++d) {
                        const size_t at = ((size_t) (pos[r] % cap) * kKvHeads + h) * kHeadDim + d;
                        (which ? gv : gk).push_back((float) h2d((which ? hv : hk)[at]));
                        (which ? rv : rk).push_back(val[d] * f);
                    }
                }
            }
        report(full ? "attn_prepare keys (full, fp16)" : "attn_prepare keys (window, fp16)", gk, rk, 1e-3);
        report(full ? "attn_prepare values (full, fp16)" : "attn_prepare values (window, fp16)", gv, rv, 1e-3);
        // queries, relative states, sinks
        const std::vector<float> gq = host(q, (size_t) R * kHeads * kHeadDim), gr = host(rr, (size_t) R * kHeads * kRel), gs = host(sink, (size_t) R * kHeads);
        std::vector<double> rq(gq.size()), rr2(gr.size()), rsk(gs.size());
        for (int r = 0; r < R; ++r) {
            const double tau = full ? 1.0 + 0.1 * std::log(std::max((pos[r] + 1.0) / 8192.0, 1.0)) : 1.0;
            for (int h = 0; h < kHeads; ++h) {
                double ss = 0;
                for (int d = 0; d < kHeadDim; ++d) {
                    const double v = (double) yq[(size_t) r * kProjQ + h * kHeadDim + d] * qs[h * kHeadDim + d];
                    ss += v * v;
                }
                const double f = std::min(22.627417 / (std::sqrt(ss) + 1e-6), 1.0) * tau;
                for (int d = 0; d < kHeadDim; ++d)
                    rq[((size_t) r * kHeads + h) * kHeadDim + d] = (double) yq[(size_t) r * kProjQ + h * kHeadDim + d] * qs[h * kHeadDim + d] * f;
                double s2 = 0;
                for (int i = 0; i < kRel; ++i) {
                    const double v = (double) yq[(size_t) r * kProjQ + kOffR + h * kRel + i] * rs[h * kRel + i];
                    s2 += v * v;
                }
                const double fr = std::min(8.0 / (std::sqrt(s2) + 1e-6), 1.0) * tau;
                for (int i = 0; i < kRel; ++i)
                    rr2[((size_t) r * kHeads + h) * kRel + i] = (double) yq[(size_t) r * kProjQ + kOffR + h * kRel + i] * rs[h * kRel + i] * fr;
                rsk[(size_t) r * kHeads + h] = yq[(size_t) r * kProjQ + kOffSink + h];
            }
        }
        report("attn_prepare queries", gq, rq, 2e-6);
        report("attn_prepare relative states", gr, rr2, 2e-6);
        report("attn_prepare sinks", gs, rsk, 0);
        // the rings took the window's inputs
        const std::vector<float> nkr = host(dkr, (size_t) kConvRing * 1024);
        bool ring_ok = true;
        for (int r = 0; r < R; ++r) ring_ok = ring_ok && !std::memcmp(&nkr[(size_t) (pos[r] % kConvRing) * 1024], &yk[(size_t) r * kProjKV], 1024 * 4);
        std::printf("%-34s %s\n", "  the key ring", ring_ok ? "ok" : "FAIL");
        failures += !ring_ok;
        if (!full)
            std::printf("  %.1f us a call (1 row)\n", time_us([&] { attn_prepare(a, 1, nullptr); }));
        // Direct KvInt8 g64 producer parity: verify attn_prepare codes, fp16 scales, and dequantized values against CPU ref.
        int8_t *kcache_i8 = zeros<int8_t>((size_t) kKvHeads * cap * kHeadDim), *vcache_i8 = zeros<int8_t>((size_t) kKvHeads * cap * kHeadDim);
        uint16_t *kscale = zeros<uint16_t>((size_t) kKvHeads * cap * (kHeadDim / 64)), *vscale = zeros<uint16_t>((size_t) kKvHeads * cap * (kHeadDim / 64));
        float *dkr2 = dev(kr), *dvr2 = dev(vr);
        AttnPrep ai = a;
        ai.k_ring = dkr2; ai.v_ring = dvr2;
        ai.k_cache = kcache_i8; ai.v_cache = vcache_i8;
        ai.k_scale = kscale; ai.v_scale = vscale;
        ai.fp8 = false;
        attn_prepare(ai, R, nullptr);
        cudaDeviceSynchronize();
        const std::vector<int8_t> hki = host(kcache_i8, (size_t) kKvHeads * cap * kHeadDim), hvi = host(vcache_i8, (size_t) kKvHeads * cap * kHeadDim);
        const std::vector<uint16_t> hks = host(kscale, (size_t) kKvHeads * cap * (kHeadDim / 64)), hvs = host(vscale, (size_t) kKvHeads * cap * (kHeadDim / 64));
        bool i8_ok = true;
        double i8_max_deq = 0.0;
        for (int r = 0; r < R; ++r) {
            for (int which = 0; which < 2; ++which) {
                const std::vector<double>& ref = which ? rv : rk;
                const std::vector<int8_t>& hcode = which ? hvi : hki;
                const std::vector<uint16_t>& hsc = which ? hvs : hks;
                for (int h = 0; h < kKvHeads; ++h) {
                    const size_t base = ((size_t) (pos[r] % cap) * kKvHeads + h) * kHeadDim;
                    const size_t sbase = ((size_t) (pos[r] % cap) * kKvHeads + h) * (kHeadDim / 64);
                    for (int grp = 0; grp < kHeadDim / 64; ++grp) {
                        double amax = 0.0;
                        for (int d = 0; d < 64; ++d)
                            amax = std::max(amax, std::fabs(ref[((size_t) r * kKvHeads + h) * kHeadDim + grp * 64 + d]));
                        const double ref_s = amax / 127.0;
                        const double got_s = h2d(hsc[sbase + grp]);
                        if (std::fabs(got_s - ref_s) > 1e-3 * std::max(1.0, ref_s) + 1e-5) i8_ok = false;
                        for (int d = 0; d < 64; ++d) {
                            const double v = ref[((size_t) r * kKvHeads + h) * kHeadDim + grp * 64 + d];
                            const int qref = amax > 0.0 ? std::max(-127, std::min(127, (int) std::lrint(v / ref_s))) : 0;
                            const int qgot = (int) hcode[base + grp * 64 + d];
                            if (std::abs(qgot - qref) > 1) i8_ok = false;
                            i8_max_deq = std::max(i8_max_deq, std::fabs(qgot * got_s - v));
                        }
                    }
                }
            }
        }
        i8_ok = i8_ok && i8_max_deq < 0.08;
        std::printf("%-34s %s (max_deq=%.4g)\n", full ? "attn_prepare KvInt8 g64 (full)" : "attn_prepare KvInt8 g64 (window)",
                    i8_ok ? "ok" : "FAIL", i8_max_deq);
        failures += !i8_ok;
    }
}

// ---- attention ---------------------------------------------------------------------------------------------------------

void test_attention() {
    for (int mode = 0; mode < 3; ++mode) {   // a window ring, a full cache, a full cache of E4M3 codes
        const bool window = mode == 0, fp8 = mode == 2;
        const int R = 4;
        const int p0 = window ? 2900 : 2995;
        const int64_t cap = window ? kExtent + 8 : 3000;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        std::vector<double> kd((size_t) kKvHeads * cap * kHeadDim), vd(kd.size());   // the cached values as the kernel reads them
        std::vector<uint16_t> kh(kd.size()), vh(kd.size());
        std::vector<uint8_t> k8(kd.size()), v8(kd.size());
        for (size_t i = 0; i < kd.size(); ++i) {
            const float kf = randn(1, 0.6f)[0], vf = randn(1, 1.0f)[0];
            if (fp8) {
                const __nv_fp8_e4m3 a(kf), b(vf);
                k8[i] = a.__x;
                v8[i] = b.__x;
                kd[i] = (double) (float) a;
                vd[i] = (double) (float) b;
            } else {
                kh[i] = f2h(kf);
                vh[i] = f2h(vf);
                kd[i] = h2d(kh[i]);
                vd[i] = h2d(vh[i]);
            }
        }
        const std::vector<float> q = randn((size_t) R * kHeads * kHeadDim, 0.8f), rs = randn((size_t) R * kHeads * kRel, 1.0f);
        const std::vector<float> sink = randn((size_t) R * kHeads, 2.0f), table = randn((size_t) kRel * kExtent, 0.3f);
        const std::vector<float> gate = randn((size_t) R * kProjQ, 1.5f);
        void* dk = fp8 ? (void*) dev(k8) : (void*) dev(kh);
        void* dv = fp8 ? (void*) dev(v8) : (void*) dev(vh);
        float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table), *out = zeros<float>((size_t) R * kHeads * kHeadDim);
        float* dgate = dev(gate);
        int* dpos = dev(pos);
        AttnArgs a;
        a.q = dq; a.r = drs; a.sink = dsink; a.rel_table = dt; a.pos = dpos; a.k_cache = dk; a.v_cache = dv; a.cap = cap; a.fp8 = fp8;
        a.window = window ? kExtent : 0;
        const int64_t max_keys = window ? kExtent : p0 + R;
        void* scr = zeros<uint8_t>(attention_decode_scratch(R, 1 << 20));
        attention_decode(a, R, max_keys, scr, out, nullptr);
        const std::vector<float> got = host(out, (size_t) R * kHeads * kHeadDim);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r)
            for (int h = 0; h < kHeads; ++h) {
                const int g = h / 3;
                const int lo = window ? std::max(0, pos[r] - kExtent + 1) : 0;
                std::vector<double> sc;
                double m = sink[(size_t) r * kHeads + h];
                for (int k = lo; k <= pos[r]; ++k) {
                    double d = 0;
                    for (int i = 0; i < kHeadDim; ++i) d += (double) q[((size_t) r * kHeads + h) * kHeadDim + i] * kd[((size_t) (k % cap) * kKvHeads + g) * kHeadDim + i];
                    d *= a.scale;
                    const int dist = pos[r] - k;
                    if (dist < kExtent)
                        for (int i = 0; i < kRel; ++i) d += (double) rs[((size_t) r * kHeads + h) * kRel + i] * table[(size_t) i * kExtent + dist];
                    sc.push_back(d);
                    m = std::max(m, d);
                }
                double den = std::exp(sink[(size_t) r * kHeads + h] - m);
                for (double& s : sc) den += (s = std::exp(s - m));
                for (int i = 0; i < kHeadDim; ++i) {
                    double acc = 0;
                    for (int k = lo; k <= pos[r]; ++k) acc += sc[(size_t) (k - lo)] * vd[((size_t) (k % cap) * kKvHeads + g) * kHeadDim + i];
                    ref[((size_t) r * kHeads + h) * kHeadDim + i] = acc / den;
                }
            }
        report(window ? "attention (window ring)" : fp8 ? "attention (full, 3k keys, E4M3)" : "attention (full, 3k keys)", got, ref, 2e-3);
        // row 2 alone, and under a launch bound of a million keys
        AttnArgs one = a;
        one.q = dq + 2 * kHeads * kHeadDim; one.r = drs + 2 * kHeads * kRel; one.sink = dsink + 2 * kHeads; one.pos = dpos + 2;
        float* out1 = zeros<float>((size_t) kHeads * kHeadDim);
        attention_decode(one, 1, max_keys, scr, out1, nullptr);
        std::vector<float> g1 = host(out1, (size_t) kHeads * kHeadDim);
        same_bits("  a row alone", g1.data(), &got[(size_t) 2 * kHeads * kHeadDim], g1.size() * 4);
        attention_decode(one, 1, 1 << 20, scr, out1, nullptr);
        g1 = host(out1, (size_t) kHeads * kHeadDim);
        same_bits("  any launch bound", g1.data(), &got[(size_t) 2 * kHeads * kHeadDim], g1.size() * 4);
        // rows whose slices start too far apart to share their keys' loads: each row's bits as alone
        const std::vector<int> apart = {p0 + 3, p0 - 37, p0 + 1, p0 - 20};
        int* dap = dev(apart);
        AttnArgs b = a;
        b.pos = dap;
        float* outa = zeros<float>((size_t) R * kHeads * kHeadDim);
        attention_decode(b, R, max_keys, scr, outa, nullptr);
        const std::vector<float> ga = host(outa, (size_t) R * kHeads * kHeadDim);
        std::vector<float> alone(ga.size());
        for (int r = 0; r < R; ++r) {
            AttnArgs o = b;
            o.q = dq + (size_t) r * kHeads * kHeadDim; o.r = drs + (size_t) r * kHeads * kRel; o.sink = dsink + (size_t) r * kHeads;
            o.pos = dap + r;
            attention_decode(o, 1, max_keys, scr, out1, nullptr);
            const std::vector<float> g = host(out1, (size_t) kHeads * kHeadDim);
            std::copy(g.begin(), g.end(), alone.begin() + (size_t) r * kHeads * kHeadDim);
        }
        same_bits("  rows far apart, each alone", alone.data(), ga.data(), ga.size() * 4);
        // gated, and as q8_1 blocks
        AttnArgs gq = a;
        gq.gate = dgate + kOffGate;
        gq.gate_ld = kProjQ;
        gq.q8 = zeros<uint8_t>((size_t) R * kHeads * kHeadDim / 32 * 36);
        float* outg = zeros<float>((size_t) R * kHeads * kHeadDim);
        attention_decode(gq, R, max_keys, scr, outg, nullptr);
        const std::vector<float> gg = host(outg, (size_t) R * kHeads * kHeadDim);
        std::vector<float> want(gg.size());
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < kHeads * kHeadDim; ++c) {
                const size_t i = (size_t) r * kHeads * kHeadDim + c;
                want[i] = got[i] * (1.f / (1.f + std::exp(-gate[(size_t) r * kProjQ + kOffGate + c])));
            }
        std::vector<double> wd(want.begin(), want.end());
        report("  gated", gg, wd, 1e-6);
        check_q8("  gated, its q8_1 blocks", gq.q8, gg, R, kHeads * kHeadDim);
        // tensor parallelism: each half of the key/value groups alone, its heads' blocks as rows of their own
        {
            const size_t row = (size_t) kHeads * kHeadDim / 32 * 36, half_row = row / 2, q8b = (size_t) R * half_row;
            uint8_t* qh[2] = {zeros<uint8_t>(q8b), zeros<uint8_t>(q8b)};
            uint32_t* sent = zeros<uint32_t>(2);
            for (int half = 0; half < 2; ++half) {
                AttnArgs t = gq;
                t.g0 = half * kKvHeads / 2;
                t.ng = kKvHeads / 2;
                t.q8 = qh[half];
                t.sent = sent + half;
                attention_decode(t, R, max_keys, scr, nullptr, nullptr);
            }
            const std::vector<uint8_t> whole = host(static_cast<const uint8_t*>(gq.q8), (size_t) R * row);
            for (int half = 0; half < 2; ++half) {
                std::vector<uint8_t> want_h(q8b);
                for (int r = 0; r < R; ++r)
                    std::memcpy(&want_h[(size_t) r * half_row], &whole[(size_t) r * row + half * half_row], half_row);
                same_bits(half ? "  halves of the groups, card 1" : "  halves of the groups, card 0", host(qh[half], q8b).data(),
                          want_h.data(), q8b);
            }
            const std::vector<uint32_t> n = host(sent, 2);
            const bool counted = n[0] == 1 && n[1] == 1;
            std::printf("%-34s %s\n", "  halves counted as sent", counted ? "ok" : "FAIL");
            failures += !counted;
        }
        const double us = time_us([&] { attention_decode(one, 1, max_keys, scr, out1, nullptr); });
        const double keys = window ? kExtent : p0 + 3;
        std::printf("  %.1f us a call (1 row, %.0f keys: %.0f GB/s of K and V)\n", us, keys,
                    keys * kKvHeads * kHeadDim * (fp8 ? 1 : 2) * 2 / us / 1e3);
        if (mode == 1) {   // a short context: the latency a decode token sees early on
            std::vector<int> p1 = {40};
            int* dp1 = dev(p1);
            AttnArgs s1 = one;
            s1.pos = dp1;
            std::printf("  %.1f us a call (1 row, 41 keys)\n", time_us([&] { attention_decode(s1, 1, 1024, scr, out1, nullptr); }));
        }
    }
}

// ---- post-sublayer convolutions ----------------------------------------------------------------------------------------

void test_post_conv() {
    const int R = 4;
    std::vector<int> pos = {40, 41, 42, 43};
    const std::vector<float> x = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f), ring = randn((size_t) kConvRing * E, 0.4f),
                             add = randn((size_t) R * E, 1.0f);
    float *dx = dev(x), *dt = dev(taps), *dring = dev(ring), *dadd = dev(add), *out = zeros<float>((size_t) R * E);
    int* dpos = dev(pos);
    post_conv(dx, dpos, dt, dring, dadd, R, out, nullptr);
    const std::vector<float> got = host(out, (size_t) R * E);
    std::vector<double> ref(got.size());
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < E; ++c) {
            double acc = 0;
            for (int j = 0; j < kConv; ++j) acc += (double) taps[(size_t) c * kConv + (kConv - 1 - j)] * conv_input(x, E, ring, E, pos, r, j, c);
            ref[(size_t) r * E + c] = (acc + x[(size_t) r * E + c]) / std::sqrt(5.0) + add[(size_t) r * E + c];
        }
    report("post_conv", got, ref, 2e-6);
}

// ---- experts -----------------------------------------------------------------------------------------------------------

void test_experts() {
    const int H = E, FF = 1152, NE = 6, R = 3, K = 4;
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    bk::NativeExpertLayout L = bk::native_expert_layout(40, 40, H, FF);
    L.gate_softcap = 4.f;
    L.up_softcap = 25.f;
    std::vector<uint8_t> blobs((size_t) NE * L.bytes);
    std::vector<std::vector<float>> wg(NE), wu(NE), wd(NE);   // the dequantised weights
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * L.bytes;
        auto put = [&](std::vector<float>& deq, int rows, int k, size_t row_bytes, uint8_t* dst, float s) {
            const std::vector<float> w = randn((size_t) rows * k, s);
            deq.resize((size_t) rows * k);
            for (int r = 0; r < rows; ++r) {
                tr->from_float_ref(&w[(size_t) r * k], dst + (size_t) r * row_bytes, k);
                tr->to_float(dst + (size_t) r * row_bytes, &deq[(size_t) r * k], k);
            }
        };
        put(wg[e], FF, H, L.gu_row, b, 0.05f);
        put(wu[e], FF, H, L.up_row, b + L.up_off, 0.05f);
        put(wd[e], H, FF, L.d_row, b + L.down_off, 0.05f);
        const float tail[4] = {1.f, 1.f, 1.f, 0.f};
        std::memcpy(b + L.bytes - 16, tail, 16);
    }
    std::vector<int32_t> ids = {0, 2, 5, 1, 2, 3, 4, 0, 5, 1, 0, 3};   // [R][K]
    const std::vector<float> w = randn((size_t) R * K, 0.5f, 1.f), x = randn((size_t) R * H, 1.0f);
    uint8_t* dblobs = dev(blobs);
    std::vector<unsigned long long> table(NE);
    for (int e = 0; e < NE; ++e) table[(size_t) e] = (unsigned long long) (dblobs + (size_t) e * L.bytes);
    unsigned long long* dtable = dev(table);
    int32_t *dids = dev(ids), *start = zeros<int32_t>(R * K + 2), *ng = zeros<int32_t>(4), *dst = zeros<int32_t>(R * K), *tok = zeros<int32_t>(R * K);
    unsigned long long* ptr = zeros<unsigned long long>(R * K);
    float *dw = dev(w), *dx = dev(x), *rows = zeros<float>((size_t) R * K * H), *y = zeros<float>((size_t) R * H);
    void* xq = zeros<uint8_t>(bk::native_q8_1_bytes(H, R));
    void* scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(R * K, FF));
    cudaStream_t st;
    cudaStreamCreate(&st);
    moe_plan(dids, R, K, NE, dtable, ptr, start, ng, dst, tok, st);
    bk::native_quantize_q8_1(dx, xq, H, R, st);
    bk::native_expert_grouped(L, ptr, start, ng, dst, tok, NE, R * K, xq, scr, rows, st);
    moe_combine(rows, dw, R, K, y, st);
    cudaStreamSynchronize(st);
    const std::vector<float> got = host(y, (size_t) R * H);
    std::vector<double> ref(got.size(), 0.0);
    for (int r = 0; r < R; ++r)
        for (int s = 0; s < K; ++s) {
            const int e = ids[(size_t) r * K + s];
            std::vector<double> h(FF);
            for (int f = 0; f < FF; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < H; ++c) {
                    g += (double) wg[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                    u += (double) wu[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                }
                h[f] = 4.0 * tanhd(g / 4.0) * sig(g) * 25.0 * tanhd(u / 25.0);
            }
            for (int c = 0; c < H; ++c) {
                double o = 0;
                for (int f = 0; f < FF; ++f) o += (double) wd[e][(size_t) c * FF + f] * h[f];
                ref[(size_t) r * H + c] += (double) w[(size_t) r * K + s] * o;
            }
        }
    report("experts (NVFP4, softcapped SwiGLU)", got, ref, 2e-2);   // q8_1 activations
    // the plan of row 1 alone gives that row's bits
    moe_plan(dids + K, 1, K, NE, dtable, ptr, start, ng, dst, tok, st);
    bk::native_quantize_q8_1(dx + H, xq, H, 1, st);
    bk::native_expert_grouped(L, ptr, start, ng, dst, tok, K, K, xq, scr, rows, st);
    float* y1 = zeros<float>(H);
    moe_combine(rows, dw + K, 1, K, y1, st);
    cudaStreamSynchronize(st);
    const std::vector<float> g1 = host(y1, H);
    same_bits("  a row alone", g1.data(), &got[H], (size_t) H * 4);
    // an expert off the card: its pairs are left out of the plan, the others planned as before
    std::vector<unsigned long long> off = table;
    off[2] = 0;
    unsigned long long* doff = dev(off);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st);
    cudaStreamSynchronize(st);
    const int n_g = host(ng, 1)[0];
    const std::vector<int32_t> hs = host(start, (size_t) R * K + 2), hd = host(dst, (size_t) R * K);
    const std::vector<unsigned long long> hp = host(ptr, (size_t) R * K);
    int want = 0;
    for (int32_t e : ids) want += e != 2;
    bool plan_ok = n_g == NE - 1 && hs[(size_t) n_g] == want;
    for (int i = 0; plan_ok && i < want; ++i) plan_ok = ids[(size_t) hd[(size_t) i]] != 2;
    for (int gi = 0; plan_ok && gi < n_g; ++gi) plan_ok = hp[(size_t) gi] != 0 && hp[(size_t) gi] != table[2];
    std::printf("%-34s %s\n", "  an expert off the card", plan_ok ? "left out" : "FAIL");
    failures += !plan_ok;
    // the console's tally: two plans add each expert's pairs on the card to its entry, the expert off it stays at zero
    uint32_t* tally = zeros<uint32_t>(NE);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st, tally);
    moe_plan(dids, R, K, NE, doff, ptr, start, ng, dst, tok, st, tally);
    cudaStreamSynchronize(st);
    const std::vector<uint32_t> ht = host(tally, NE);
    bool tally_ok = true;
    for (int e = 0; e < NE; ++e) {
        uint32_t n = 0;
        for (int32_t i : ids) n += i == e && e != 2 ? 2u : 0u;
        tally_ok = tally_ok && ht[(size_t) e] == n;
    }
    std::printf("%-34s %s\n", "  the experts' tally", tally_ok ? "counted" : "FAIL");
    failures += !tally_ok;
}

// Two cards' halves of the experts (tensor parallelism): expert blobs of `type` at n_ff 1152 cut into two blobs at n_ff 576,
// half c the gate and up rows [576c, 576c + 576) and the down matrix's columns [576c, 576c + 576) (NVFP4's per-tensor
// scale tail copied). Each half's weighted sum, added in card order, against the plain reference and against the whole
// experts' sum.
void test_expert_halves(int type) {
    const int H = E, FF = 1152, HF = FF / 2, NE = 6, R = 3, K = 4;
    const auto* tr = ggml_get_type_traits((ggml_type) type);
    bk::NativeExpertLayout L = bk::native_expert_layout(type, type, H, FF), Lh = bk::native_expert_layout(type, type, H, HF);
    for (bk::NativeExpertLayout* l : {&L, &Lh}) {
        l->gate_softcap = 4.f;
        l->up_softcap = 25.f;
    }
    const size_t tail = type == 40 ? 16 : 0;
    std::vector<uint8_t> blobs((size_t) NE * L.bytes), halves[2];
    std::vector<std::vector<float>> wg(NE), wu(NE), wd(NE);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * L.bytes;
        auto put = [&](std::vector<float>& deq, int rows, int k, size_t row_bytes, uint8_t* dst) {
            const std::vector<float> w = randn((size_t) rows * k, 0.05f);
            deq.resize((size_t) rows * k);
            for (int r = 0; r < rows; ++r) {
                tr->from_float_ref(&w[(size_t) r * k], dst + (size_t) r * row_bytes, k);
                tr->to_float(dst + (size_t) r * row_bytes, &deq[(size_t) r * k], k);
            }
        };
        put(wg[e], FF, H, L.gu_row, b);
        put(wu[e], FF, H, L.up_row, b + L.up_off);
        put(wd[e], H, FF, L.d_row, b + L.down_off);
        if (tail) {
            const float t[4] = {0.8f + 0.01f * e, 1.2f - 0.02f * e, 0.9f + 0.03f * e, 0.f};
            std::memcpy(b + L.bytes - 16, t, 16);
            for (int f = 0; f < FF * H; ++f) {
                wg[e][(size_t) f] *= t[0];
                wu[e][(size_t) f] *= t[1];
                wd[e][(size_t) f] *= t[2];
            }
        }
    }
    if (Lh.d_row * 2 != L.d_row) {
        std::printf("%-34s FAIL (a down row of %zu bytes does not halve)\n", "expert halves", (size_t) L.d_row);
        ++failures;
        return;
    }
    for (int c = 0; c < 2; ++c) {
        halves[c].assign((size_t) NE * Lh.bytes, 0);
        for (int e = 0; e < NE; ++e) {
            const uint8_t* b = blobs.data() + (size_t) e * L.bytes;
            uint8_t* h = halves[c].data() + (size_t) e * Lh.bytes;
            std::memcpy(h, b + (size_t) c * HF * L.gu_row, (size_t) HF * L.gu_row);
            std::memcpy(h + Lh.up_off, b + L.up_off + (size_t) c * HF * L.up_row, (size_t) HF * L.up_row);
            for (int r = 0; r < H; ++r)
                std::memcpy(h + Lh.down_off + (size_t) r * Lh.d_row, b + L.down_off + (size_t) r * L.d_row + (size_t) c * Lh.d_row, Lh.d_row);
            if (tail) std::memcpy(h + Lh.bytes - 16, b + L.bytes - 16, 16);
        }
    }
    const std::vector<int32_t> ids = {0, 2, 5, 1, 2, 3, 4, 0, 5, 1, 0, 3};   // [R][K]
    const std::vector<float> w = randn((size_t) R * K, 0.5f, 1.f), x = randn((size_t) R * H, 1.0f);
    int32_t *dids = dev(ids), *start = zeros<int32_t>(R * K + 2), *ng = zeros<int32_t>(4), *dst = zeros<int32_t>(R * K), *tok = zeros<int32_t>(R * K);
    unsigned long long* ptr = zeros<unsigned long long>(R * K);
    float *dw = dev(w), *dx = dev(x), *rows = zeros<float>((size_t) R * K * H), *y = zeros<float>((size_t) R * H);
    void* xq = zeros<uint8_t>(bk::native_q8_1_bytes(H, R));
    void* scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(R * K, FF));
    cudaStream_t st;
    cudaStreamCreate(&st);
    bk::native_quantize_q8_1(dx, xq, H, R, st);
    // the whole experts, then each half
    auto run = [&](const std::vector<uint8_t>& set, const bk::NativeExpertLayout& l) {
        uint8_t* d = dev(set);
        std::vector<unsigned long long> table(NE);
        for (int e = 0; e < NE; ++e) table[(size_t) e] = (unsigned long long) (d + (size_t) e * l.bytes);
        unsigned long long* dt = dev(table);
        moe_plan(dids, R, K, NE, dt, ptr, start, ng, dst, tok, st);
        bk::native_expert_grouped(l, ptr, start, ng, dst, tok, NE, R * K, xq, scr, rows, st);
        moe_combine(rows, dw, R, K, y, st);
        cudaStreamSynchronize(st);
        std::vector<float> out = host(y, (size_t) R * H);
        cudaFree(dt);
        cudaFree(d);
        return out;
    };
    const std::vector<float> whole = run(blobs, L), p0 = run(halves[0], Lh), p1 = run(halves[1], Lh);
    std::vector<float> got(whole.size());
    for (size_t i = 0; i < got.size(); ++i) got[i] = p0[i] + p1[i];
    std::vector<double> ref(got.size(), 0.0), ref_whole(whole.begin(), whole.end());
    for (int r = 0; r < R; ++r)
        for (int s = 0; s < K; ++s) {
            const int e = ids[(size_t) r * K + s];
            std::vector<double> h(FF);
            for (int f = 0; f < FF; ++f) {
                double g = 0, u = 0;
                for (int c = 0; c < H; ++c) {
                    g += (double) wg[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                    u += (double) wu[e][(size_t) f * H + c] * x[(size_t) r * H + c];
                }
                h[f] = 4.0 * tanhd(g / 4.0) * sig(g) * 25.0 * tanhd(u / 25.0);
            }
            for (int c = 0; c < H; ++c) {
                double o = 0;
                for (int f = 0; f < FF; ++f) o += (double) wd[e][(size_t) c * FF + f] * h[f];
                ref[(size_t) r * H + c] += (double) w[(size_t) r * K + s] * o;
            }
        }
    char what[64];
    std::snprintf(what, sizeof what, "expert halves (%s), reference", type == 40 ? "NVFP4" : "Q8_0");
    report(what, got, ref, 2e-2);   // q8_1 activations
    // the halves' sum is the whole experts' up to the order of the down product's sums (the same q8_1 rows of h)
    std::snprintf(what, sizeof what, "  against the whole experts");
    report(what, got, ref_whole, 1e-5);
    // timed as a decode window runs them: 8 experts a row out of 64, one row and eight, in a graph, whole and half
    {
        const int TE = 64, TK = 8, TR = 8;
        std::vector<int32_t> tids((size_t) TR * TK);
        for (int r = 0; r < TR; ++r) {
            std::vector<int32_t> pick(TE);
            for (int e = 0; e < TE; ++e) pick[(size_t) e] = e;
            std::shuffle(pick.begin(), pick.end(), rng);
            for (int s = 0; s < TK; ++s) tids[(size_t) r * TK + s] = pick[(size_t) s];
        }
        int32_t *d_ids = dev(tids), *t_start = zeros<int32_t>(TR * TK + 2), *t_ng = zeros<int32_t>(4), *t_dst = zeros<int32_t>(TR * TK),
                *t_tok = zeros<int32_t>(TR * TK);
        unsigned long long* t_ptr = zeros<unsigned long long>(TR * TK);
        const std::vector<float> tx = randn((size_t) TR * H, 1.0f);
        float *d_tx = dev(tx), *t_rows = zeros<float>((size_t) TR * TK * H);
        void* t_xq = zeros<uint8_t>(bk::native_q8_1_bytes(H, TR));
        void* t_scr = zeros<uint8_t>(bk::native_expert_scratch_bytes(TR * TK, FF));
        bk::native_quantize_q8_1(d_tx, t_xq, H, TR, st);
        double us[2][2] = {};
        for (int c = 0; c < 2; ++c) {
            const bk::NativeExpertLayout& l = c ? Lh : L;
            const std::vector<uint8_t>& src = c ? halves[0] : blobs;
            uint8_t* d = zeros<uint8_t>((size_t) TE * l.bytes);
            for (int e = 0; e < TE; ++e)   // the six experts' bytes over the 64
                cudaMemcpy(d + (size_t) e * l.bytes, src.data() + (size_t) (e % NE) * l.bytes, l.bytes, cudaMemcpyHostToDevice);
            std::vector<unsigned long long> table(TE);
            for (int e = 0; e < TE; ++e) table[(size_t) e] = (unsigned long long) (d + (size_t) e * l.bytes);
            unsigned long long* dt = dev(table);
            for (int ri = 0; ri < 2; ++ri) {
                const int rws = ri ? TR : 1, pairs = rws * TK;
                moe_plan(d_ids, rws, TK, TE, dt, t_ptr, t_start, t_ng, t_dst, t_tok, st);
                constexpr int kCalls = 16, kReps = 20;
                cudaGraph_t gr;
                cudaGraphExec_t ex;
                cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed);
                for (int i = 0; i < kCalls; ++i)
                    bk::native_expert_grouped(l, t_ptr, t_start, t_ng, t_dst, t_tok, std::min(pairs, TE), pairs, t_xq, t_scr, t_rows, st);
                cudaStreamEndCapture(st, &gr);
                cudaGraphInstantiate(&ex, gr, 0);
                cudaGraphLaunch(ex, st);
                cudaStreamSynchronize(st);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, st);
                for (int i = 0; i < kReps; ++i) cudaGraphLaunch(ex, st);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                us[c][ri] = 1000.0 * ms / (kCalls * kReps);
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
                cudaGraphExecDestroy(ex);
                cudaGraphDestroy(gr);
            }
            cudaFree(dt);
            cudaFree(d);
        }
        for (int ri = 0; ri < 2; ++ri)
            std::printf("  %d row%s of %d experts, a graph: whole %.2f us, half %.2f us (%.0f%%)\n", ri ? TR : 1, ri ? "s" : "", TK,
                        us[0][ri], us[1][ri], 100.0 * us[1][ri] / us[0][ri]);
        for (void* p : {(void*) d_ids, (void*) t_start, (void*) t_ng, (void*) t_dst, (void*) t_tok, (void*) t_ptr, (void*) d_tx,
                        (void*) t_rows, t_xq, t_scr})
            cudaFree(p);
    }
    cudaStreamDestroy(st);
    for (void* p : {(void*) dids, (void*) start, (void*) ng, (void*) dst, (void*) tok, (void*) ptr, (void*) dw, (void*) dx,
                    (void*) rows, (void*) y, xq, scr})
        cudaFree(p);
}

void test_softcap() {
    std::vector<float> x = randn(3 * 1000, 40.f);
    float* dx = dev(x);
    softcap(dx, 3, 1000, 1000, 30.f, nullptr);
    const std::vector<float> got = host(dx, x.size());
    std::vector<double> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = 30.0 * tanhd(x[i] / 30.0);
    report("softcap", got, ref, 2e-6);
}

// ---- a decode window's fused glue ---------------------------------------------------------------------------------------

void test_mix_norm() {
    const int R = 5;
    for (int n_out : {1, 3}) {
        const int n = 4;
        std::vector<std::vector<float>> st(n);
        std::vector<const std::vector<float>*> sp;
        MixStates ms;
        for (int i = 0; i < n; ++i) {
            st[i] = randn((size_t) R * E, 0.5f + 0.4f * i);
            ms.s[ms.n++] = dev(st[i]);
            sp.push_back(&st[i]);
        }
        const std::vector<float> q = randn((size_t) n_out * E, 0.05f);
        std::vector<std::vector<float>> w(3);
        MixNorm a;
        a.st = ms;
        a.queries = dev(q);
        for (int j = 0; j < n_out; ++j) {
            w[j] = randn(E, 0.1f, 1.f);
            a.norm[j] = dev(w[j]);
            a.q8[j] = zeros<uint8_t>((size_t) R * E / 32 * 36);
        }
        a.norm_out = zeros<float>((size_t) R * E);
        mix_norm(a, n_out, R, nullptr);
        const std::vector<float> got = host(a.norm_out, (size_t) R * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R; ++r) {
            const std::vector<double> y = norm_ref(mix_ref(sp, q.data(), r), w[0]);
            std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
        }
        char name[64];
        std::snprintf(name, sizeof name, "mix_norm (%d outputs): output 0", n_out);
        report(name, got, ref, 4e-6);
        check_q8("  its q8_1 blocks", a.q8[0], got, R, E);
        if (n_out == 3) {   // output 2 against the reference, through its q8_1 blocks
            const std::vector<uint8_t> q2 = host(static_cast<uint8_t*>(a.q8[2]), (size_t) R * E / 32 * 36);
            std::vector<float> deq((size_t) R * E);
            std::vector<double> r2(deq.size());
            for (int r = 0; r < R; ++r) {
                const std::vector<double> y = norm_ref(mix_ref(sp, q.data() + 2 * E, r), w[2]);
                std::copy(y.begin(), y.end(), r2.begin() + (size_t) r * E);
            }
            for (size_t b = 0; b < deq.size() / 32; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &q2[b * 36], 2);
                for (int i = 0; i < 32; ++i) deq[b * 32 + i] = (float) (h2d(dh) * (int8_t) q2[b * 36 + 4 + i]);
            }
            report("  output 2 through q8_1", deq, r2, 1.5e-2);
        }
        // the mixes as they are: the same bits as attnres_mix
        MixNorm m2 = a;
        m2.mix_out = zeros<float>((size_t) n_out * R * E);
        m2.out_stride = (int64_t) R * E;
        mix_norm(m2, n_out, R, nullptr);
        float* old = zeros<float>((size_t) n_out * R * E);
        attnres_mix(ms, a.queries, n_out, R, 1e-6f, old, (int64_t) R * E, nullptr);
        {
            const std::vector<float> mg = host(m2.mix_out, (size_t) n_out * R * E), mo = host(old, (size_t) n_out * R * E);
            report("  the mixes against attnres_mix", mg, std::vector<double>(mo.begin(), mo.end()), 1e-6);
        }
        // row 3 alone
        MixNorm one = a;
        for (int i = 0; i < n; ++i) one.st.s[i] = ms.s[i] + 3 * E;
        float* nrm1 = zeros<float>(E);
        void* q81 = zeros<uint8_t>(E / 32 * 36);
        one.norm_out = nrm1;
        one.q8[0] = q81;
        mix_norm(one, n_out, 1, nullptr);
        same_bits("  a row alone", host(nrm1, E).data(), &got[(size_t) 3 * E], (size_t) E * 4);
        same_bits("  a row alone (q8_1)", host(static_cast<uint8_t*>(q81), E / 32 * 36).data(),
                  host(static_cast<uint8_t*>(a.q8[0]) + (size_t) 3 * E / 32 * 36, E / 32 * 36).data(), E / 32 * 36);
        std::printf("  %.1f us a call (1 row, %d states, %d outputs)\n", time_us([&] { mix_norm(one, n_out, 1, nullptr); }), n, n_out);
    }
    {   // no query: the last state normed
        const int R2 = 2;
        const std::vector<float> x = randn((size_t) R2 * E, 1.3f), w = randn(E, 0.1f, 1.f);
        MixNorm a;
        a.st.s[a.st.n++] = dev(x);
        a.norm[0] = dev(w);
        a.q8[0] = zeros<uint8_t>((size_t) R2 * E / 32 * 36);
        a.norm_out = zeros<float>((size_t) R2 * E);
        mix_norm(a, 1, R2, nullptr);
        const std::vector<float> got = host(a.norm_out, (size_t) R2 * E);
        std::vector<double> ref(got.size());
        for (int r = 0; r < R2; ++r) {
            const std::vector<double> y = norm_ref(std::vector<double>(x.begin() + (size_t) r * E, x.begin() + (size_t) (r + 1) * E), w);
            std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
        }
        report("mix_norm (no query)", got, ref, 2e-6);
    }
}

void test_controllers() {
    const int R = 8, S = 3, rank = kCtrlRank;
    std::vector<std::vector<float>> dest(S), msg(S), nw(S);
    std::vector<std::vector<uint16_t>> down(S), up(S);
    CtrlArgs a;
    a.scratch = zeros<float>(ctrl_scratch_floats());
    auto bf16 = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return (uint16_t) ((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
    };
    auto bf2d = [](uint16_t h) {
        const uint32_t u = (uint32_t) h << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return (double) f;
    };
    for (int s = 0; s < S; ++s) {
        dest[s] = randn((size_t) R * E, 0.8f + 0.2f * s);
        msg[s] = randn((size_t) R * E, 1.2f);
        nw[s] = randn(E, 0.1f, 1.f);
        const std::vector<float> d = randn((size_t) rank * 2 * E, 0.02f), u = randn((size_t) 2 * E * rank, 0.3f);
        for (float f : d) down[s].push_back(bf16(f));
        for (float f : u) up[s].push_back(bf16(f));
        a.dest[s] = dev(dest[s]);
        a.msg[s] = dev(msg[s]);
        a.down[s] = dev(down[s]);
        a.up[s] = dev(up[s]);
        a.norm[s] = dev(nw[s]);
        a.q8[s] = zeros<uint8_t>((size_t) R * E / 32 * 36);
    }
    a.norm_out = zeros<float>((size_t) R * E);
    controllers(a, S, R, nullptr);
    const std::vector<float> got = host(a.norm_out, (size_t) R * E);
    auto ref_of = [&](int s, int r) {
        double sa = 0, sb = 0;
        for (int c = 0; c < E; ++c) {
            sa += (double) dest[s][(size_t) r * E + c] * dest[s][(size_t) r * E + c];
            sb += (double) msg[s][(size_t) r * E + c] * msg[s][(size_t) r * E + c];
        }
        const double ia = 1 / std::sqrt(sa / E + 1e-6), ib = 1 / std::sqrt(sb / E + 1e-6);
        std::vector<double> hid(rank, 0.0), y(E);
        for (int o = 0; o < rank; ++o)
            for (int c = 0; c < E; ++c)
                hid[o] += bf2d(down[s][(size_t) o * 2 * E + c]) * dest[s][(size_t) r * E + c] * ia +
                          bf2d(down[s][(size_t) o * 2 * E + E + c]) * msg[s][(size_t) r * E + c] * ib;
        for (int c = 0; c < E; ++c) {
            double g = 0, f = 0;
            for (int k = 0; k < rank; ++k) {
                g += bf2d(up[s][(size_t) c * rank + k]) * hid[k];
                f += bf2d(up[s][(size_t) (E + c) * rank + k]) * hid[k];
            }
            y[c] = (1 - std::tanh(f)) * dest[s][(size_t) r * E + c] + std::tanh(g) * msg[s][(size_t) r * E + c];
        }
        return norm_ref(y, nw[s]);
    };
    std::vector<double> ref(got.size());
    for (int r = 0; r < R; ++r) {
        const std::vector<double> y = ref_of(0, r);
        std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
    }
    report("controllers: stream 0", got, ref, 2e-5);
    check_q8("  its q8_1 blocks", a.q8[0], got, R, E);
    {   // stream 2 through its q8_1 blocks
        const std::vector<uint8_t> q2 = host(static_cast<uint8_t*>(a.q8[2]), (size_t) R * E / 32 * 36);
        std::vector<float> deq((size_t) R * E);
        std::vector<double> r2(deq.size());
        for (int r = 0; r < R; ++r) {
            const std::vector<double> y = ref_of(2, r);
            std::copy(y.begin(), y.end(), r2.begin() + (size_t) r * E);
        }
        for (size_t b = 0; b < deq.size() / 32; ++b) {
            uint16_t dh;
            std::memcpy(&dh, &q2[b * 36], 2);
            for (int i = 0; i < 32; ++i) deq[b * 32 + i] = (float) (h2d(dh) * (int8_t) q2[b * 36 + 4 + i]);
        }
        report("  stream 2 through q8_1", deq, r2, 1.5e-2);
    }
    CtrlArgs one = a;   // row 1 alone
    for (int s = 0; s < S; ++s) {
        one.dest[s] = a.dest[s] + E;
        one.msg[s] = a.msg[s] + E;
        one.q8[s] = zeros<uint8_t>(E / 32 * 36);
    }
    one.norm_out = zeros<float>(E);
    controllers(one, S, 1, nullptr);
    same_bits("  a row alone", host(one.norm_out, E).data(), &got[E], (size_t) E * 4);
    {   // every other count of rows and streams (each count's kernel) against the 8 rows of 3 streams: stream 0's rows
        std::vector<float> each, want;
        for (int st : {S, 1})
            for (int r = 1; r <= R; ++r) {
                if (st == S && r == R) continue;
                CtrlArgs b = a;
                for (int s = 0; s < S; ++s) b.q8[s] = zeros<uint8_t>((size_t) r * E / 32 * 36);
                b.norm_out = zeros<float>((size_t) r * E);
                controllers(b, st, r, nullptr);
                const std::vector<float> o = host(b.norm_out, (size_t) r * E);
                each.insert(each.end(), o.begin(), o.end());
                want.insert(want.end(), got.begin(), got.begin() + (ptrdiff_t) r * E);
            }
        same_bits("  1-8 rows of 1 or 3 streams", each.data(), want.data(), each.size() * 4);
    }
    std::printf("  %.1f us a call (1 row, 3 streams)\n", time_us([&] { controllers(one, S, 1, nullptr); }));
    std::printf("  %.1f us a call (%d rows, 3 streams)\n", time_us([&] { controllers(a, S, R, nullptr); }), R);
}

void test_decode_prepare() {
    for (bool full : {false, true}) {
        const int R = 4, p0 = full ? 9000 : 1;
        const int64_t cap = full ? 9100 : kExtent + 8;
        std::vector<int> pos(R);
        for (int r = 0; r < R; ++r) pos[r] = p0 + r;
        // the projections as one [rows][kProj] array (the stride of a single product over all rows)
        const std::vector<float> y = randn((size_t) R * kProj, 1.5f);
        const std::vector<float> kc = randn(1024 * kConv, 0.3f), vc = randn(1024 * kConv, 0.3f);
        const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rs = randn(kHeads * kRel, 0.2f, 1.f);
        const std::vector<float> kr = randn((size_t) kConvRing * 1024, 2.0f), vr = randn((size_t) kConvRing * 1024, 0.6f);
        float* dy = dev(y);
        int* dpos = dev(pos);
        AttnPrep a;
        a.yq = dy; a.yk = dy + kProjQ; a.yv = dy + kProjQ + kProjKV; a.ldq = a.ldk = a.ldv = kProj;
        a.pos = dpos; a.full = full; a.k_conv = dev(kc); a.v_conv = dev(vc); a.q_scale = dev(qs); a.r_scale = dev(rs);
        a.cap = cap;
        AttnPrep b = a;
        for (AttnPrep* x : {&a, &b}) {
            x->k_ring = dev(kr); x->v_ring = dev(vr);
            x->k_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim); x->v_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim);
            x->q = zeros<float>((size_t) R * kHeads * kHeadDim); x->r = zeros<float>((size_t) R * kHeads * kRel); x->sink = zeros<float>((size_t) R * kHeads);
        }
        decode_prepare(a, R, nullptr);
        attn_prepare(b, R, nullptr);
        cudaDeviceSynchronize();
        const size_t kvb = (size_t) kKvHeads * cap * kHeadDim * 2;
        const char* tag = full ? "decode_prepare (full)" : "decode_prepare (window)";
        bool same = !std::memcmp(host((uint8_t*) a.k_cache, kvb).data(), host((uint8_t*) b.k_cache, kvb).data(), kvb) &&
                    !std::memcmp(host((uint8_t*) a.v_cache, kvb).data(), host((uint8_t*) b.v_cache, kvb).data(), kvb) &&
                    host(a.q, (size_t) R * kHeads * kHeadDim) == host(b.q, (size_t) R * kHeads * kHeadDim) &&
                    host(a.r, (size_t) R * kHeads * kRel) == host(b.r, (size_t) R * kHeads * kRel) &&
                    host(a.sink, (size_t) R * kHeads) == host(b.sink, (size_t) R * kHeads) &&
                    host(a.k_ring, (size_t) kConvRing * 1024) == host(b.k_ring, (size_t) kConvRing * 1024) &&
                    host(a.v_ring, (size_t) kConvRing * 1024) == host(b.v_ring, (size_t) kConvRing * 1024);
        std::printf("%-34s %s\n", tag, same ? "bitwise = attn_prepare" : "FAIL");
        failures += !same;
        // tensor parallelism: each half of the groups prepares every key and value and only its own query heads
        for (int half = 0; half < 2; ++half) {
            AttnPrep t = a;
            t.g0 = half * kKvHeads / 2;
            t.ng = kKvHeads / 2;
            t.k_ring = dev(kr); t.v_ring = dev(vr);
            t.k_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim); t.v_cache = zeros<uint16_t>((size_t) kKvHeads * cap * kHeadDim);
            t.q = zeros<float>((size_t) R * kHeads * kHeadDim); t.r = zeros<float>((size_t) R * kHeads * kRel); t.sink = zeros<float>((size_t) R * kHeads);
            decode_prepare(t, R, nullptr);
            cudaDeviceSynchronize();
            bool ok = !std::memcmp(host((uint8_t*) t.k_cache, kvb).data(), host((uint8_t*) a.k_cache, kvb).data(), kvb) &&
                      !std::memcmp(host((uint8_t*) t.v_cache, kvb).data(), host((uint8_t*) a.v_cache, kvb).data(), kvb) &&
                      host(t.k_ring, (size_t) kConvRing * 1024) == host(a.k_ring, (size_t) kConvRing * 1024) &&
                      host(t.v_ring, (size_t) kConvRing * 1024) == host(a.v_ring, (size_t) kConvRing * 1024);
            const int h0 = t.g0 * (kHeads / kKvHeads), nh = t.ng * (kHeads / kKvHeads);
            const std::vector<float> tq = host(t.q, (size_t) R * kHeads * kHeadDim), aq = host(a.q, tq.size());
            const std::vector<float> tr = host(t.r, (size_t) R * kHeads * kRel), ar = host(a.r, tr.size());
            const std::vector<float> ts = host(t.sink, (size_t) R * kHeads), as = host(a.sink, ts.size());
            for (int r = 0; r < R && ok; ++r)
                ok = !std::memcmp(&tq[((size_t) r * kHeads + h0) * kHeadDim], &aq[((size_t) r * kHeads + h0) * kHeadDim], (size_t) nh * kHeadDim * 4) &&
                     !std::memcmp(&tr[((size_t) r * kHeads + h0) * kRel], &ar[((size_t) r * kHeads + h0) * kRel], (size_t) nh * kRel * 4) &&
                     !std::memcmp(&ts[(size_t) r * kHeads + h0], &as[(size_t) r * kHeads + h0], (size_t) nh * 4);
            std::printf("  %-32s %s\n", half ? "groups 4..7 alone" : "groups 0..3 alone", ok ? "bitwise" : "FAIL");
            failures += !ok;
        }
        if (!full) std::printf("  %.1f us a call (1 row)\n", time_us([&] { decode_prepare(a, 1, nullptr); }));
    }
}

void test_post_attn() {
    for (bool boundary : {false, true}) {
        for (bool mix_only : {false, true}) {
            const int R = 4, nb = 2;
            std::vector<int> pos = {70, 71, 72, 73};
            const std::vector<float> o = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f),
                                     ring = randn((size_t) kConvRing * E, 0.4f), hid = randn((size_t) R * E, 1.0f), q = randn(E, 0.05f),
                                     w = randn(E, 0.1f, 1.f);
            std::vector<std::vector<float>> blocks(nb);
            PostAttn a;
            for (int i = 0; i < nb; ++i) {
                blocks[i] = randn((size_t) R * E, 0.7f + 0.5f * i);
                a.blocks[a.n_blocks++] = dev(blocks[i]);
            }
            a.o = dev(o); a.pos = dev(pos); a.taps = dev(taps); a.ring = dev(ring); a.hid = dev(hid); a.boundary = boundary;
            a.resid = zeros<float>((size_t) R * E); a.query = dev(q); a.norm = dev(w);
            a.q8 = zeros<uint8_t>((size_t) R * E / 32 * 36); a.norm_out = zeros<float>((size_t) R * E);
            if (mix_only) a.mix_out = zeros<float>((size_t) R * E);
            post_attn(a, R, nullptr);
            const std::vector<float> res = host(a.resid, (size_t) R * E);
            std::vector<double> rres(res.size());
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < E; ++c) {
                    double acc = 0;
                    for (int j = 0; j < kConv; ++j) acc += (double) taps[(size_t) c * kConv + (kConv - 1 - j)] * conv_input(o, E, ring, E, pos, r, j, c);
                    rres[(size_t) r * E + c] = (acc + o[(size_t) r * E + c]) / std::sqrt(5.0) + (boundary ? 0.0 : hid[(size_t) r * E + c]);
                }
            char name[80];
            std::snprintf(name, sizeof name, "post_attn%s%s: residual", boundary ? " (boundary)" : "", mix_only ? " (mix)" : "");
            report(name, res, rres, 2e-6);
            std::vector<const std::vector<float>*> sp;
            for (auto& b : blocks) sp.push_back(&b);
            if (boundary) sp.push_back(&hid);
            sp.push_back(&res);
            std::vector<double> ref((size_t) R * E);
            for (int r = 0; r < R; ++r) {
                std::vector<double> y = mix_ref(sp, q.data(), r);
                if (!mix_only) y = norm_ref(y, w);
                std::copy(y.begin(), y.end(), ref.begin() + (size_t) r * E);
            }
            const std::vector<float> got = host(mix_only ? a.mix_out : a.norm_out, (size_t) R * E);
            report(mix_only ? "  the MLP's mix" : "  the MLP's normed input", got, ref, 4e-6);
            if (!mix_only) check_q8("  its q8_1 blocks", a.q8, got, R, E);
            const std::vector<float> nr = host(a.ring, (size_t) kConvRing * E);
            bool ring_ok = true;
            for (int r = 0; r < R; ++r) ring_ok = ring_ok && !std::memcmp(&nr[(size_t) (pos[r] % kConvRing) * E], &o[(size_t) r * E], E * 4);
            std::printf("%-34s %s\n", "  the ring", ring_ok ? "ok" : "FAIL");
            failures += !ring_ok;
            {   // tensor parallelism: the projection as the two cards' sums, after an exchange with itself
                const std::vector<float> p0 = randn((size_t) R * E, 0.3f), p1 = randn((size_t) R * E, 0.3f);
                std::vector<float> o2(p0.size());
                for (size_t i = 0; i < o2.size(); ++i) o2[i] = p0[i] + p1[i];
                PostAttn whole = a, parts = a;
                whole.o = dev(o2);
                parts.o = nullptr;
                parts.part[0] = dev(p0);
                parts.part[1] = dev(p1);
                const std::vector<uint32_t> words0 = {1, 0, 0};
                uint32_t* words = dev(words0);
                parts.xch.sent = words;
                parts.xch.box = parts.xch.peer_box = words + 1;
                parts.xch.abort = words + 2;
                for (PostAttn* p : {&whole, &parts}) {
                    p->ring = dev(ring);
                    p->resid = zeros<float>((size_t) R * E);
                    p->norm_out = zeros<float>((size_t) R * E);
                    p->q8 = zeros<uint8_t>((size_t) R * E / 32 * 36);
                    if (mix_only) p->mix_out = zeros<float>((size_t) R * E);
                    post_attn(*p, R, nullptr);
                }
                float* const out_w = mix_only ? whole.mix_out : whole.norm_out;
                float* const out_p = mix_only ? parts.mix_out : parts.norm_out;
                same_bits("  two cards' sums: residual", host(parts.resid, (size_t) R * E).data(),
                          host(whole.resid, (size_t) R * E).data(), (size_t) R * E * 4);
                same_bits("  two cards' sums: output", host(out_p, (size_t) R * E).data(), host(out_w, (size_t) R * E).data(),
                          (size_t) R * E * 4);
                const std::vector<uint32_t> wd = host(words, 3);
                const bool waited = wd[1] == 1 && wd[2] == 0;
                std::printf("%-34s %s\n", "  two cards' sums: the exchange", waited ? "ok" : "FAIL");
                failures += !waited;
            }
            if (!boundary && !mix_only) {   // row 2 alone (its window rows before it read from a ring holding them)
                std::vector<float> ring2 = ring;
                for (int r = 0; r < 2; ++r) std::memcpy(&ring2[(size_t) (pos[r] % kConvRing) * E], &o[(size_t) r * E], E * 4);
                PostAttn one = a;
                one.o = a.o + 2 * E; one.pos = a.pos + 2; one.ring = dev(ring2); one.hid = a.hid + 2 * E;
                for (int i = 0; i < nb; ++i) one.blocks[i] = a.blocks[i] + 2 * E;
                one.resid = zeros<float>(E); one.norm_out = zeros<float>(E); one.q8 = zeros<uint8_t>(E / 32 * 36);
                post_attn(one, 1, nullptr);
                same_bits("  a row alone", host(one.norm_out, E).data(), &got[(size_t) 2 * E], (size_t) E * 4);
                std::printf("  %.1f us a call (1 row, %d blocks)\n", time_us([&] { post_attn(one, 1, nullptr); }), nb);
            }
        }
    }
}

// The guessed experts' prefetch over a stand-in blob table runs and reads only.
void test_expert_prefetch() {
    const int NE = 128;
    const size_t blob = 65536;
    uint8_t* blobs = zeros<uint8_t>((size_t) NE * blob);
    std::vector<unsigned long long> tab(NE);
    for (int e = 0; e < NE; ++e) tab[e] = (unsigned long long) (blobs + (size_t) e * blob);
    const std::vector<int32_t> ids = {3, 17, 64, 127, 0, 90, 41, 8};
    unsigned long long* dtab = dev(tab);
    int32_t* dids = dev(ids);
    uint32_t* sink = zeros<uint32_t>(1);
    expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr);
    const bool ok = cudaDeviceSynchronize() == cudaSuccess;
    std::printf("%-34s %s\n", "expert_prefetch: runs", ok ? "ok" : "FAIL");
    failures += !ok;
    std::printf("  %.1f us a call (8 blobs of 64 KiB)\n", time_us([&] { expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr); }));
    // experts in RAM (0 in the table, the CPU's) are skipped: some of them, then all
    for (int all : {0, 1}) {
        for (int e = 0; e < NE; ++e) tab[e] = all || e % 2 ? 0ull : (unsigned long long) (blobs + (size_t) e * blob);
        cudaMemcpy(dtab, tab.data(), NE * sizeof(tab[0]), cudaMemcpyHostToDevice);
        expert_prefetch(dids, (int) ids.size(), dtab, blob, sink, nullptr);
        const bool skip_ok = cudaDeviceSynchronize() == cudaSuccess;
        std::printf("%-34s %s\n", all ? "expert_prefetch: every one in RAM" : "expert_prefetch: some in RAM", skip_ok ? "ok" : "FAIL");
        failures += !skip_ok;
    }
}

void test_moe_finish() {
    const int R = 4, K = 8;
    std::vector<int> pos = {130, 131, 132, 133};
    const std::vector<float> rows = randn((size_t) R * K * E, 0.3f), w = randn((size_t) R * K, 0.3f, 1.f), taps = randn((size_t) E * kConv, 0.3f),
                             ring = randn((size_t) kConvRing * E, 0.4f), resid = randn((size_t) R * E, 1.0f);
    MoeFinish a;
    a.rows = dev(rows); a.w = dev(w); a.k = K; a.pos = dev(pos); a.taps = dev(taps); a.ring = dev(ring); a.resid = dev(resid);
    a.out = zeros<float>((size_t) R * E);
    moe_finish(a, R, nullptr);
    // the kernels it replaces: moe_combine, then post_conv with the residual
    float *y = zeros<float>((size_t) R * E), *out2 = zeros<float>((size_t) R * E), *ring2 = dev(ring);
    moe_combine(a.rows, a.w, R, K, y, nullptr);
    post_conv(y, a.pos, a.taps, ring2, a.resid, R, out2, nullptr);
    const std::vector<float> got = host(a.out, (size_t) R * E);
    same_bits("moe_finish = moe_combine + post_conv", got.data(), host(out2, (size_t) R * E).data(), (size_t) R * E * 4);
    same_bits("  the ring", host(a.ring, (size_t) kConvRing * E).data(), host(ring2, (size_t) kConvRing * E).data(), (size_t) kConvRing * E * 4);
    MoeFinish t = a;   // with the drafter's tap: the same rows in both outputs
    t.ring = dev(ring);
    t.out = zeros<float>((size_t) R * E);
    t.tap = zeros<float>((size_t) R * E);
    moe_finish(t, R, nullptr);
    same_bits("  with a tap: the output", host(t.out, (size_t) R * E).data(), got.data(), (size_t) R * E * 4);
    same_bits("  with a tap: the tap", host(t.tap, (size_t) R * E).data(), got.data(), (size_t) R * E * 4);
    {   // two cards: each half experts' sum (moe_combine, the peer copy too), then the finish of their sum in card order
        const std::vector<float> rows1 = randn((size_t) R * K * E, 0.3f);
        float *p0 = zeros<float>((size_t) R * E), *p0_peer = zeros<float>((size_t) R * E), *p1 = zeros<float>((size_t) R * E);
        moe_combine(a.rows, a.w, R, K, p0, nullptr, p0_peer);
        moe_combine(dev(rows1), a.w, R, K, p1, nullptr);
        const std::vector<float> h0 = host(p0, (size_t) R * E), h1 = host(p1, (size_t) R * E);
        same_bits("  the partial's peer copy", host(p0_peer, (size_t) R * E).data(), h0.data(), (size_t) R * E * 4);
        MoeFinish tp = a;
        tp.rows = nullptr;
        tp.w = nullptr;
        tp.part[0] = p0_peer;
        tp.part[1] = p1;
        tp.ring = dev(ring);
        tp.out = zeros<float>((size_t) R * E);
        moe_finish(tp, R, nullptr);
        std::vector<float> sum((size_t) R * E);
        for (size_t i = 0; i < sum.size(); ++i) sum[i] = h0[i] + h1[i];
        float *ysum = dev(sum), *out3 = zeros<float>((size_t) R * E), *ring3 = dev(ring);
        post_conv(ysum, a.pos, a.taps, ring3, a.resid, R, out3, nullptr);
        same_bits("  two cards' parts = post_conv of their sum", host(tp.out, (size_t) R * E).data(), host(out3, (size_t) R * E).data(),
                  (size_t) R * E * 4);
        same_bits("  two cards' parts: the ring", host(tp.ring, (size_t) kConvRing * E).data(), host(ring3, (size_t) kConvRing * E).data(),
                  (size_t) kConvRing * E * 4);
        // the exchange's words (one card standing in for both): the sum counted as sent, its count already in the box ->
        // the same bits, the count told to the other card's box; a box that never reaches the count -> the abort word set
        // past the bound; once it is set, a wait gives up at once
        uint32_t* words = zeros<uint32_t>(3);   // the sums sent, the box, the other card's box
        void* word = nullptr;
        cudaHostAlloc(&word, 4, cudaHostAllocMapped | cudaHostAllocPortable);
        volatile uint32_t* const abort_word = static_cast<uint32_t*>(word);
        *abort_word = 0;
        const auto counts = [&] { return host(words, 3); };
        moe_combine(a.rows, a.w, R, K, p0, nullptr, p0_peer, words);
        const uint32_t one_sent = 1;
        cudaMemcpy(words + 1, &one_sent, 4, cudaMemcpyHostToDevice);
        MoeFinish xp = tp;
        xp.xch.sent = words;
        xp.xch.box = words + 1;
        xp.xch.peer_box = words + 2;
        xp.xch.abort = static_cast<uint32_t*>(word);
        xp.ring = dev(ring);
        xp.out = zeros<float>((size_t) R * E);
        moe_finish(xp, R, nullptr);
        same_bits("  the exchange's wait: the output", host(xp.out, (size_t) R * E).data(), host(tp.out, (size_t) R * E).data(),
                  (size_t) R * E * 4);
        std::vector<uint32_t> c = counts();
        const bool told = c[0] == 1 && c[2] == 1 && *abort_word == 0;
        std::printf("%-34s %s\n", "  the exchange's wait: the counts", told ? "ok" : "FAIL");
        failures += !told;
        const auto timed = [&] {
            moe_combine(a.rows, a.w, R, K, p0, nullptr, p0_peer, words);
            const auto t0 = std::chrono::steady_clock::now();
            moe_finish(xp, R, nullptr);
            cudaDeviceSynchronize();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        const double bound_s = timed();
        c = counts();
        const bool gave_up = cudaGetLastError() == cudaSuccess && *abort_word != 0 && c[0] == 2 && c[2] == 2 && bound_s > 1.0;
        std::printf("%-34s %s (%.2f s)\n", "  a wait past its bound", gave_up ? "ok" : "FAIL", bound_s);
        failures += !gave_up;
        const double after_s = timed();
        const bool at_once = *abort_word != 0 && counts()[2] == 3 && after_s < 0.1;
        std::printf("%-34s %s (%.4f s)\n", "  a wait once the word is set", at_once ? "ok" : "FAIL", after_s);
        failures += !at_once;
        cudaFreeHost(word);
    }
    MoeFinish one = a;
    one.rows = a.rows + (size_t) 3 * K * E; one.w = a.w + 3 * K; one.pos = a.pos + 3; one.resid = a.resid + 3 * E; one.out = zeros<float>(E);
    std::printf("  %.1f us a call (1 row)\n", time_us([&] { moe_finish(one, 1, nullptr); }));
}

// The CPU experts' rows into the experts' output (moe_cpu_merge): from mapped host memory a host function filled, in a
// captured graph replayed twice; the pairs whose expert is in RAM (0 in the table) take their rows and their lookups, the
// card's pairs keep theirs. Every expert on the card, none, and every other one, for 1 to 8 rows.
void test_cpu_merge() {
    struct Fill {
        float* data = nullptr;
        size_t n = 0;
        const float* src = nullptr;
        int replay = 0;
        static void CUDART_CB run(void* p) {
            const Fill& f = *static_cast<const Fill*>(p);
            for (size_t i = 0; i < f.n; ++i) f.data[i] = f.src[i] + (float) f.replay * 0.125f;
        }
    };
    const int K = 8, NE = 128;
    bool ok = true;
    for (int rows = 1; rows <= kDecodeRows; ++rows)
        for (int mode : {0, 1, 2}) {
            const int pairs = rows * K;
            std::vector<int32_t> ids((size_t) pairs);
            for (int p = 0; p < pairs; ++p) ids[(size_t) p] = (p % K * 7 + p / K) % NE;
            std::vector<unsigned long long> tab(NE);
            for (int e = 0; e < NE; ++e) tab[(size_t) e] = mode == 0 ? 1 : mode == 1 ? 0 : e % 2;
            const std::vector<float> before((size_t) pairs * E, -17.f), src = randn((size_t) pairs * E);
            int32_t* di = dev(ids);
            unsigned long long* dt = dev(tab);
            float* dy = dev(before);
            uint32_t* tally = zeros<uint32_t>(NE);
            Fill fill;
            fill.n = (size_t) pairs * E;
            fill.src = src.data();
            float* mapped = nullptr;
            cudaHostAlloc((void**) &fill.data, fill.n * 4, cudaHostAllocMapped);
            cudaHostGetDevicePointer((void**) &mapped, fill.data, 0);
            cudaStream_t st;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
            cudaLaunchHostFunc(st, Fill::run, &fill);
            moe_cpu_merge(di, rows, K, dt, mapped, dy, tally, st);
            cudaStreamEndCapture(st, &graph);
            cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
            for (int replay = 1; replay <= 2; ++replay) {
                fill.replay = replay;
                cudaGraphLaunch(exec, st);
                ok &= cudaStreamSynchronize(st) == cudaSuccess;
                const std::vector<float> y = host(dy, (size_t) pairs * E);
                const std::vector<uint32_t> counts = host(tally, NE);
                std::vector<uint32_t> want(NE, 0);
                for (int p = 0; p < pairs; ++p) {
                    const bool ram = !tab[(size_t) ids[(size_t) p]];
                    if (ram) want[(size_t) ids[(size_t) p]] += replay;
                    for (int c = 0; c < E; ++c) {
                        const size_t i = (size_t) p * E + c;
                        const float ref = ram ? src[i] + (float) replay * 0.125f : -17.f;
                        ok &= std::memcmp(&y[i], &ref, 4) == 0;
                    }
                }
                ok &= counts == want;
            }
            if (mode == 1 && rows == kDecodeRows)
                std::printf("  %.1f us a call (%d rows from mapped memory)\n",
                            time_us([&] { moe_cpu_merge(di, rows, K, dt, mapped, dy, nullptr, nullptr); }), rows);
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(st);
            cudaFreeHost(fill.data);
        }
    std::printf("%-34s %s\n", "moe_cpu_merge: graph, mapped rows", ok ? "ok" : "FAIL");
    failures += !ok;
}

// A batch window (several conversations' rows in one call, docs/design/batching.md) against each conversation's own call:
// the attention's q8_1 output, caches and rings, post_attn's and moe_finish's outputs and rings, bitwise. Three segments
// of 3, 1 and 2 rows (one at position 1, one past the window ring's wrap), over a window ring and a full cache, fp16 and E4M3.
void test_batch_window() {
    struct Seg { int p0, T; };
    const Seg segs[3] = {{300, 3}, {1, 1}, {2900, 2}};
    const int S = 3, R = 6, row0[S + 1] = {0, 3, 4, 6};
    Segments sg;
    sg.at.n = S;
    for (int i = 0; i <= S; ++i) sg.at.row0[i] = row0[i];
    std::vector<int> pos(R);
    for (int i = 0; i < S; ++i)
        for (int t = 0; t < segs[i].T; ++t) pos[row0[i] + t] = segs[i].p0 + t;
    int* dpos = dev(pos);
    const size_t q8_row = (size_t) kHeads * kHeadDim / 32 * 36;
    for (bool fp8 : {false, true})
        for (bool window : {true, false}) {
            const int64_t cap = window ? kExtent + 8 : 4096;
            const size_t n_kv = (size_t) kKvHeads * cap * kHeadDim, elem = fp8 ? 1 : 2, kvb = n_kv * elem, ringb = (size_t) kConvRing * 1024;
            std::vector<std::vector<uint8_t>> kc(S, std::vector<uint8_t>(kvb)), vc(S, std::vector<uint8_t>(kvb));
            std::vector<std::vector<float>> kr(S), vr(S);
            for (int i = 0; i < S; ++i) {
                for (size_t j = 0; j < n_kv; ++j) {
                    const float kf = randn(1, 0.6f)[0], vf = randn(1, 1.0f)[0];
                    if (fp8) {
                        kc[i][j] = __nv_fp8_e4m3(kf).__x;
                        vc[i][j] = __nv_fp8_e4m3(vf).__x;
                    } else {
                        const uint16_t a = f2h(kf), b = f2h(vf);
                        std::memcpy(&kc[i][2 * j], &a, 2);
                        std::memcpy(&vc[i][2 * j], &b, 2);
                    }
                }
                kr[i] = randn(ringb, 2.0f);
                vr[i] = randn(ringb, 0.6f);
            }
            const std::vector<float> y = randn((size_t) R * kProj, 1.5f);
            const std::vector<float> tk = randn(1024 * kConv, 0.3f), tv = randn(1024 * kConv, 0.3f);
            const std::vector<float> qs = randn(kHeads * kHeadDim, 0.2f, 1.f), rsc = randn(kHeads * kRel, 0.2f, 1.f);
            const std::vector<float> table = randn((size_t) kRel * kExtent, 0.3f);
            float* dy = dev(y);
            const auto prep_of = [&](int r0, int rows, const int* p) {
                AttnPrep a;
                a.yq = dy + (size_t) r0 * kProj; a.yk = a.yq + kProjQ; a.yv = a.yq + kProjQ + kProjKV; a.ldq = a.ldk = a.ldv = kProj;
                a.pos = p; a.full = !window; a.k_conv = dev(tk); a.v_conv = dev(tv); a.q_scale = dev(qs); a.r_scale = dev(rsc);
                a.cap = cap; a.fp8 = fp8;
                a.q = zeros<float>((size_t) rows * kHeads * kHeadDim); a.r = zeros<float>((size_t) rows * kHeads * kRel);
                a.sink = zeros<float>((size_t) rows * kHeads);
                return a;
            };
            const auto args_of = [&](const AttnPrep& p, int rows) {
                AttnArgs a;
                a.q = p.q; a.r = p.r; a.sink = p.sink; a.rel_table = dev(table); a.pos = p.pos; a.k_cache = p.k_cache; a.v_cache = p.v_cache;
                a.cap = cap; a.fp8 = fp8; a.window = window ? kExtent : 0; a.gate = p.yq + kOffGate; a.gate_ld = kProj;
                a.q8 = zeros<uint8_t>((size_t) rows * q8_row);
                return a;
            };
            const auto keys_of = [&](int64_t last) {
                int64_t mk = 1024;
                while (mk < last + 1) mk <<= 1;
                return window ? (int64_t) kExtent : mk;
            };
            // each conversation alone (its own call: the fused kernel for these rows), then all of them in one call
            std::vector<AttnPrep> ps(S);
            std::vector<AttnArgs> as(S);
            for (int i = 0; i < S; ++i) {
                ps[i] = prep_of(row0[i], segs[i].T, dpos + row0[i]);
                ps[i].k_ring = dev(kr[i]); ps[i].v_ring = dev(vr[i]); ps[i].k_cache = dev(kc[i]); ps[i].v_cache = dev(vc[i]);
                as[i] = args_of(ps[i], segs[i].T);
                const int64_t mk = keys_of(segs[i].p0 + segs[i].T - 1);
                attention_window(ps[i], as[i], segs[i].T, mk, zeros<uint8_t>(attention_decode_scratch(segs[i].T, mk)), nullptr);
            }
            AttnPrep pb = prep_of(0, R, dpos);
            pb.seg = sg;
            for (int i = 0; i < S; ++i) {
                SegCaches& c = pb.seg.c[i];
                c.k = dev(kc[i]); c.v = dev(vc[i]); c.k_ring = dev(kr[i]); c.v_ring = dev(vr[i]);
            }
            pb.k_cache = pb.seg.c[0].k; pb.v_cache = pb.seg.c[0].v; pb.k_ring = pb.seg.c[0].k_ring; pb.v_ring = pb.seg.c[0].v_ring;
            AttnArgs ab = args_of(pb, R);
            ab.seg = pb.seg;
            const int64_t mkb = keys_of(2901);
            attention_window(pb, ab, R, mkb, zeros<uint8_t>(attention_decode_scratch(R, mkb)), nullptr);
            cudaDeviceSynchronize();
            bool same = true;
            const std::vector<uint8_t> q8b = host((uint8_t*) ab.q8, (size_t) R * q8_row);
            for (int i = 0; i < S; ++i) {
                const std::vector<uint8_t> q8s = host((uint8_t*) as[i].q8, (size_t) segs[i].T * q8_row);
                same = same && !std::memcmp(q8s.data(), q8b.data() + (size_t) row0[i] * q8_row, q8s.size()) &&
                       host((uint8_t*) ps[i].k_cache, kvb) == host((uint8_t*) pb.seg.c[i].k, kvb) &&
                       host((uint8_t*) ps[i].v_cache, kvb) == host((uint8_t*) pb.seg.c[i].v, kvb) &&
                       host(ps[i].k_ring, ringb) == host(pb.seg.c[i].k_ring, ringb) && host(ps[i].v_ring, ringb) == host(pb.seg.c[i].v_ring, ringb);
            }
            char tag[96];
            std::snprintf(tag, sizeof tag, "batch window: attention (%s%s)", window ? "window" : "full", fp8 ? ", E4M3" : "");
            std::printf("%-46s %s\n", tag, same ? "bitwise = each conversation alone" : "FAIL");
            failures += !same;
        }
    // post_attn and moe_finish: each conversation's ring and its own rows before it
    const int nb = 2, K = 8;
    const std::vector<float> o = randn((size_t) R * E, 0.4f), taps = randn((size_t) E * kConv, 0.3f), hid = randn((size_t) R * E, 1.0f),
                             q = randn(E, 0.05f), w = randn(E, 0.1f, 1.f), xr = randn((size_t) R * K * E, 0.3f), xw = randn((size_t) R * K, 0.3f, 1.f),
                             mt = randn((size_t) E * kConv, 0.3f), mres = randn((size_t) R * E, 1.0f);
    std::vector<std::vector<float>> blocks(nb), ar(S), mr(S);
    for (int i = 0; i < nb; ++i) blocks[i] = randn((size_t) R * E, 0.7f + 0.5f * i);
    for (int i = 0; i < S; ++i) {
        ar[i] = randn((size_t) kConvRing * E, 0.4f);
        mr[i] = randn((size_t) kConvRing * E, 0.4f);
    }
    float *d_o = dev(o), *d_hid = dev(hid), *d_xr = dev(xr), *d_xw = dev(xw), *d_mres = dev(mres);
    std::vector<float*> d_blocks;
    for (auto& b : blocks) d_blocks.push_back(dev(b));
    const auto post_of = [&](int r0, int rows) {
        PostAttn a;
        for (int i = 0; i < nb; ++i) a.blocks[a.n_blocks++] = d_blocks[i] + (size_t) r0 * E;
        a.o = d_o + (size_t) r0 * E; a.pos = dpos + r0; a.taps = dev(taps); a.hid = d_hid + (size_t) r0 * E;
        a.resid = zeros<float>((size_t) rows * E); a.query = dev(q); a.norm = dev(w);
        a.q8 = zeros<uint8_t>((size_t) rows * E / 32 * 36); a.norm_out = zeros<float>((size_t) rows * E);
        return a;
    };
    const auto moe_of = [&](int r0, int rows) {
        MoeFinish a;
        a.rows = d_xr + (size_t) r0 * K * E; a.w = d_xw + (size_t) r0 * K; a.k = K; a.pos = dpos + r0; a.taps = dev(mt);
        a.resid = d_mres + (size_t) r0 * E; a.out = zeros<float>((size_t) rows * E);
        return a;
    };
    std::vector<PostAttn> pa(S);
    std::vector<MoeFinish> ma(S);
    for (int i = 0; i < S; ++i) {
        pa[i] = post_of(row0[i], segs[i].T);
        pa[i].ring = dev(ar[i]);
        post_attn(pa[i], segs[i].T, nullptr);
        ma[i] = moe_of(row0[i], segs[i].T);
        ma[i].ring = dev(mr[i]);
        moe_finish(ma[i], segs[i].T, nullptr);
    }
    PostAttn pb = post_of(0, R);
    MoeFinish mb = moe_of(0, R);
    pb.seg = mb.seg = sg;
    for (int i = 0; i < S; ++i) {
        pb.seg.c[i].a_ring = dev(ar[i]);
        mb.seg.c[i].m_ring = dev(mr[i]);
    }
    pb.ring = pb.seg.c[0].a_ring;
    mb.ring = mb.seg.c[0].m_ring;
    post_attn(pb, R, nullptr);
    moe_finish(mb, R, nullptr);
    cudaDeviceSynchronize();
    bool post_same = true, moe_same = true;
    const size_t ring_e = (size_t) kConvRing * E;
    const std::vector<float> nb_all = host(pb.norm_out, (size_t) R * E), res_all = host(pb.resid, (size_t) R * E), out_all = host(mb.out, (size_t) R * E);
    const std::vector<uint8_t> q8_all = host((uint8_t*) pb.q8, (size_t) R * E / 32 * 36);
    for (int i = 0; i < S; ++i) {
        const size_t n = (size_t) segs[i].T * E, at = (size_t) row0[i] * E;
        post_same = post_same && !std::memcmp(host(pa[i].norm_out, n).data(), nb_all.data() + at, n * 4) &&
                    !std::memcmp(host(pa[i].resid, n).data(), res_all.data() + at, n * 4) &&
                    !std::memcmp(host((uint8_t*) pa[i].q8, n / 32 * 36).data(), q8_all.data() + at / 32 * 36, n / 32 * 36) &&
                    host(pa[i].ring, ring_e) == host(pb.seg.c[i].a_ring, ring_e);
        moe_same = moe_same && !std::memcmp(host(ma[i].out, n).data(), out_all.data() + at, n * 4) &&
                   host(ma[i].ring, ring_e) == host(mb.seg.c[i].m_ring, ring_e);
    }
    std::printf("%-46s %s\n", "batch window: post_attn", post_same ? "bitwise = each conversation alone" : "FAIL");
    std::printf("%-46s %s\n", "batch window: moe_finish", moe_same ? "bitwise = each conversation alone" : "FAIL");
    failures += !post_same + !moe_same;
}

}  // namespace

void test_e4m3_fast_unpack() {
    uint16_t* dout = zeros<uint16_t>(256);
    e4m3_fast_unpack_all_256(dout, nullptr);
    cudaDeviceSynchronize();
    const std::vector<uint16_t> got = host(dout, 256);
    bool finite_ok = true, nan_ok = true;
    for (int b = 0; b < 256; ++b) {
        const bool is_nan = (b & 0x7F) == 0x7F;
        if (is_nan) {
            const uint16_t want_nan_map = (b & 0x80) ? 0xDF80u : 0x5F80u;   // +/-480.0h
            if (got[b] != want_nan_map) nan_ok = false;
        } else {
            const uint8_t u8 = (uint8_t) b;
            const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) u8, __NV_E4M3);
            if (got[b] != hr.x) {
                std::printf("  e4m3 mismatch byte 0x%02x: got 0x%04x want 0x%04x\n", b, got[b], hr.x);
                finite_ok = false;
            }
        }
    }
    std::printf("%-34s %s\n", "e4m3 fast unpack (254 finite)", finite_ok ? "bitwise" : "FAIL");
    std::printf("%-34s %s\n", "e4m3 fast unpack (NaN->+-480h)", nan_ok ? "ok" : "FAIL");
    failures += !finite_ok + !nan_ok;
    cudaFree(dout);
}

void test_attention_krows_invariance() {
    // Verify bitwise T=1..8 invariance across kRows in {1, 2, 4} for both FP16 and FP8, including far-tile kRel skip (pos > 2048).
    const int maxR = 8;
    const int p0 = 2990;
    const int64_t cap = 3072;
    std::vector<int> pos(maxR);
    for (int r = 0; r < maxR; ++r) pos[r] = p0 + r;
    const std::vector<float> q = randn((size_t) maxR * kHeads * kHeadDim, 0.8f);
    const std::vector<float> rs = randn((size_t) maxR * kHeads * kRel, 1.0f);
    const std::vector<float> sink = randn((size_t) maxR * kHeads, 2.0f);
    const std::vector<float> table = randn((size_t) kRel * kExtent, 0.3f);
    float *dq = dev(q), *drs = dev(rs), *dsink = dev(sink), *dt = dev(table);
    int* dpos = dev(pos);
    void* scr = zeros<uint8_t>(attention_decode_scratch(maxR, cap));

    for (int mode = 0; mode < 3; ++mode) {   // 0: FP16, 1: E4M3, 2: KvInt8 g64
        const bool fp8 = mode == 1, i8 = mode == 2;
        const size_t n = (size_t) kKvHeads * cap * kHeadDim;
        void *dk = nullptr, *dv = nullptr;
        uint16_t *dks = nullptr, *dvs = nullptr;
        std::vector<uint16_t> kh_deq, vh_deq;
        if (fp8) {
            std::vector<uint8_t> k8(n), v8(n);
            for (size_t i = 0; i < n; ++i) {
                k8[i] = __nv_fp8_e4m3(randn(1, 0.6f)[0]).__x;
                v8[i] = __nv_fp8_e4m3(randn(1, 1.0f)[0]).__x;
            }
            dk = dev(k8);
            dv = dev(v8);
        } else if (i8) {
            std::vector<int8_t> k8(n), v8(n);
            std::vector<uint16_t> ks(n / 64), vs(n / 64);
            kh_deq.resize(n); vh_deq.resize(n);
            for (size_t g = 0; g < n / 64; ++g) {
                std::vector<float> fk = randn(64, 0.6f), fv = randn(64, 1.0f);
                float ak = 0.f, av = 0.f;
                for (int i = 0; i < 64; ++i) { ak = std::max(ak, std::fabs(fk[i])); av = std::max(av, std::fabs(fv[i])); }
                ks[g] = f2h(ak / 127.f); vs[g] = f2h(av / 127.f);
                const float sk = (float) h2d(ks[g]), sv = (float) h2d(vs[g]);
                for (int i = 0; i < 64; ++i) {
                    k8[g * 64 + i] = (int8_t) std::max(-127.f, std::min(127.f, sk > 0 ? std::rint(fk[i] / sk) : 0.f));
                    v8[g * 64 + i] = (int8_t) std::max(-127.f, std::min(127.f, sv > 0 ? std::rint(fv[i] / sv) : 0.f));
                    kh_deq[g * 64 + i] = f2h((float) k8[g * 64 + i] * sk);
                    vh_deq[g * 64 + i] = f2h((float) v8[g * 64 + i] * sv);
                }
            }
            dk = dev(k8); dv = dev(v8); dks = dev(ks); dvs = dev(vs);
        } else {
            std::vector<uint16_t> kh(n), vh(n);
            for (size_t i = 0; i < n; ++i) {
                kh[i] = f2h(randn(1, 0.6f)[0]);
                vh[i] = f2h(randn(1, 1.0f)[0]);
            }
            dk = dev(kh);
            dv = dev(vh);
        }
        AttnArgs base;
        base.q = dq; base.r = drs; base.sink = dsink; base.rel_table = dt; base.pos = dpos;
        base.k_cache = dk; base.v_cache = dv; base.k_scale = dks; base.v_scale = dvs;
        base.cap = cap; base.fp8 = fp8; base.window = 0;

        // Reference: each row run alone (kRows=1)
        std::vector<float> ref_alone((size_t) maxR * kHeads * kHeadDim);
        float* out1 = zeros<float>((size_t) kHeads * kHeadDim);
        for (int r = 0; r < maxR; ++r) {
            AttnArgs o = base;
            o.q = dq + (size_t) r * kHeads * kHeadDim;
            o.r = drs + (size_t) r * kHeads * kRel;
            o.sink = dsink + (size_t) r * kHeads;
            o.pos = dpos + r;
            attention_decode(o, 1, p0 + maxR, scr, out1, nullptr);
            const std::vector<float> g = host(out1, (size_t) kHeads * kHeadDim);
            std::copy(g.begin(), g.end(), ref_alone.begin() + (size_t) r * kHeads * kHeadDim);
        }
        bool all_ok = true;
        float* outR = zeros<float>((size_t) maxR * kHeads * kHeadDim);
        for (int R : {2, 3, 4, 5, 7, 8}) {
            attention_decode(base, R, p0 + maxR, scr, outR, nullptr);
            const std::vector<float> got = host(outR, (size_t) R * kHeads * kHeadDim);
            if (std::memcmp(got.data(), ref_alone.data(), (size_t) R * kHeads * kHeadDim * 4) != 0) all_ok = false;
        }
        char label[64];
        std::snprintf(label, sizeof label, "attention T=1..8 (%s, far)", i8 ? "KvInt8" : fp8 ? "E4M3" : "FP16");
        std::printf("%-34s %s\n", label, all_ok ? "bitwise" : "FAIL");
        failures += !all_ok;
        if (i8) {
            // KvInt8 g64 must be bitwise identical to FP16 on the dequantized K/V cells!
            void *dkd = dev(kh_deq), *dvd = dev(vh_deq);
            AttnArgs fd = base;
            fd.k_cache = dkd; fd.v_cache = dvd; fd.k_scale = nullptr; fd.v_scale = nullptr; fd.fp8 = false;
            float* out_deq = zeros<float>((size_t) maxR * kHeads * kHeadDim);
            attention_decode(fd, maxR, p0 + maxR, scr, out_deq, nullptr);
            const std::vector<float> gdeq = host(out_deq, (size_t) maxR * kHeads * kHeadDim);
            same_bits("attention KvInt8 == dequant FP16", ref_alone.data(), gdeq.data(), gdeq.size() * 4);
            cudaFree(dkd); cudaFree(dvd); cudaFree(out_deq);
        }
        // P2.1 TP2 sharded KV parity: two 4-head sharded slabs (kv_stride = 4) must match 8-head ref_alone bitwise!
        {
            const int ng = kKvHeads / 2;
            const size_t elem_b = (fp8 || i8) ? 1 : 2;
            const size_t half_cell = (size_t) ng * kHeadDim * elem_b;
            const size_t full_cell = 2 * half_cell;
            const size_t half_sc = (size_t) ng * (kHeadDim / 64);
            const std::vector<uint8_t> fk_h = host(static_cast<const uint8_t*>(dk), (size_t) cap * full_cell);
            const std::vector<uint8_t> fv_h = host(static_cast<const uint8_t*>(dv), (size_t) cap * full_cell);
            std::vector<uint16_t> fks_h, fvs_h;
            if (i8) {
                fks_h = host(dks, (size_t) cap * 2 * half_sc);
                fvs_h = host(dvs, (size_t) cap * 2 * half_sc);
            }
            std::vector<float> combined((size_t) maxR * kHeads * kHeadDim, 0.f);
            float* out_sh = zeros<float>((size_t) maxR * kHeads * kHeadDim);
            for (int card = 0; card < 2; ++card) {
                std::vector<uint8_t> sk((size_t) cap * half_cell), sv((size_t) cap * half_cell);
                std::vector<uint16_t> sks, svs;
                if (i8) { sks.resize((size_t) cap * half_sc); svs.resize((size_t) cap * half_sc); }
                for (int64_t ce = 0; ce < cap; ++ce) {
                    std::memcpy(&sk[ce * half_cell], &fk_h[ce * full_cell + card * half_cell], half_cell);
                    std::memcpy(&sv[ce * half_cell], &fv_h[ce * full_cell + card * half_cell], half_cell);
                    if (i8) {
                        std::memcpy(&sks[ce * half_sc], &fks_h[ce * 2 * half_sc + card * half_sc], half_sc * 2);
                        std::memcpy(&svs[ce * half_sc], &fvs_h[ce * 2 * half_sc + card * half_sc], half_sc * 2);
                    }
                }
                void *dsk = dev(sk), *dsv = dev(sv);
                uint16_t *dsks = i8 ? dev(sks) : nullptr, *dsvs = i8 ? dev(svs) : nullptr;
                AttnArgs sa = base;
                sa.k_cache = dsk; sa.v_cache = dsv; sa.k_scale = dsks; sa.v_scale = dsvs;
                sa.g0 = card * ng; sa.ng = ng; sa.kv_stride = ng;
                attention_decode(sa, maxR, p0 + maxR, scr, out_sh, nullptr);
                const std::vector<float> got_c = host(out_sh, combined.size());
                const int h0 = sa.g0 * (kHeads / kKvHeads), nh = sa.ng * (kHeads / kKvHeads);
                for (int r = 0; r < maxR; ++r)
                    std::memcpy(&combined[((size_t) r * kHeads + h0) * kHeadDim],
                                &got_c[((size_t) r * kHeads + h0) * kHeadDim], (size_t) nh * kHeadDim * 4);
                cudaFree(dsk); cudaFree(dsv);
                if (dsks) cudaFree(dsks);
                if (dsvs) cudaFree(dsvs);
            }
            cudaFree(out_sh);
            char sh_label[64];
            std::snprintf(sh_label, sizeof sh_label, "attention TP2 KV shard (%s)", i8 ? "KvInt8" : fp8 ? "E4M3" : "FP16");
            same_bits(sh_label, ref_alone.data(), combined.data(), combined.size() * 4);
        }
        if (dks) cudaFree(dks);
        if (dvs) cudaFree(dvs);
        cudaFree(out1); cudaFree(outR); cudaFree(dk); cudaFree(dv);
    }
    cudaFree(dq); cudaFree(drs); cudaFree(dsink); cudaFree(dt); cudaFree(dpos); cudaFree(scr);
}

// P1.5b: bit-exact parity of router_plan_fused vs standalone bk::router + moe_plan, plus Section 7.1
// >= 1,200 CUDA Graph replay stress test for BOTH fused controllers() and router_plan_fused() with concurrent side stream.
void test_p15_fusions_and_graph_replay() {
    const int n_exp = 128, k = 8, maxR = kDecodeRows;
    const std::vector<float> x = randn((size_t) maxR * E, 0.8f);
    const std::vector<float> gate = randn((size_t) n_exp * E, 0.2f);
    const std::vector<float> bias = randn((size_t) n_exp, 0.3f);
    std::vector<unsigned long long> blobs_all(n_exp), blobs_spill(n_exp);
    for (int e = 0; e < n_exp; ++e) {
        blobs_all[e] = 0x10000000ull + (unsigned long long) e * 4096ull;
        blobs_spill[e] = (e % 5 == 0) ? 0ull : blobs_all[e];
    }
    float *dx = dev(x), *dgate = dev(gate), *dbias = dev(bias);
    unsigned long long *dblobs_all = dev(blobs_all), *dblobs_spill = dev(blobs_spill);
    float *rlog_ref = zeros<float>((size_t) maxR * n_exp), *rlog_got = zeros<float>((size_t) maxR * n_exp);
    float *rw_ref = zeros<float>((size_t) maxR * k), *rw_got = zeros<float>((size_t) maxR * k);
    int32_t *ids_ref = zeros<int32_t>((size_t) maxR * k), *ids_got = zeros<int32_t>((size_t) maxR * k);
    const size_t pairs = (size_t) maxR * k;
    unsigned long long *ptr_ref = zeros<unsigned long long>(pairs), *ptr_got = zeros<unsigned long long>(pairs);
    int32_t *st_ref = zeros<int32_t>(pairs + 2), *st_got = zeros<int32_t>(pairs + 2);
    int32_t *ng_ref = zeros<int32_t>(1), *ng_got = zeros<int32_t>(1);
    int32_t *dst_ref = zeros<int32_t>(pairs), *dst_got = zeros<int32_t>(pairs);
    int32_t *tok_ref = zeros<int32_t>(pairs), *tok_got = zeros<int32_t>(pairs);
    uint32_t *tal_ref = zeros<uint32_t>(n_exp), *tal_got = zeros<uint32_t>(n_exp);
    unsigned long long *rec_ref = zeros<unsigned long long>(kAheadExperts), *rec_got = zeros<unsigned long long>(kAheadExperts);
    int32_t *recn_ref = zeros<int32_t>(1), *recn_got = zeros<int32_t>(1);
    uint32_t* done = zeros<uint32_t>(1);

    bool bit_ok = true;
    for (bool spill : {false, true}) {
        unsigned long long* dblobs = spill ? dblobs_spill : dblobs_all;
        for (int R = 1; R <= maxR; ++R) {
            cudaMemset(tal_ref, 0, (size_t) n_exp * 4);
            cudaMemset(tal_got, 0, (size_t) n_exp * 4);
            bk::RouterArgs ra;
            ra.n_expert = n_exp; ra.n_embd = E; ra.k = k; ra.weight_scale = 2.5f; ra.gate = bk::RouterGate::Sigmoid;
            bk::router(dx, dgate, dbias, R, ra, rlog_ref, ids_ref, rw_ref, nullptr);
            moe_plan(ids_ref, R, k, n_exp, dblobs, ptr_ref, st_ref, ng_ref, dst_ref, tok_ref, nullptr, tal_ref, rec_ref, recn_ref);

            RouterPlanArgs rp;
            rp.x = dx; rp.gate = dgate; rp.bias = dbias; rp.n_expert = n_exp; rp.k = k; rp.weight_scale = 2.5f;
            rp.blobs = dblobs; rp.logits = rlog_got; rp.ids = ids_got; rp.weights = rw_got;
            rp.ptr = ptr_got; rp.start = st_got; rp.n_groups = ng_got; rp.dst = dst_got; rp.tok = tok_got;
            rp.done = done; rp.tally = tal_got; rp.rec_blobs = rec_got; rp.rec_n = recn_got;
            router_plan_fused(rp, R, nullptr);
            cudaDeviceSynchronize();

            const int ng = host(ng_ref, 1)[0];
            const int ne = host(st_ref, (size_t) ng + 1)[ng];
            const uint32_t done_after = host(done, 1)[0];
            bool ok = done_after == 0u &&
                      host(ng_got, 1)[0] == ng &&
                      host(recn_got, 1)[0] == host(recn_ref, 1)[0] &&
                      host(rlog_got, (size_t) R * n_exp) == host(rlog_ref, (size_t) R * n_exp) &&
                      host(ids_got, (size_t) R * k) == host(ids_ref, (size_t) R * k) &&
                      host(rw_got, (size_t) R * k) == host(rw_ref, (size_t) R * k) &&
                      host(ptr_got, (size_t) ng) == host(ptr_ref, (size_t) ng) &&
                      host(st_got, (size_t) ng + 2) == host(st_ref, (size_t) ng + 2) &&
                      host(dst_got, (size_t) ne) == host(dst_ref, (size_t) ne) &&
                      host(tok_got, (size_t) ne) == host(tok_ref, (size_t) ne) &&
                      host(tal_got, (size_t) n_exp) == host(tal_ref, (size_t) n_exp) &&
                      host(rec_got, kAheadExperts) == host(rec_ref, kAheadExperts);
            if (!ok) bit_ok = false;
        }
    }
    std::printf("%-34s %s\n", "router_plan_fused T=1..8 (all+sp)", bit_ok ? "bitwise" : "FAIL");
    failures += !bit_ok;
    // Also test fallback branch (n_expert > 128) across n_expert={160, 256} and k={1, 8, 16}.
    bool fb_ok = true;
    for (int ne_fb : {160, 256}) {
        const std::vector<float> g_fb = randn((size_t) ne_fb * E, 0.2f);
        const std::vector<float> b_fb = randn((size_t) ne_fb, 0.3f);
        std::vector<unsigned long long> bl_fb(ne_fb);
        for (int e = 0; e < ne_fb; ++e) bl_fb[e] = (e % 7 == 0) ? 0ull : (0x20000000ull + (unsigned long long) e * 4096ull);
        float *dg_fb = dev(g_fb), *db_fb = dev(b_fb);
        unsigned long long* dbl_fb = dev(bl_fb);
        float *lr_fb = zeros<float>((size_t) maxR * ne_fb), *lg_fb = zeros<float>((size_t) maxR * ne_fb);
        float *wr_fb = zeros<float>((size_t) maxR * 16), *wg_fb = zeros<float>((size_t) maxR * 16);
        int32_t *ir_fb = zeros<int32_t>((size_t) maxR * 16), *ig_fb = zeros<int32_t>((size_t) maxR * 16);
        unsigned long long *pr_fb = zeros<unsigned long long>((size_t) maxR * 16), *pg_fb = zeros<unsigned long long>((size_t) maxR * 16);
        int32_t *sr_fb = zeros<int32_t>((size_t) maxR * 16 + 2), *sg_fb = zeros<int32_t>((size_t) maxR * 16 + 2);
        int32_t *dr_fb = zeros<int32_t>((size_t) maxR * 16), *dg2_fb = zeros<int32_t>((size_t) maxR * 16);
        int32_t *tr_fb = zeros<int32_t>((size_t) maxR * 16), *tg_fb = zeros<int32_t>((size_t) maxR * 16);
        for (int k_fb : {1, 8, 16}) {
            for (int R : {1, 3, 4}) {
                bk::RouterArgs ra;
                ra.n_expert = ne_fb; ra.n_embd = E; ra.k = k_fb; ra.weight_scale = 2.5f; ra.gate = bk::RouterGate::Sigmoid;
                bk::router(dx, dg_fb, db_fb, R, ra, lr_fb, ir_fb, wr_fb, nullptr);
                moe_plan(ir_fb, R, k_fb, ne_fb, dbl_fb, pr_fb, sr_fb, ng_ref, dr_fb, tr_fb, nullptr);
                RouterPlanArgs rp;
                rp.x = dx; rp.gate = dg_fb; rp.bias = db_fb; rp.n_expert = ne_fb; rp.k = k_fb; rp.weight_scale = 2.5f;
                rp.blobs = dbl_fb; rp.logits = lg_fb; rp.ids = ig_fb; rp.weights = wg_fb;
                rp.ptr = pg_fb; rp.start = sg_fb; rp.n_groups = ng_got; rp.dst = dg2_fb; rp.tok = tg_fb;
                rp.done = done;
                router_plan_fused(rp, R, nullptr);
                cudaDeviceSynchronize();
                const int ng = host(ng_ref, 1)[0];
                const int ne = host(sr_fb, (size_t) ng + 1)[ng];
                if (host(done, 1)[0] != 0u || host(ng_got, 1)[0] != ng ||
                    host(ig_fb, (size_t) R * k_fb) != host(ir_fb, (size_t) R * k_fb) ||
                    host(wg_fb, (size_t) R * k_fb) != host(wr_fb, (size_t) R * k_fb) ||
                    host(pg_fb, (size_t) ng) != host(pr_fb, (size_t) ng) ||
                    host(sg_fb, (size_t) ng + 2) != host(sr_fb, (size_t) ng + 2) ||
                    host(dg2_fb, (size_t) ne) != host(dr_fb, (size_t) ne) ||
                    host(tg_fb, (size_t) ne) != host(tr_fb, (size_t) ne))
                    fb_ok = false;
            }
        }
        cudaFree(dg_fb); cudaFree(db_fb); cudaFree(dbl_fb);
        cudaFree(lr_fb); cudaFree(lg_fb); cudaFree(wr_fb); cudaFree(wg_fb);
        cudaFree(ir_fb); cudaFree(ig_fb); cudaFree(pr_fb); cudaFree(pg_fb);
        cudaFree(sr_fb); cudaFree(sg_fb); cudaFree(dr_fb); cudaFree(dg2_fb);
        cudaFree(tr_fb); cudaFree(tg_fb);
    }
    std::printf("%-34s %s\n", "router_plan_fused >128 fallback", fb_ok ? "bitwise" : "FAIL");
    failures += !fb_ok;
    for (int R : {1, 2, 4, 8}) {
        bk::RouterArgs ra;
        ra.n_expert = n_exp; ra.n_embd = E; ra.k = k; ra.weight_scale = 2.5f; ra.gate = bk::RouterGate::Sigmoid;
        RouterPlanArgs rp;
        rp.x = dx; rp.gate = dgate; rp.bias = dbias; rp.n_expert = n_exp; rp.k = k; rp.weight_scale = 2.5f;
        rp.blobs = dblobs_all; rp.logits = rlog_got; rp.ids = ids_got; rp.weights = rw_got;
        rp.ptr = ptr_got; rp.start = st_got; rp.n_groups = ng_got; rp.dst = dst_got; rp.tok = tok_got;
        rp.done = done;
        const double t_sep = time_us([&] {
            bk::router(dx, dgate, dbias, R, ra, rlog_ref, ids_ref, rw_ref, nullptr);
            moe_plan(ids_ref, R, k, n_exp, dblobs_all, ptr_ref, st_ref, ng_ref, dst_ref, tok_ref, nullptr);
        }, 200);
        const double t_fus = time_us([&] { router_plan_fused(rp, R, nullptr); }, 200);
        std::printf("  router+plan R=%d: sep=%.2f us  fused=%.2f us  (delta=%+.2f us)\n", R, t_sep, t_fus, t_fus - t_sep);
    }

    // Section 7.1 CUDA Graph stress test: 1,200 consecutive replays across T={1,2,4,8} of BOTH fused controllers()
    // and router_plan_fused() on main stream + concurrent standalone bk::router on side stream (k.ra_st model).
    cudaStream_t main_st = nullptr, side_st = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    cudaStreamCreateWithFlags(&main_st, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&side_st, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming);

    const int S = 3, rank = kCtrlRank;
    CtrlArgs ca;
    ca.scratch = zeros<float>(ctrl_scratch_floats());
    ca.norm_out = zeros<float>((size_t) maxR * E);
    for (int s = 0; s < S; ++s) {
        ca.dest[s] = dev(randn((size_t) maxR * E, 0.7f));
        ca.msg[s] = dev(randn((size_t) maxR * E, 1.1f));
        std::vector<uint16_t> dn((size_t) rank * 2 * E), up((size_t) 2 * E * rank);
        const auto rf = randn(dn.size(), 0.05f), uf = randn(up.size(), 0.05f);
        const auto to_bf = [](float v) -> uint16_t {
            uint32_t u = 0;
            std::memcpy(&u, &v, 4);
            return (uint16_t) (u >> 16);
        };
        for (size_t i = 0; i < dn.size(); ++i) dn[i] = to_bf(rf[i]);
        for (size_t i = 0; i < up.size(); ++i) up[i] = to_bf(uf[i]);
        ca.down[s] = dev(dn);
        ca.up[s] = dev(up);
        ca.norm[s] = dev(randn(E, 0.2f, 1.f));
        ca.q8[s] = zeros<uint8_t>((size_t) maxR * E / 32 * 36);
    }
    float *gs_log = zeros<float>((size_t) maxR * n_exp), *gs_w = zeros<float>((size_t) maxR * k);
    int32_t* gs_ids = zeros<int32_t>((size_t) maxR * k);

    bool graph_ok = true;
    const int kShapeList[4] = {1, 2, 4, 8};
    cudaGraphExec_t execs[4] = {};
    std::vector<float> exp_ctrl[4], exp_rw[4];
    std::vector<int32_t> exp_ids[4], exp_dst[4];
    for (int si = 0; si < 4; ++si) {
        const int R = kShapeList[si];
        RouterPlanArgs rp;
        rp.x = dx; rp.gate = dgate; rp.bias = dbias; rp.n_expert = n_exp; rp.k = k; rp.weight_scale = 2.5f;
        rp.blobs = dblobs_all; rp.logits = rlog_got; rp.ids = ids_got; rp.weights = rw_got;
        rp.ptr = ptr_got; rp.start = st_got; rp.n_groups = ng_got; rp.dst = dst_got; rp.tok = tok_got;
        rp.done = done;
        bk::RouterArgs ga;
        ga.n_expert = n_exp; ga.n_embd = E; ga.k = k; ga.weight_scale = 2.5f; ga.gate = bk::RouterGate::Sigmoid;

        // Reference direct run
        controllers(ca, S, R, main_st);
        router_plan_fused(rp, R, main_st);
        cudaStreamSynchronize(main_st);
        exp_ctrl[si] = host(ca.norm_out, (size_t) R * E);
        exp_ids[si] = host(ids_got, (size_t) R * k);
        exp_rw[si] = host(rw_got, (size_t) R * k);
        exp_dst[si] = host(dst_got, (size_t) R * k);

        // Capture graph with concurrent side stream
        cudaGraph_t gr = nullptr;
        if (cudaStreamBeginCapture(main_st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) graph_ok = false;
        controllers(ca, S, R, main_st);
        cudaEventRecord(ev_fork, main_st);
        router_plan_fused(rp, R, main_st);
        cudaStreamWaitEvent(side_st, ev_fork, 0);
        bk::router(dx, dgate, dbias, R, ga, gs_log, gs_ids, gs_w, side_st);
        cudaEventRecord(ev_join, side_st);
        cudaStreamWaitEvent(main_st, ev_join, 0);
        if (cudaStreamEndCapture(main_st, &gr) != cudaSuccess ||
            cudaGraphInstantiate(&execs[si], gr, nullptr, nullptr, 0) != cudaSuccess)
            graph_ok = false;
        if (gr) cudaGraphDestroy(gr);
    }
    // 1,200 consecutive replays cycling across T={1,2,4,8}
    for (int rep = 0; rep < 1200 && graph_ok; ++rep) {
        const int si = rep & 3;
        const int R = kShapeList[si];
        if (cudaGraphLaunch(execs[si], main_st) != cudaSuccess) { graph_ok = false; break; }
        if ((rep % 100) == 99 || rep == 1199) {
            if (cudaStreamSynchronize(main_st) != cudaSuccess) { graph_ok = false; break; }
            if (host(done, 1)[0] != 0u ||
                host(ca.norm_out, (size_t) R * E) != exp_ctrl[si] ||
                host(ids_got, (size_t) R * k) != exp_ids[si] ||
                host(rw_got, (size_t) R * k) != exp_rw[si] ||
                host(dst_got, (size_t) R * k) != exp_dst[si]) {
                graph_ok = false;
                break;
            }
        }
    }
    for (int si = 0; si < 4; ++si)
        if (execs[si]) cudaGraphExecDestroy(execs[si]);
    cudaEventDestroy(ev_fork); cudaEventDestroy(ev_join);
    cudaStreamDestroy(main_st); cudaStreamDestroy(side_st);
    std::printf("%-34s %s\n", "P1.5b 1200x CUDA Graph stress", graph_ok ? "bitwise (0 hangs)" : "FAIL");
    failures += !graph_ok;
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    test_e4m3_fast_unpack();
    test_mix();
    test_ctrl();
    test_prepare();
    test_attention();
    test_attention_krows_invariance();
    test_post_conv();
    test_experts();
    test_expert_halves(40);
    test_expert_halves(8);
    test_softcap();
    test_mix_norm();
    test_controllers();
    test_decode_prepare();
    test_post_attn();
    test_expert_prefetch();
    test_moe_finish();
    test_cpu_merge();
    test_batch_window();
    test_p15_fusions_and_graph_replay();
    std::printf("inkling2_kernels_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
