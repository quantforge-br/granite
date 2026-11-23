// prefill_mmq_mxfp4_test - the prompt path's MXFP4 down products through MMQ (moe_mmq.hpp): the MXFP4 tile loader
// (IQ4_NL's with byte loads, the FP4 table and the E8M0 scale) at the GSQ GGUFs' down geometry, [2304, 1152] and a
// card's [2304, 576] half under tensor parallelism.  Four experts in one launch, their matrices a stride apart that is
// whole 17-byte blocks but leaves the rows only byte aligned at 576 (306 bytes); random bytes follow each matrix, as an
// expert blob's do, for MMQ's reads past a row's end.  Random codes with E8M0 scales over 2^-13 .. 2^-6 and the two
// denormal codes.  Checked against double-precision math on MMQ's own activation rounding (x rounded to int8 per 32
// values the way the quantizer does): only fp32 accumulation order differs.  Tolerance: RMS 1e-5, worst row 1e-4.
// Exit 77 without a CUDA device.
#include "granite/prefill/moe_mmq.hpp"

#include "granite/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = granite::prefill::mmq;

constexpr int N = 2304, NE = 4, BLOCK = 17;
constexpr int kRows[NE] = {1, 7, 33, 64};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

// MMQ's activation rounding (quantize.cu): per 32 values d = 1 / (127 / amax), q = round(x * 127 / amax)
void q8_round(const float* x, float* y, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (int j = 0; j < 32; ++j) y[b + j] = amax == 0.0f ? 0.0f : std::round(x[b + j] * d_inv) * d;
    }
}

int check(int ff, std::mt19937& rng, cudaStream_t s) {
    const int t = GGML_TYPE_MXFP4;
    const auto* tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    const size_t mat = mmq::matrix_bytes(t, N, ff), stride = (mat / BLOCK + 61) * BLOCK;   // 61 blocks of other bytes
    std::vector<uint8_t> w(stride * NE + mmq::read_past_bytes(t, ff));
    for (auto& b : w) b = (uint8_t) (rng() & 0xFF);
    std::vector<float> wf((size_t) NE * N * ff);
    for (int e = 0; e < NE; ++e) {
        uint8_t* m = w.data() + (size_t) e * stride;
        for (size_t o = 0; o < mat; o += BLOCK) m[o] = (uint8_t) (rng() % 16 == 0 ? rng() % 2 : 115 + rng() % 8);
        tr->to_float(m, wf.data() + (size_t) e * N * ff, (int64_t) N * ff);
    }
    int total = 0;
    std::vector<int32_t> bounds(1, 0);
    for (int r : kRows) bounds.push_back(total += r);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) total * ff), xr(x.size());
    for (int r = 0; r < total; ++r)
        for (int k = 0; k < ff; ++k) x[(size_t) r * ff + k] = nd(rng) * (0.25f + (float) (r % 5));
    for (int r = 0; r < total; ++r) q8_round(x.data() + (size_t) r * ff, xr.data() + (size_t) r * ff, ff);
    std::vector<float> ref((size_t) total * N);
    for (int e = 0; e < NE; ++e)
        for (int r = bounds[e]; r < bounds[e + 1]; ++r)
            for (int o = 0; o < N; ++o) {
                const float* wr = wf.data() + ((size_t) e * N + o) * ff;
                double acc = 0;
                for (int k = 0; k < ff; ++k) acc += (double) wr[k] * xr[(size_t) r * ff + k];
                ref[(size_t) r * N + o] = (float) acc;
            }

    Dev dw(w.size()), dx(x.size() * 4), dxq(mmq::q8_bytes(total, ff)), db(bounds.size() * 4), di((size_t) total * 4),
        dd((size_t) total * N * 4);
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemset(dd.p, 0xFF, (size_t) total * N * 4), "dst");
    mmq::iota(di.as<int32_t>(), total, s);
    mmq::load(t, N);
    mmq::quantize(dx.as<float>(), nullptr, dxq.p, t, ff, ff, total, s);
    mmq::Product p;
    p.w = dw.p;
    p.type = t;
    p.w_rows = N;
    p.w_cols = ff;
    p.expert_bytes = stride;
    p.n = NE;
    p.xq = dxq.p;
    p.bounds = db.as<int32_t>();
    p.ids = di.as<int32_t>();
    p.total_rows = total;
    p.max_rows = kRows[NE - 1];
    p.dst = dd.as<float>();
    p.ld_dst = N;
    mmq::Context ctx;
    ctx.run(p, s);
    ck(cudaStreamSynchronize(s), "run");
    std::vector<float> got((size_t) total * N);
    ck(cudaMemcpy(got.data(), dd.p, got.size() * 4, cudaMemcpyDeviceToHost), "got");

    double e2 = 0, r2 = 0, worst = 0;
    bool finite = true;
    for (int r = 0; r < total; ++r) {
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            const float g = got[(size_t) r * N + o], v = ref[(size_t) r * N + o];
            finite = finite && std::isfinite(g);
            const double d = (double) g - v;
            e2 += d * d;
            r2 += (double) v * v;
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(v));
        }
        worst = std::max(worst, mr > 0 ? me / mr : me > 0 ? 1e30 : 0.0);
    }
    const double rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    const bool ok = finite && rms <= 1e-5 && worst <= 1e-4;
    std::printf("MXFP4 down [%d, %d], %d experts, %d rows: RMS %.2e, worst row %.2e%s  %s\n", N, ff, NE, total, rms,
                worst, finite ? "" : ", non-finite outputs", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("prefill_mmq_mxfp4_test: no CUDA device, skipped\n");
        return 77;
    }
    try {
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        std::mt19937 rng(39);
        int failures = 0;
        for (int ff : {1152, 576}) failures += check(ff, rng, s);
        cudaStreamDestroy(s);
        std::printf("prefill_mmq_mxfp4_test: %d failures\n", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_mmq_mxfp4_test: %s\n", e.what());
        return 1;
    }
}
