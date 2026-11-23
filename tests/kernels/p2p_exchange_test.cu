// tests/kernels/p2p_exchange_test.cu - the exchange a two-card tensor-parallel decode makes after every visit, measured
// and checked on the first two cards.  Each card holds a partial [rows][2304] FP32 sum (rows 1..8: 9 to 72 KiB) and needs
// the other's; both add them in card order (card 0's + card 1's), so the two cards hold the same bits.
//
// Reported: the copy engines' peer bandwidth (each direction, both at once, and both at once while a memory-bound kernel
// runs on each card), then per exchange, from CUDA graphs of kIters exchanges replayed kReplays times, the time over a
// graph of the same kernels without the exchange (p50 / p95 / p99 over the replays) for two mechanisms:
//   copy    the producer writes its partial on its own card, a copy engine sends it (cudaMemcpyAsync on the unified
//           addresses: cudaMemcpyPeerAsync is not captured), an event orders the sum;
//   store   the producer writes its partial on its own card and straight into the peer's receive buffer, an event
//           orders the sum.
// The receive buffers alternate by iteration: a card's sum of iteration k comes before its event of iteration k + 1,
// which the other card waits for before its write of iteration k + 2 into the same buffer.
//
// Checked: every sum against the exact payload (small integers, so FP32 sums are exact) on both cards, over every
// replay, and over a long replay of the store mechanism; the two cards' last sums bit for bit.  Exit 77 without two
// cards with peer access both ways.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kWidth = 2304;          // Inkling 2 Mini's hidden width
constexpr int kMaxRows = 8;
constexpr int kIters = 32;            // exchanges in one graph
constexpr int kReplays = 300;
constexpr int kLongReplays = 3000;    // the long check: about 100k exchanges
constexpr int kThreads = 256;

int fails = 0;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("FAIL %s: %s\n", what, cudaGetErrorString(e));
    ++fails;
    return false;
}

// the payload card c writes at iteration it for element i: a small integer, so any sum of two is exact
__host__ __device__ inline float payload(uint32_t it, int c, int i) { return (float) ((it * 7u + (uint32_t) c * 3u + (uint32_t) i) & 1023u); }

struct Card {
    int dev = 0;
    cudaStream_t s = nullptr, busy_s = nullptr;
    float* part = nullptr;            // this card's partial
    float* recv[2] = {};              // the other card's partial, by iteration parity
    float* out = nullptr;             // the sum
    uint32_t* it = nullptr;           // the iteration counter
    uint32_t* bad = nullptr;          // wrong sums seen
    cudaEvent_t sent[kIters] = {};    // behind this card's half of an exchange
    float* big = nullptr;             // bandwidth buffers
    float* busy_buf = nullptr;
};

// card c's partial for this iteration; with `peer` set also into the peer's receive buffer (a peer pointer)
__global__ void produce(int c, int n, const uint32_t* __restrict__ it, float* __restrict__ part, float* peer) {
    const uint32_t k = *it;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = payload(k, c, i);
        part[i] = v;
        if (peer) peer[i] = v;
    }
}

// out = card 0's partial + card 1's, checked against the payload; the last block to finish advances the counter.
// `other` null: the base graph, the card's own partial twice (the same work, no exchange).
__global__ void sum(int c, int n, uint32_t* it, const float* __restrict__ mine, const float* other, float* __restrict__ out,
                    uint32_t* __restrict__ bad, uint32_t* __restrict__ done) {
    const uint32_t k = *it;
    const float* p0 = c == 0 ? mine : (other ? other : mine);
    const float* p1 = c == 0 ? (other ? other : mine) : mine;
    uint32_t wrong = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float v = p0[i] + p1[i];
        out[i] = v;
        const float want = other ? payload(k, 0, i) + payload(k, 1, i) : 2.0f * payload(k, c, i);
        wrong += v != want;
    }
    if (wrong) atomicAdd(bad, wrong);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(done, 1u) == gridDim.x - 1) {   // the last block: every block has read *it
            *done = 0;
            *it = k + 1;
        }
    }
}

// a memory-bound load: reads n floats `reps` times into a sink
__global__ void stream_read(const float4* __restrict__ p, size_t n4, int reps, float* sink) {
    float a = 0.0f;
    for (int r = 0; r < reps; ++r)
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
            const float4 v = __ldcs(&p[i]);
            a += v.x + v.y + v.z + v.w;
        }
    if (a == 12345.678f) *sink = a;
}

enum class Mech { Base, Copy, Store };
const char* name(Mech m) { return m == Mech::Base ? "base" : m == Mech::Copy ? "copy" : "store"; }

uint32_t* g_done[2] = {};

// kIters exchanges of `rows` rows into one graph, captured from card 0's stream (card 1 forked in and joined back)
bool capture(Card (&cd)[2], Mech m, int rows, cudaGraphExec_t& exec) {
    const int n = rows * kWidth;
    const int blocks = (n + kThreads - 1) / kThreads;
    cudaEvent_t fork, join;
    cudaSetDevice(0);
    cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
    cudaSetDevice(1);
    cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    cudaSetDevice(0);
    if (!ok(cudaStreamBeginCapture(cd[0].s, cudaStreamCaptureModeRelaxed), "begin capture")) return false;
    cudaEventRecord(fork, cd[0].s);
    cudaSetDevice(1);
    cudaStreamWaitEvent(cd[1].s, fork, 0);
    for (int k = 0; k < kIters; ++k) {
        const int par = k & 1;
        for (int c = 0; c < 2; ++c) {
            Card& a = cd[c];
            Card& b = cd[1 - c];
            cudaSetDevice(a.dev);
            produce<<<blocks, kThreads, 0, a.s>>>(c, n, a.it, a.part, m == Mech::Store ? b.recv[par] : nullptr);
            if (m == Mech::Copy) cudaMemcpyAsync(b.recv[par], a.part, (size_t) n * 4, cudaMemcpyDefault, a.s);
            if (m != Mech::Base) cudaEventRecord(a.sent[k], a.s);
        }
        for (int c = 0; c < 2; ++c) {
            Card& a = cd[c];
            cudaSetDevice(a.dev);
            if (m != Mech::Base) cudaStreamWaitEvent(a.s, cd[1 - c].sent[k], 0);
            sum<<<blocks, kThreads, 0, a.s>>>(c, n, a.it, a.part, m == Mech::Base ? nullptr : a.recv[par], a.out, a.bad, g_done[c]);
        }
    }
    cudaSetDevice(1);
    cudaEventRecord(join, cd[1].s);
    cudaSetDevice(0);
    cudaStreamWaitEvent(cd[0].s, join, 0);
    cudaGraph_t g;
    if (!ok(cudaStreamEndCapture(cd[0].s, &g), "end capture")) return false;
    const bool r = ok(cudaGraphInstantiate(&exec, g, 0), "instantiate");
    cudaGraphDestroy(g);
    return r;
}

struct Times {
    double p50 = 0, p95 = 0, p99 = 0;
};

// replays the graph: per replay its time on card 0's stream (the graph joins card 1 back into it), per exchange
bool replay(Card (&cd)[2], cudaGraphExec_t exec, int replays, std::vector<double>* per_exchange_us) {
    cudaSetDevice(0);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int r = 0; r < replays; ++r) {
        cudaEventRecord(t0, cd[0].s);
        if (!ok(cudaGraphLaunch(exec, cd[0].s), "graph launch")) return false;
        cudaEventRecord(t1, cd[0].s);
        if (!ok(cudaEventSynchronize(t1), "replay")) return false;
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        if (per_exchange_us) per_exchange_us->push_back(ms * 1000.0 / kIters);
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return true;
}

Times pct(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t) (q * (double) (v.size() - 1) + 0.5))]; };
    return {at(0.50), at(0.95), at(0.99)};
}

uint32_t bad_sums(Card (&cd)[2]) {
    uint32_t total = 0;
    for (Card& c : cd) {
        cudaSetDevice(c.dev);
        uint32_t b = 0;
        cudaMemcpy(&b, c.bad, 4, cudaMemcpyDeviceToHost);
        cudaMemset(c.bad, 0, 4);
        total += b;
    }
    return total;
}

// GB/s of `reps` copies of `bytes`, each direction alone, then both at once (wall time, both streams)
void bandwidth(Card (&cd)[2], size_t bytes, int reps, const char* label) {
    auto run = [&](bool d01, bool d10) {
        for (Card& c : cd) { cudaSetDevice(c.dev); cudaStreamSynchronize(c.s); }
        const auto t = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            if (d01) cudaMemcpyPeerAsync(cd[1].big, 1, cd[0].big + bytes / 4, 0, bytes, cd[0].s);
            if (d10) cudaMemcpyPeerAsync(cd[0].big, 0, cd[1].big + bytes / 4, 1, bytes, cd[1].s);
        }
        for (Card& c : cd) { cudaSetDevice(c.dev); cudaStreamSynchronize(c.s); }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        return (double) bytes * reps * ((d01 ? 1 : 0) + (d10 ? 1 : 0)) / s / 1e9;
    };
    run(true, true);   // warm
    const double a = run(true, false), b = run(false, true), both = run(true, true);
    std::printf("%-24s 0->1 %5.2f GB/s   1->0 %5.2f GB/s   both %5.2f GB/s (sum)\n", label, a, b, both);
    ok(cudaGetLastError(), label);
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 2) {
        std::printf("p2p_exchange_test: needs two GPUs, skipped\n");
        return 77;
    }
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);
    if (!can01 || !can10) {
        std::printf("p2p_exchange_test: no peer access between cards 0 and 1, skipped\n");
        return 77;
    }
    Card cd[2];
    const size_t big_bytes = (size_t) 64 << 20, busy_bytes = (size_t) 1 << 30;
    for (int c = 0; c < 2; ++c) {
        Card& k = cd[c];
        k.dev = c;
        cudaSetDevice(c);
        const cudaError_t pe = cudaDeviceEnablePeerAccess(1 - c, 0);
        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled && !ok(pe, "enable peer access")) return 1;
        (void) cudaGetLastError();
        cudaStreamCreateWithFlags(&k.s, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&k.busy_s, cudaStreamNonBlocking);
        const size_t n = (size_t) kMaxRows * kWidth * 4;
        if (!ok(cudaMalloc(&k.part, n), "malloc") || !ok(cudaMalloc(&k.recv[0], n), "malloc") || !ok(cudaMalloc(&k.recv[1], n), "malloc") ||
            !ok(cudaMalloc(&k.out, n), "malloc") || !ok(cudaMalloc(&k.it, 4), "malloc") || !ok(cudaMalloc(&k.bad, 4), "malloc") ||
            !ok(cudaMalloc(&g_done[c], 4), "malloc") || !ok(cudaMalloc(&k.big, 2 * big_bytes), "malloc") ||
            !ok(cudaMalloc(&k.busy_buf, busy_bytes + 64), "malloc"))
            return 1;
        cudaMemset(k.it, 0, 4);
        cudaMemset(k.bad, 0, 4);
        cudaMemset(g_done[c], 0, 4);
        cudaMemset(k.big, 0, 2 * big_bytes);
        cudaMemset(k.busy_buf, 0, busy_bytes + 64);
        for (cudaEvent_t& e : k.sent) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        cudaDeviceSynchronize();
    }

    // bandwidth: idle, then with a memory-bound kernel on each card
    bandwidth(cd, big_bytes, 20, "peer copy, idle");
    cudaEvent_t busy_done[2];
    for (Card& c : cd) {
        cudaSetDevice(c.dev);
        cudaEventCreateWithFlags(&busy_done[c.dev], cudaEventDisableTiming);
        stream_read<<<512, 256, 0, c.busy_s>>>((const float4*) c.busy_buf, busy_bytes / 16, 1000, c.busy_buf + busy_bytes / 4);
        cudaEventRecord(busy_done[c.dev], c.busy_s);
    }
    bandwidth(cd, big_bytes, 20, "peer copy, cards busy");
    for (Card& c : cd) {   // the load must outlast the copies, or the line above measured idle cards
        cudaSetDevice(c.dev);
        if (cudaEventQuery(busy_done[c.dev]) != cudaErrorNotReady)
            std::printf("  (card %d's load ended before the copies did: the busy figures are partly idle)\n", c.dev);
        (void) cudaGetLastError();
    }
    for (Card& c : cd) {
        cudaSetDevice(c.dev);
        cudaStreamSynchronize(c.busy_s);
        cudaEventDestroy(busy_done[c.dev]);
    }

    // the exchange
    std::printf("\nper exchange over the base graph (us; kernels alone: base p50), %d exchanges a graph, %d replays\n", kIters, kReplays);
    std::printf("rows  KiB   base p50 | copy  p50    p95    p99 | store p50    p95    p99\n");
    for (int rows : {1, 2, 4, 8}) {
        Times t[3];
        for (Mech m : {Mech::Base, Mech::Copy, Mech::Store}) {
            cudaGraphExec_t exec;
            if (!capture(cd, m, rows, exec)) return 1;
            replay(cd, exec, 20, nullptr);   // warm
            std::vector<double> us;
            if (!replay(cd, exec, kReplays, &us)) return 1;
            t[(int) m] = pct(us);
            cudaGraphExecDestroy(exec);
            if (const uint32_t b = bad_sums(cd)) {
                std::printf("FAIL %s, %d rows: %u wrong sums\n", name(m), rows, b);
                ++fails;
            }
        }
        const double b = t[0].p50;
        std::printf("%4d %5.0f %9.2f | %9.2f %6.2f %6.2f | %9.2f %6.2f %6.2f\n", rows, rows * kWidth * 4 / 1024.0, b,
                    t[1].p50 - b, t[1].p95 - b, t[1].p99 - b, t[2].p50 - b, t[2].p95 - b, t[2].p99 - b);
    }

    // the long check: the store mechanism (the one without a copy to order), 8 rows
    {
        cudaGraphExec_t exec;
        if (!capture(cd, Mech::Store, kMaxRows, exec)) return 1;
        const auto t = std::chrono::steady_clock::now();
        if (!replay(cd, exec, kLongReplays, nullptr)) return 1;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        cudaGraphExecDestroy(exec);
        const uint32_t b = bad_sums(cd);
        std::printf("\nlong replay, store, %d rows: %d exchanges in %.2f s, %u wrong sums %s\n", kMaxRows, kLongReplays * kIters, s, b,
                    b ? "FAIL" : "ok");
        if (b) ++fails;
    }

    // both cards hold the same bits
    {
        const size_t n = (size_t) kMaxRows * kWidth;
        std::vector<float> h0(n), h1(n);
        cudaSetDevice(0);
        cudaMemcpy(h0.data(), cd[0].out, n * 4, cudaMemcpyDeviceToHost);
        cudaSetDevice(1);
        cudaMemcpy(h1.data(), cd[1].out, n * 4, cudaMemcpyDeviceToHost);
        const bool same = std::memcmp(h0.data(), h1.data(), n * 4) == 0;
        std::printf("the two cards' sums: %s\n", same ? "same bits ok" : "DIFFERENT FAIL");
        if (!same) ++fails;
    }
    ok(cudaGetLastError(), "last error");
    std::printf("p2p_exchange_test: %d failures\n", fails);
    return fails ? 1 : 0;
}
