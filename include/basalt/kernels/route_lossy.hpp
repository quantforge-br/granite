// include/basalt/kernels/route_lossy.hpp - the opt-in routing changes that trade accuracy for fewer missed experts
// (--route-tail-skip, --route-prior; basalt/model/route_flags.hpp): which experts the GPUs hold, and the tail skip's rule.
//
// Both change the output, so a model applies them to its decode windows only.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define ROUTE_LOSSY_HD __host__ __device__
#else
#define ROUTE_LOSSY_HD
#endif

namespace basalt::kernels {

/// One layer's residency tables, one per GPU tier ([n_expert]: the expert's slot, or -1). A GPU holds an expert
/// when any table has a slot for it.
struct ResidentTabs {
    static constexpr int kMax = 4;
    const int32_t* t[kMax] = {};
    int n = 0;
    void add(const int32_t* tab) { if (tab != nullptr && n < kMax) t[n++] = tab; }
    ROUTE_LOSSY_HD bool held(int32_t e) const {
        bool in = false;
        for (int q = 0; q < n; ++q) in |= t[q][e] >= 0;
        return in;
    }
};

/// The rank (0-based) of entry i by router weight among its token's k entries: the heavier first, a tie to the lower slot.
ROUTE_LOSSY_HD inline int route_weight_rank(const float* w, int i, int k) {
    const float wi = w[i];
    const int base = i - i % k;
    int rank = 0;
    for (int j = 0; j < k; ++j) {
        const float wj = w[base + j];
        rank += wj > wi || (wj == wi && base + j < i);
    }
    return rank;
}

/// Whether entry i is dropped: its expert `miss`es every GPU and every entry of the window naming that expert is in its
/// token's tail (`tail[j]`: rank R or lower).
ROUTE_LOSSY_HD inline bool route_tail_drop(const int32_t* ids, const uint8_t* tail, int n, int i, bool miss) {
    bool drop = miss;
    for (int j = 0; drop && j < n; ++j)
        if (ids[j] == ids[i] && !tail[j]) drop = false;
    return drop;
}

}  // namespace basalt::kernels
