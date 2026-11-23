// include/basalt/kernels/route_lossy.hpp - the opt-in routing changes that trade accuracy for fewer missed experts
// (--route-tail-skip, --route-prior; basalt/model/route_flags.hpp): which experts the GPUs hold.
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

}  // namespace basalt::kernels
