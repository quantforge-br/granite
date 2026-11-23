// tests/kernels/route_lossy_test.cpp - the residency check over several tables (basalt/kernels/route_lossy.hpp) on the
// CPU, against hand-made tables.
#include "basalt/kernels/route_lossy.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::kernels;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    // residency: any table holds it
    {
        std::vector<int32_t> a(8, -1), b(8, -1);
        a[2] = 5;
        b[6] = 0;
        ResidentTabs h;
        h.add(a.data());
        h.add(nullptr);
        h.add(b.data());
        check(h.n == 2, "a null table is not added");
        check(h.held(2) && h.held(6) && !h.held(3), "held when any table has a slot");
        ResidentTabs none;
        check(!none.held(2), "no table holds nothing");
    }
    std::printf("route_lossy_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
