// include/basalt/kernels/prefetch_l2.cuh - bytes into L2 ahead of their reads.  Device code only: include from a .cu.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// The bytes [p, p + n) into L2 from the calling thread: one prefetch a 128-byte line (sm_86 has no bulk prefetch, which
/// takes a whole run in one instruction).  Spread a long run over the threads of a block, a line or a few each.
__device__ __forceinline__ void prefetch_l2(const void* p, uint64_t n) {
    const uintptr_t end = (uintptr_t) p + n;
    for (uintptr_t a = (uintptr_t) p & ~(uintptr_t) 127; a < end; a += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
}

}  // namespace basalt::kernels
