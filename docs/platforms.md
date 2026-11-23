# Platforms

Linux x64 with NVIDIA RTX 30-series Ampere cards (sm_86). The kernels are built for compute capability 8.6 only: CMake refuses
any other `CMAKE_CUDA_ARCHITECTURES`, and the engine refuses a card that is not sm_86 at start ([GPUs](configuration/gpus.md)).
CI builds the engine and runs the CPU tests; the GPU tests and the quality gate need a card.

| | Linux |
| --- | --- |
| OS | x86-64, AVX2 (AVX-512 used when present) |
| Compiler, build | GCC 13+, CMake 3.24+ |
| CUDA, Python | 13.0 or newer, 3.10+ |

## Driver

CUDA 13 needs an NVIDIA driver of the R580 branch or newer (580.95.05 for CUDA 13.0.2). Upstream found that an older 615
build made every reply loop endlessly and that 615.71.09 fixed it (#514): if output loops or looks garbled, update the driver
before anything else.

## Platform code

OS code lives only in `src/platform/` and `include/granite/platform/`, and CMake `if(WIN32)` blocks. Anywhere else there is no `_WIN32`,
`_MSC_VER`, `__linux__`, `<windows.h>`, `<sys/mman.h>`, `<unistd.h>` or raw OS call (mmap, madvise, pread, pthread/affinity,
fork/exec/pipe, dlopen, /proc or /sys reads). `tools/check_platform_boundary.sh` enforces it. Each seam has a Linux and a Windows body,
the Linux body being the code you would have written anyway, and a test in `tests/platform/` that runs on both. Paths are UTF-8 and go
through the platform file calls.

The Windows bodies are kept from Basalt, and CI still compiles a Windows build and runs its CPU tests there. granite is run and
tested on Linux only.
