#pragma once

// Force-included ahead of every .cu when building the CuMetal backend, to paper over the few
// places where the CUDA toolkit provides something libc++/CuMetal does not.

#include <cuda_runtime.h>

// The CUDA headers define the short vector-component spellings; libc++ does not.
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

// Apple's assert() expands to __assert_rtn, which is __host__, so any assert inside a __global__ or
// __device__ function is rejected -- in both compilation passes, since clang parses device bodies
// during the host pass too. CUDA solves this by shipping a __device__ assert that traps; Metal has
// no equivalent trap-and-report path, so the device overload is a no-op and device-side assertions
// are dropped. Declaring it as a __device__ overload rather than redefining the assert macro keeps
// host-side assertions in .cu files working normally.
#include <cassert>

__device__ inline void __assert_rtn(const char*, const char*, int, const char*) {}
