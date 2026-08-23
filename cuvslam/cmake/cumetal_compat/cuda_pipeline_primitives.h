#pragma once

// CuMetal declares the legacy __pipeline_* async-copy helpers in cuda_pipeline.h as host-only
// inline functions, so calling them from device code fails to compile. Metal has no equivalent of
// the Ampere async-copy pipeline, so the copy is performed synchronously and the commit/wait
// helpers become no-ops -- which is a correct (merely slower) implementation, since a synchronous
// copy trivially satisfies any subsequent wait.

#include <cuda_runtime.h>

__host__ __device__ __forceinline__ void __pipeline_memcpy_async(void* destination, const void* source,
                                                                 size_t size_in_bytes) {
  char* destination_bytes = static_cast<char*>(destination);
  const char* source_bytes = static_cast<const char*>(source);
  for (size_t index = 0; index < size_in_bytes; ++index) {
    destination_bytes[index] = source_bytes[index];
  }
}

__host__ __device__ __forceinline__ void __pipeline_commit() {}

__host__ __device__ __forceinline__ void __pipeline_wait_prior(int) {}
