/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include <stdio.h>

#include <cuda_runtime.h>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

constexpr float scaler_kernel[] = {
    1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f, 1.f / 16.f,
};

#define KERNEL_SIZE 5
__constant__ float scaler_kernel_const[KERNEL_SIZE * KERNEL_SIZE];
static bool scaler_initialized = false;

cudaError_t init_scaler() {
  if (scaler_initialized) {
    return cudaSuccess;
  }

  float scaler_kernel_array[KERNEL_SIZE * KERNEL_SIZE];

  for (int i = 0; i < KERNEL_SIZE; i++) {
    for (int j = 0; j < KERNEL_SIZE; j++) {
      int idx = i * KERNEL_SIZE + j;
      scaler_kernel_array[idx] = scaler_kernel[i] * scaler_kernel[j];
    }
  }

  cudaError_t error = cudaMemcpyToSymbol(scaler_kernel_const, scaler_kernel_array,
                                         KERNEL_SIZE * KERNEL_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  scaler_initialized = true;
  return cudaSuccess;
}

__global__ void gaussian_scaling_kernel(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch,
                                        uint2 dstSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int x_src = 2 * x;
  int y_src = 2 * y;

  if (x >= dstSize.x || y >= dstSize.y) {
    return;
  }

  float accum = 0;
#pragma unroll
  for (int k = 0; k < KERNEL_SIZE; k++) {
#pragma unroll
    for (int l = 0; l < KERNEL_SIZE; l++) {
      size_t kernel_idx = k * KERNEL_SIZE + l;
      int x_c = x_src - 2 + l;
      int y_c = y_src - 2 + k;

      if (x_c < 0 && y_c < 0) {
        continue;
      }
      if (x_c < 0 && y_c >= srcSize.y) {
        continue;
      }
      if (x_c >= srcSize.x && y_c < 0) {
        continue;
      }
      if (x_c >= srcSize.x && y_c >= srcSize.y) {
        continue;
      }

      x_c = x_c < 0 ? -x_c - 1 : x_c;
      x_c = x_c >= srcSize.x ? 2 * (int)srcSize.x - x_c - 1 : x_c;

      y_c = y_c < 0 ? -y_c - 1 : y_c;
      y_c = y_c >= srcSize.y ? 2 * (int)srcSize.y - y_c - 1 : y_c;

      accum += scaler_kernel_const[kernel_idx] * tex2D<float>(src, x_c, y_c);
    }
  }

  float* v = (float*)((char*)dst + y * dpitch) + x;
  *v = accum;
}

cudaError_t gaussian_scaling(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, uint2 dstSize,
                             cudaStream_t stream) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((dstSize.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS,
              (dstSize.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);
  gaussian_scaling_kernel<<<blocks, threads, 0, stream>>>(src, srcSize, dst, dpitch, dstSize);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
