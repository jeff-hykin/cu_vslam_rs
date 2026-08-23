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

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

constexpr float gaussian_kernel[] = {0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f};

constexpr float dsp_kernel[] = {112.0f / 8418.0f,   913.0f / 8418.0f,  3047.0f / 8418.0f, 0,
                                -3047.0f / 8418.0f, -913.0f / 8418.0f, -112.0f / 8418.0f};

constexpr float box_blur_kernel[] = {
    0.25f,
    0.5f,
    0.25f,
};

__constant__ float dsp_kernel_const[7];
__constant__ float gauss_kernel_const[7];
__constant__ float box_blur_const[3];
static bool kernels_initialized = false;
static bool box_prefilter_initialized = false;

cudaError_t init_conv_kernels() {
  if (kernels_initialized) {
    return cudaSuccess;
  }
  cudaError_t error = cudaMemcpyToSymbol(dsp_kernel_const, dsp_kernel, 7 * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(gauss_kernel_const, gaussian_kernel, 7 * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  kernels_initialized = true;
  return cudaSuccess;
}

cudaError_t init_box_prefilter_kernels() {
  if (box_prefilter_initialized) {
    return cudaSuccess;
  }
  cudaError_t error = cudaMemcpyToSymbol(box_blur_const, box_blur_kernel, 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  box_prefilter_initialized = true;
  return cudaSuccess;
}

#define TILE_WIDTH 16
#define HALO_TILES 1
#define RESULT_TILES 8

#define TILE_HEIGHT 4

__global__ void conv_grad_x_kernel(cudaTextureObject_t src, uint2 size, float* dst, size_t dpitch) {
  int base_x = (blockIdx.x * RESULT_TILES - HALO_TILES) * TILE_WIDTH + threadIdx.x;
  int base_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

  __shared__ float shared_src[TILE_HEIGHT][(RESULT_TILES + 2 * HALO_TILES) * TILE_WIDTH];

#pragma unroll
  for (int i = 0; i < HALO_TILES + RESULT_TILES + HALO_TILES; i++) {
    shared_src[threadIdx.y][threadIdx.x + i * TILE_WIDTH] = tex2D<float>(src, base_x + i * TILE_WIDTH, base_y);
  }
  __syncthreads();
#pragma unroll

  for (int i = HALO_TILES; i < HALO_TILES + RESULT_TILES; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -3; j <= 3; j++) {
      sum += dsp_kernel_const[3 + j] * shared_src[threadIdx.y][threadIdx.x + i * TILE_WIDTH + j];
    }

    unsigned int out_x = base_x + i * TILE_WIDTH;
    if (out_x < size.x && base_y < size.y) {
      float v = out_x > 2 ? sum : 0;
      v = out_x < size.x - 3 ? v : 0;

      float* d_Dst = (float*)((char*)dst + base_y * dpitch) + out_x;
      *d_Dst = v;
    }
  }
}

#define COL_TILE_HEIGHT 4
__global__ void conv_grad_y_kernel(cudaTextureObject_t src, uint2 size, float* dst, size_t dpitch) {
  __shared__ float s_Data[TILE_WIDTH][(RESULT_TILES + 2 * HALO_TILES) * COL_TILE_HEIGHT];

  const int baseX = blockIdx.x * TILE_WIDTH + threadIdx.x;
  const int baseY = (blockIdx.y * RESULT_TILES - HALO_TILES) * COL_TILE_HEIGHT + threadIdx.y;

#pragma unroll

  for (int i = 0; i < HALO_TILES + RESULT_TILES + HALO_TILES; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COL_TILE_HEIGHT] = tex2D<float>(src, baseX, baseY + i * COL_TILE_HEIGHT);
  }
  __syncthreads();
#pragma unroll

  for (int i = HALO_TILES; i < HALO_TILES + RESULT_TILES; i++) {
    float sum = 0;
#pragma unroll

    for (int j = -3; j <= 3; j++) {
      sum += dsp_kernel_const[3 + j] * s_Data[threadIdx.x][threadIdx.y + i * COL_TILE_HEIGHT + j];
    }

    unsigned int out_y = baseY + i * COL_TILE_HEIGHT;
    if (out_y < size.y && baseX < size.x) {
      float v = out_y > 2 ? sum : 0;
      v = out_y < size.y - 3 ? v : 0;

      float* d_Dst = (float*)((char*)dst + out_y * dpitch) + baseX;
      *d_Dst = v;
    }
  }
}

__global__ void box_blur_x_kernel(cudaTextureObject_t src, uint2 size, float* dst, size_t dpitch) {
  int base_x = (blockIdx.x * RESULT_TILES - HALO_TILES) * TILE_WIDTH + threadIdx.x;
  unsigned int base_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

  __shared__ float shared_src[TILE_HEIGHT][(RESULT_TILES + 2 * HALO_TILES) * TILE_WIDTH];

#pragma unroll
  for (int i = 0; i < HALO_TILES + RESULT_TILES + HALO_TILES; i++) {
    shared_src[threadIdx.y][threadIdx.x + i * TILE_WIDTH] = tex2D<float>(src, base_x + i * TILE_WIDTH, base_y);
  }
  __syncthreads();
#pragma unroll

  for (int i = HALO_TILES; i < HALO_TILES + RESULT_TILES; i++) {
    float sum = 0;

#pragma unroll

    for (int j = -1; j <= 1; j++) {
      sum += box_blur_const[1 + j] * shared_src[threadIdx.y][threadIdx.x + i * TILE_WIDTH + j];
    }

    unsigned int out_x = base_x + i * TILE_WIDTH;  // = threadIdx.x + i * TILE_WIDTH == size.x - 1
    if (out_x < size.x && base_y < size.y) {
      float v = out_x > 0 ? sum : 0;
      v = out_x < size.x - 1 ? v : 0;

      float* d_Dst = (float*)((char*)dst + base_y * dpitch) + out_x;
      *d_Dst = v;
    }
  }
}

__global__ void box_blur_y_kernel(cudaTextureObject_t src, uint2 size, float* dst, size_t dpitch) {
  __shared__ float s_Data[TILE_WIDTH][(RESULT_TILES + 2 * HALO_TILES) * COL_TILE_HEIGHT];

  const int baseX = blockIdx.x * TILE_WIDTH + threadIdx.x;
  const int baseY = (blockIdx.y * RESULT_TILES - HALO_TILES) * COL_TILE_HEIGHT + threadIdx.y;

#pragma unroll

  for (int i = 0; i < HALO_TILES + RESULT_TILES + HALO_TILES; i++) {
    s_Data[threadIdx.x][threadIdx.y + i * COL_TILE_HEIGHT] = tex2D<float>(src, baseX, baseY + i * COL_TILE_HEIGHT);
  }
  __syncthreads();
#pragma unroll

  for (int i = HALO_TILES; i < HALO_TILES + RESULT_TILES; i++) {
    float sum = 0;
#pragma unroll

    for (int j = -1; j <= 1; j++) {
      sum += box_blur_const[1 + j] * s_Data[threadIdx.x][threadIdx.y + i * COL_TILE_HEIGHT + j];
    }

    unsigned int out_y = baseY + i * COL_TILE_HEIGHT;
    if (out_y < size.y && baseX < size.x) {
      float v = out_y > 0 ? sum : 0;
      v = out_y < size.y - 1 ? v : 0;

      float* d_Dst = (float*)((char*)dst + out_y * dpitch) + baseX;
      *d_Dst = (uint8_t)v;
    }
  }
}

__global__ void copy_border_kernel(cudaTextureObject_t src, uint2 size, float* dst, size_t dpitch, size_t border_size) {
  size_t idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x >= size.x || idx_y >= size.y) {
    return;
  }

  if (idx_x < border_size || idx_x >= size.x - border_size || idx_y < border_size || idx_y >= size.y - border_size) {
    float* d_Dst = (float*)((char*)dst + idx_y * dpitch) + idx_x;
    *d_Dst = tex2D<float>(src, idx_x, idx_y);
    ;
  }
}

cudaError_t conv_grad_x(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s) {
  dim3 threads(TILE_WIDTH, TILE_HEIGHT);
  size_t num_blocks_x = (srcSize.x + TILE_WIDTH * RESULT_TILES - 1) / (TILE_WIDTH * RESULT_TILES);
  size_t num_blocks_y = (srcSize.y + TILE_HEIGHT - 1) / TILE_HEIGHT;

  dim3 blocks(num_blocks_x, num_blocks_y);

  conv_grad_x_kernel<<<blocks, threads, 0, s>>>(src, srcSize, dst, dpitch);
  return cudaGetLastError();
}

cudaError_t conv_grad_y(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s) {
  dim3 threads(TILE_WIDTH, COL_TILE_HEIGHT);
  size_t num_blocks_x = (srcSize.x + TILE_WIDTH - 1) / TILE_WIDTH;
  size_t num_blocks_y = (srcSize.y + COL_TILE_HEIGHT * RESULT_TILES - 1) / (COL_TILE_HEIGHT * RESULT_TILES);
  dim3 blocks(num_blocks_x, num_blocks_y);

  conv_grad_y_kernel<<<blocks, threads, 0, s>>>(src, srcSize, dst, dpitch);
  return cudaGetLastError();
}

cudaError_t box_blur_x(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s) {
  dim3 threads(TILE_WIDTH, TILE_HEIGHT);
  size_t num_blocks_x = (srcSize.x + TILE_WIDTH * RESULT_TILES - 1) / (TILE_WIDTH * RESULT_TILES);
  size_t num_blocks_y = (srcSize.y + TILE_HEIGHT - 1) / TILE_HEIGHT;

  dim3 blocks(num_blocks_x, num_blocks_y);

  box_blur_x_kernel<<<blocks, threads, 0, s>>>(src, srcSize, dst, dpitch);
  return cudaGetLastError();
}

cudaError_t box_blur_y(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, cudaStream_t s) {
  dim3 threads(TILE_WIDTH, COL_TILE_HEIGHT);
  size_t num_blocks_x = (srcSize.x + TILE_WIDTH - 1) / TILE_WIDTH;
  size_t num_blocks_y = (srcSize.y + COL_TILE_HEIGHT * RESULT_TILES - 1) / (COL_TILE_HEIGHT * RESULT_TILES);
  dim3 blocks(num_blocks_x, num_blocks_y);

  box_blur_y_kernel<<<blocks, threads, 0, s>>>(src, srcSize, dst, dpitch);
  return cudaGetLastError();
}

cudaError_t copy_border(cudaTextureObject_t src, uint2 srcSize, float* dst, size_t dpitch, size_t border_size,
                        cudaStream_t s) {
  dim3 threads(SQRT_MAX_THREADS, SQRT_MAX_THREADS);
  dim3 blocks((srcSize.x + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS,
              (srcSize.y + SQRT_MAX_THREADS - 1) / SQRT_MAX_THREADS);

  copy_border_kernel<<<blocks, threads, 0, s>>>(src, srcSize, dst, dpitch, border_size);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
