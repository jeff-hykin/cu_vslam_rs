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

#define KERNEL_SIZE 7

namespace cuvslam::cuda {

constexpr float gaussian_kernel[] = {0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f};

__constant__ float gaussian_kernel_const[KERNEL_SIZE * KERNEL_SIZE];
__constant__ float gaussian_kernel_1d_const[KERNEL_SIZE];
static bool gaussian_kernel_initialized = false;

cudaError_t init_gauss_square_kernel() {
  if (gaussian_kernel_initialized) {
    return cudaSuccess;
  }

  float gauss_kernel_1d_array[KERNEL_SIZE];
  float gauss_kernel_array[KERNEL_SIZE * KERNEL_SIZE];

  for (int i = 0; i < KERNEL_SIZE; i++) {
    gauss_kernel_1d_array[i] = gaussian_kernel[i];
    for (int j = 0; j < KERNEL_SIZE; j++) {
      int idx = i * KERNEL_SIZE + j;
      gauss_kernel_array[idx] = gaussian_kernel[i] * gaussian_kernel[j];
    }
  }

  cudaError_t error = cudaMemcpyToSymbol(gaussian_kernel_const, gauss_kernel_array,
                                         KERNEL_SIZE * KERNEL_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(gaussian_kernel_1d_const, gauss_kernel_1d_array, KERNEL_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  gaussian_kernel_initialized = true;
  return cudaSuccess;
}

__device__ __forceinline__ float GFTTMeasure(const float gxx, const float gxy, const float gyy) noexcept {
  const float D = __fsqrt_rn((gxx - gyy) * (gxx - gyy) + 4.f * gxy * gxy);
  const float T = gxx + gyy;
  const float eMin = (T - D) * 0.5f;
  return __logf(eMin + 1.f);
}

#undef BLOCK_HEIGHT
#define BLOCK_HEIGHT 32
#define N_TILE_HEIGHT (BLOCK_HEIGHT - (KERNEL_SIZE - 1))

#undef BLOCK_WIDTH
#define BLOCK_WIDTH 16
#define N_TILE_WIDTH BLOCK_WIDTH

#define LOAD_ITERATIONS                                                                       \
  (((BLOCK_HEIGHT * (N_TILE_WIDTH + (KERNEL_SIZE - 1))) + (BLOCK_WIDTH * BLOCK_HEIGHT - 1)) / \
   (BLOCK_WIDTH * BLOCK_HEIGHT))

__launch_bounds__(BLOCK_WIDTH* BLOCK_HEIGHT) __global__
    void gftt_values_kernel(cudaTextureObject_t gradX, cudaTextureObject_t gradY, float* values, size_t v_pitch,
                            uint2 image_size) {
  // indexing variables
  int out_row = blockIdx.y * N_TILE_HEIGHT + threadIdx.y;
  int out_col = blockIdx.x * N_TILE_WIDTH + threadIdx.x;

  __shared__ float2 tile_xy[BLOCK_HEIGHT][N_TILE_WIDTH + (KERNEL_SIZE - 1)];
  __shared__ float2 xx_yy_convolved_h[BLOCK_HEIGHT][BLOCK_WIDTH];
  __shared__ float xy_convolved_h[BLOCK_HEIGHT][BLOCK_WIDTH];

  int flat_id = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
  float2 tile_xy_t[LOAD_ITERATIONS];
  for (int i = 0; i < LOAD_ITERATIONS; ++i, flat_id += (BLOCK_WIDTH * BLOCK_HEIGHT)) {
    int load_row = flat_id / (N_TILE_WIDTH + (KERNEL_SIZE - 1));
    int load_col = flat_id - load_row * (N_TILE_WIDTH + (KERNEL_SIZE - 1));

    if ((i < LOAD_ITERATIONS - 1) || (load_row < BLOCK_HEIGHT)) {
      int in_row = load_row + blockIdx.y * N_TILE_HEIGHT - KERNEL_SIZE / 2;

      in_row = in_row >= 0 ? in_row : -in_row - 1;
      in_row = in_row < image_size.y ? in_row : 2 * image_size.y - in_row - 1;

      int in_col = load_col + blockIdx.x * N_TILE_WIDTH - KERNEL_SIZE / 2;

      in_col = in_col >= 0 ? in_col : -in_col - 1;
      in_col = in_col < image_size.x ? in_col : 2 * image_size.x - in_col - 1;

      tile_xy_t[i].x = tex2D<float>(gradX, in_col, in_row);
      tile_xy_t[i].y = tex2D<float>(gradY, in_col, in_row);
    }
  }
  flat_id = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
  for (int i = 0; i < LOAD_ITERATIONS; ++i, flat_id += (BLOCK_WIDTH * BLOCK_HEIGHT)) {
    int load_row = flat_id / (N_TILE_WIDTH + (KERNEL_SIZE - 1));
    int load_col = flat_id - load_row * (N_TILE_WIDTH + (KERNEL_SIZE - 1));

    if ((i < LOAD_ITERATIONS - 1) || (load_row < BLOCK_HEIGHT)) {
      tile_xy[load_row][load_col] = tile_xy_t[i];
    }
  }
  __syncthreads();

  // thread boundary check for calculation
  if (threadIdx.x < N_TILE_WIDTH && out_col < image_size.x) {
    float xx_h = 0;
    float xy_h = 0;
    float yy_h = 0;

    for (int j = 0; j < KERNEL_SIZE; ++j) {
      float2 gxy = tile_xy[threadIdx.y][threadIdx.x + j];
      float gx = gxy.x;
      float gy = gxy.y;
      xx_h += gaussian_kernel_1d_const[j] * gx * gx;
      xy_h += gaussian_kernel_1d_const[j] * gx * gy;
      yy_h += gaussian_kernel_1d_const[j] * gy * gy;
    }

    // write result of horizontal 1d convolution
    xx_yy_convolved_h[threadIdx.y][threadIdx.x] = make_float2(xx_h, yy_h);
    xy_convolved_h[threadIdx.y][threadIdx.x] = xy_h;
  }

  __syncthreads();

  // thread boundary check for calculation
  if (threadIdx.y < N_TILE_HEIGHT && threadIdx.x < N_TILE_WIDTH && out_row < image_size.y && out_col < image_size.x) {
    float xx_data = 0;
    float xy_data = 0;
    float yy_data = 0;

    for (int i = 0; i < KERNEL_SIZE; ++i) {
      float2 xx_yy_convolved = xx_yy_convolved_h[threadIdx.y + i][threadIdx.x];
      xx_data += gaussian_kernel_1d_const[i] * xx_yy_convolved.x;
      xy_data += gaussian_kernel_1d_const[i] * xy_convolved_h[threadIdx.y + i][threadIdx.x];
      yy_data += gaussian_kernel_1d_const[i] * xx_yy_convolved.y;
    }

    // write result
    float* v = (float*)((char*)values + out_row * v_pitch) + out_col;
    *v = GFTTMeasure(xx_data, xy_data, yy_data);
  }
}

cudaError_t gftt_values(cudaTextureObject_t gradX, cudaTextureObject_t gradY, float* values, size_t v_pitch,
                        uint2 image_size, cudaStream_t stream) {
  dim3 threads(BLOCK_WIDTH, BLOCK_HEIGHT);
  size_t num_blocks_x = (image_size.x + N_TILE_WIDTH - 1) / N_TILE_WIDTH;
  size_t num_blocks_y = (image_size.y + N_TILE_HEIGHT - 1) / N_TILE_HEIGHT;
  dim3 blocks(num_blocks_x, num_blocks_y);
  gftt_values_kernel<<<blocks, threads, 0, stream>>>(gradX, gradY, values, v_pitch, image_size);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
