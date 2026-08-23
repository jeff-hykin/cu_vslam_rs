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

#include <assert.h>
#include <stdio.h>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

#define MAX_POINT_TRACK_ITERATIONS 11
#define LK_CONVERGENCE_THRESHOLD 0.12f

#define ELEMS_PER_THREAD ((PATCH_DIM * PATCH_DIM + 31) / 32)

namespace cuvslam::cuda {

constexpr float gauss_coeffs[9] = {0.0135196f, 0.0476622f, 0.11723f,   0.201168f, 0.240841f,
                                   0.201168f,  0.11723f,   0.0476622f, 0.0135196f};

__device__ float gauss_coeffs_const[PATCH_DIM * PATCH_DIM];
static bool gauss_coeffs_initialized = false;

cudaError_t init_gauss_coeffs() {
  if (gauss_coeffs_initialized) {
    return cudaSuccess;
  }

  float gauss_coeffs_array[PATCH_DIM * PATCH_DIM];

  for (int i = 0; i < PATCH_DIM; i++) {
    for (int j = 0; j < PATCH_DIM; j++) {
      int idx = i * PATCH_DIM + j;
      gauss_coeffs_array[idx] = gauss_coeffs[i] * gauss_coeffs[j];
    }
  }

  cudaError_t error = cudaMemcpyToSymbol(gauss_coeffs_const, gauss_coeffs_array, PATCH_DIM * PATCH_DIM * sizeof(float),
                                         0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  gauss_coeffs_initialized = true;
  return cudaSuccess;
}

#define PATCH_HALF 4.5f

__device__ __forceinline__ float reduce(float x) {
  int lane_mask = 1;
#pragma unroll
  for (int i = 0; i < 5; ++i, lane_mask <<= 1) {
    x += __shfl_xor_sync(0xffffffff, x, lane_mask);
  }
  return x;
}

__launch_bounds__(32) __global__
    void lk_track_kernel(Pyramid prevFrameGradXPyramid, Pyramid prevFrameGradYPyramid, Pyramid prevFrameImagePyramid,
                         Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks) {
  const int lane_id = threadIdx.x & 31;
  const bool master = (lane_id == 0);
  const int x = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (x >= num_tracks) {
    return;
  }

  float2& track = track_data[x].track;
  const float2& offset = track_data[x].offset;
  float* info_mat = track_data[x].info;
  bool& status = track_data[x].track_status;
  float search_radius_px = track_data[x].search_radius_px;
  float ncc_threshold = track_data[x].ncc_threshold;

  if (search_radius_px < 0.f) {
    if (master) {
      status = false;
    }
    return;
  }

  // logic here comes from camera::GetDefaultObservationInfoUV()
  info_mat[0] = 1.f / 9.f;
  info_mat[1] = 0.f;
  info_mat[2] = 0.f;
  info_mat[3] = 1.f / 9.f;

  const int levels = prevFrameGradYPyramid.levels;

  assert(levels <= prevFrameGradXPyramid.levels);

  const int topLevelIndex = levels - 1;  // smallest image

  bool is_point_in_prev_frame = track.x >= 0 && track.x < (float)prevFrameImagePyramid.level_sizes[0].width &&
                                track.y >= 0 && track.y < (float)prevFrameImagePyramid.level_sizes[0].height;

  bool is_point_in_cur_frame =
      track.x + offset.x >= 0 && track.x + offset.x < (float)currentFrameImagePyramid.level_sizes[0].width &&
      track.y + offset.y >= 0 && track.y + offset.y < (float)currentFrameImagePyramid.level_sizes[0].height;

  if (!is_point_in_prev_frame || !is_point_in_cur_frame) {
    if (master) status = false;
    return;
  }

  const float scale = __fdividef(1.f, static_cast<float>(1 << topLevelIndex));  // scale = pow(0.5, level);
  // account for the fact that the image content is scaled around (0.5 0.5)
  const float shift = 0.5f;

  float2 shifted_xy = {(track.x - shift) * scale + shift, (track.y - shift) * scale + shift};

  float2 shifted_curr_xy = {(track.x + offset.x - shift) * scale + shift, (track.y + offset.y - shift) * scale + shift};

  bool isConverged = false;
  const float windowSizeRcp = (1.f / (float)(PATCH_DIM * PATCH_DIM));

  float ix_val[ELEMS_PER_THREAD];
  float iy_val[ELEMS_PER_THREAD];
  float weights[ELEMS_PER_THREAD];
  bool last_elem_valid;
  bool ncc_mask[ELEMS_PER_THREAD];
  int flat_id = lane_id;
  const int start_ncc = (PATCH_DIM - NCC_DIM) / 2;
#pragma unroll
  for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id, flat_id += 32) {
    int iy = flat_id / PATCH_DIM;
    int ix = flat_id - iy * PATCH_DIM;
    if (elem_id == ELEMS_PER_THREAD - 1) {
      last_elem_valid = (iy < PATCH_DIM);
    }
    ncc_mask[elem_id] =
        (ix >= start_ncc) && (ix < (start_ncc + NCC_DIM)) && (iy >= start_ncc) && (iy < (start_ncc + NCC_DIM));
    ix_val[elem_id] = (float)ix;
    iy_val[elem_id] = (float)iy;
    if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
      weights[elem_id] = gauss_coeffs_const[flat_id];
    }
  }

  for (int i = 0; i < levels; ++i) {
    assert(topLevelIndex >= i);
    const int level = topLevelIndex - i;

    cudaTextureObject_t grad_x = prevFrameGradXPyramid.level_tex[level];
    cudaTextureObject_t grad_y = prevFrameGradYPyramid.level_tex[level];
    cudaTextureObject_t prev_image = prevFrameImagePyramid.level_tex[level];
    cudaTextureObject_t curr_image = currentFrameImagePyramid.level_tex[level];
    Size level_size = currentFrameImagePyramid.level_sizes[level];

    float xmin = shifted_xy.x + (0.5f - PATCH_HALF);
    float xmax = shifted_xy.x + (0.5f - PATCH_HALF + (float)PATCH_DIM);
    float ymin = shifted_xy.y + (0.5f - PATCH_HALF);
    float ymax = shifted_xy.y + (0.5f - PATCH_HALF + (float)PATCH_DIM);

    bool good_patch = true;

    if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
      good_patch = false;  // the patch bound box is out of the image
    }

    if (good_patch) {
      float gradPatchX[ELEMS_PER_THREAD];
      float gradPatchY[ELEMS_PER_THREAD];
      float imagePatch1[ELEMS_PER_THREAD];

      float grad_x_sum = 0.f;
      float grad_y_sum = 0.f;
      float prev_patch_sum = 0.f;

#pragma unroll
      for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
        if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
          float gx = tex2D<float>(grad_x, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);
          float gy = tex2D<float>(grad_y, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);
          float prev_value = tex2D<float>(prev_image, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);

          grad_x_sum += gx;
          grad_y_sum += gy;
          prev_patch_sum += prev_value;

          gradPatchX[elem_id] = gx;
          gradPatchY[elem_id] = gy;
          imagePatch1[elem_id] = prev_value;
        }
      }
      grad_x_sum = reduce(grad_x_sum);
      grad_y_sum = reduce(grad_y_sum);
      prev_patch_sum = reduce(prev_patch_sum);

      float gxx = 1.f / 32.f;
      float gxy = 0.f;
      float gyy = 1.f / 32.f;

#pragma unroll
      for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
        if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
          float gx = gradPatchX[elem_id] - grad_x_sum * windowSizeRcp;
          float gy = gradPatchY[elem_id] - grad_y_sum * windowSizeRcp;

          gradPatchX[elem_id] = gx;
          gradPatchY[elem_id] = gy;

          gxx += gx * gx * weights[elem_id];
          gxy += gx * gy * weights[elem_id];
          gyy += gy * gy * weights[elem_id];
        }
      }
      gxx = reduce(gxx);
      gxy = reduce(gxy);
      gyy = reduce(gyy);

      const float determinant = gxx * gyy - gxy * gxy;

      if (determinant < kFloatEpsilon) {
        if (master) {
          status = false;
        }
        return;
      }

      float imagePatch2[ELEMS_PER_THREAD];
      float curr_patch_sum = 0.f;

      const float2 xy2Save = shifted_curr_xy;

      const float determinantRcp = __fdividef(1.0f, determinant);
      for (int iterations = 0; iterations < MAX_POINT_TRACK_ITERATIONS; iterations++) {
        xmin = shifted_curr_xy.x + (0.5f - PATCH_HALF);
        xmax = shifted_curr_xy.x + (0.5f - PATCH_HALF + (float)PATCH_DIM);
        ymin = shifted_curr_xy.y + (0.5f - PATCH_HALF);
        ymax = shifted_curr_xy.y + (0.5f - PATCH_HALF + (float)PATCH_DIM);

        if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
          good_patch = false;  // the patch bound box is out of the image
        }

        if (!good_patch) {
          break;
        }

        curr_patch_sum = 0.f;

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
            float curr_value = tex2D<float>(curr_image, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);
            curr_patch_sum += curr_value;

            imagePatch2[elem_id] = curr_value;
          }
        }
        curr_patch_sum = reduce(curr_patch_sum);

        float delta = (curr_patch_sum - prev_patch_sum) * windowSizeRcp;

        float2 e = {0.f, 0.f};

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
            float curr_value = imagePatch2[elem_id];
            float prev_value = imagePatch1[elem_id];
            float diff = curr_value - prev_value - delta;

            e.x += weights[elem_id] * diff * gradPatchX[elem_id];
            e.y += weights[elem_id] * diff * gradPatchY[elem_id];
          }
        }
        e.x = reduce(e.x);
        e.y = reduce(e.y);

        const float2 v = {(gyy * e.x - gxy * e.y) * determinantRcp, (gxx * e.y - gxy * e.x) * determinantRcp};

        const float v_norm_squared = v.x * v.x + v.y * v.y;

        isConverged = (v_norm_squared < (float)(LK_CONVERGENCE_THRESHOLD * LK_CONVERGENCE_THRESHOLD));

        if (isConverged) {
          // TODO: investigate info matrix calculation,
          // now it's not correct and requires to use default info matrix
          // if (lane_id < 4) {
          //     info_mat[lane_id] = (lane_id == 0) ? gxx : ((lane_id == 3) ? gyy : gxy);
          // }
          break;
        }

        shifted_curr_xy.x += v.x;
        shifted_curr_xy.y += v.y;
      }

      if (isConverged) {
        const float level_radius = search_radius_px / (float)(1 << level);
        const float dx = shifted_curr_xy.x - xy2Save.x;
        const float dy = shifted_curr_xy.y - xy2Save.y;
        if (dx * dx + dy * dy > level_radius * level_radius) {
          isConverged = false;
        }
      }

      if (isConverged) {
        float sum_prev_squared = 0.f;
        float sum_curr_squared = 0.f;
        float sum_prev_curr = 0.f;
        float sum_prev = 0.f;
        float sum_curr = 0.f;

        const float sizeRcp = 1.0f / (float)(NCC_DIM * NCC_DIM);

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if (ncc_mask[elem_id]) {
            float curr_value = imagePatch2[elem_id];
            float prev_value = imagePatch1[elem_id];

            sum_prev += prev_value;
            sum_curr += curr_value;

            sum_prev_squared += prev_value * prev_value;
            sum_curr_squared += curr_value * curr_value;
            sum_prev_curr += curr_value * prev_value;
          }
        }
        sum_prev = reduce(sum_prev);
        sum_curr = reduce(sum_curr);
        sum_prev_squared = reduce(sum_prev_squared);
        sum_curr_squared = reduce(sum_curr_squared);
        sum_prev_curr = reduce(sum_prev_curr);

        const float a1 = sum_prev * sizeRcp;
        const float a2 = sum_curr * sizeRcp;

        const float d1 = sum_prev_squared * sizeRcp - a1 * a1;
        const float d2 = sum_curr_squared * sizeRcp - a2 * a2;
        const float d12 = sum_prev_curr * sizeRcp - a1 * a2;

        const float d1d2 = d1 * d2;
        float ncc = d1d2 < kFloatEpsilon ? 0.f : d12 * __frsqrt_rn(d1d2);

        isConverged = ncc > ncc_threshold;
      }

      if (!isConverged && level > 0) {
        shifted_curr_xy.x = xy2Save.x;
        shifted_curr_xy.y = xy2Save.y;
      }

      isConverged = (level == 0) && isConverged;
    }  // can getGradXYPatches

    if (i != levels - 1) {
      assert(level > 0);

      shifted_xy.x = shifted_xy.x * 2 - 0.5f;
      shifted_xy.y = shifted_xy.y * 2 - 0.5f;

      shifted_curr_xy.x = shifted_curr_xy.x * 2 - 0.5f;
      shifted_curr_xy.y = shifted_curr_xy.y * 2 - 0.5f;
    }
  }

  if (master) {
    if (isConverged) {
      track = shifted_curr_xy;
    }

    status = isConverged;
  }
}

__launch_bounds__(32) __global__
    void lk_track_horizontal_kernel(Pyramid prevFrameGradXPyramid, Pyramid prevFrameImagePyramid,
                                    Pyramid currentFrameImagePyramid, TrackData* __restrict__ track_data,
                                    int num_tracks) {
  const int lane_id = threadIdx.x & 31;
  const bool master = (lane_id == 0);
  const int x = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (x >= num_tracks) {
    return;
  }

  float2& track = track_data[x].track;
  const float2& offset = track_data[x].offset;
  float* info_mat = track_data[x].info;
  bool& status = track_data[x].track_status;
  float search_radius_px = track_data[x].search_radius_px;
  float ncc_threshold = track_data[x].ncc_threshold;

  if (search_radius_px < 0.f) {
    if (master) status = false;
    return;
  }

  // logic here comes from camera::GetDefaultObservationInfoUV()
  info_mat[0] = 1.f / 9.f;
  info_mat[1] = 0.f;
  info_mat[2] = 0.f;
  info_mat[3] = 1.f / 9.f;

  const int levels = prevFrameGradXPyramid.levels;

  const int topLevelIndex = levels - 1;  // smallest image

  bool is_point_in_prev_frame = track.x >= 0 && track.x < (float)prevFrameImagePyramid.level_sizes[0].width &&
                                track.y >= 0 && track.y < (float)prevFrameImagePyramid.level_sizes[0].height;

  bool is_point_in_cur_frame =
      track.x + offset.x >= 0 && track.x + offset.x < (float)currentFrameImagePyramid.level_sizes[0].width &&
      track.y + offset.y >= 0 && track.y + offset.y < (float)currentFrameImagePyramid.level_sizes[0].height;

  if (!is_point_in_prev_frame || !is_point_in_cur_frame) {
    if (master) status = false;
    return;
  }

  const float scale = __fdividef(1.f, static_cast<float>(1 << topLevelIndex));  // scale = pow(0.5, level);
  // account for the fact that the image content is scaled around (0.5 0.5)
  const float shift = 0.5f;

  float2 shifted_xy = {(track.x - shift) * scale + shift, (track.y - shift) * scale + shift};

  float2 shifted_curr_xy = {(track.x + offset.x - shift) * scale + shift, (track.y + offset.y - shift) * scale + shift};

  bool isConverged = false;
  const float windowSizeRcp = (1.f / (float)(PATCH_DIM * PATCH_DIM));

  float ix_val[ELEMS_PER_THREAD];
  float iy_val[ELEMS_PER_THREAD];
  float weights[ELEMS_PER_THREAD];
  bool last_elem_valid;
  bool ncc_mask[ELEMS_PER_THREAD];
  int flat_id = lane_id;
  const int start_ncc = (PATCH_DIM - NCC_DIM) / 2;
#pragma unroll
  for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id, flat_id += 32) {
    int iy = flat_id / PATCH_DIM;
    int ix = flat_id - iy * PATCH_DIM;
    if (elem_id == ELEMS_PER_THREAD - 1) {
      last_elem_valid = (iy < PATCH_DIM);
    }
    ncc_mask[elem_id] =
        (ix >= start_ncc) && (ix < (start_ncc + NCC_DIM)) && (iy >= start_ncc) && (iy < (start_ncc + NCC_DIM));
    ix_val[elem_id] = (float)ix;
    iy_val[elem_id] = (float)iy;
    if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
      weights[elem_id] = gauss_coeffs_const[flat_id];
    }
  }

  for (int i = 0; i < levels; ++i) {
    assert(topLevelIndex >= i);
    const int level = topLevelIndex - i;

    cudaTextureObject_t grad_x = prevFrameGradXPyramid.level_tex[level];
    cudaTextureObject_t prev_image = prevFrameImagePyramid.level_tex[level];
    cudaTextureObject_t curr_image = currentFrameImagePyramid.level_tex[level];
    Size level_size = currentFrameImagePyramid.level_sizes[level];

    float xmin = shifted_xy.x + (0.5f - PATCH_HALF);
    float xmax = shifted_xy.x + (0.5f - PATCH_HALF + (float)PATCH_DIM);
    float ymin = shifted_xy.y + (0.5f - PATCH_HALF);
    float ymax = shifted_xy.y + (0.5f - PATCH_HALF + (float)PATCH_DIM);

    bool good_patch = true;

    if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
      good_patch = false;  // the patch bound box is out of the image
    }

    if (good_patch) {
      float gradPatchX[ELEMS_PER_THREAD];
      float imagePatch1[ELEMS_PER_THREAD];

      float grad_x_sum = 0.f;
      float prev_patch_sum = 0.f;

#pragma unroll
      for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
        if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
          float gx = tex2D<float>(grad_x, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);
          float prev_value = tex2D<float>(prev_image, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);

          grad_x_sum += gx;
          prev_patch_sum += prev_value;

          gradPatchX[elem_id] = gx;
          imagePatch1[elem_id] = prev_value;
        }
      }
      grad_x_sum = reduce(grad_x_sum);
      prev_patch_sum = reduce(prev_patch_sum);

      float gxx = 1.f / 32.f;

#pragma unroll
      for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
        if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
          float gx = gradPatchX[elem_id] - grad_x_sum * windowSizeRcp;

          gradPatchX[elem_id] = gx;
          gxx += weights[elem_id] * gx * gx;
        }
      }
      gxx = reduce(gxx);

      if (gxx < kFloatEpsilon) {
        if (master) {
          status = false;
        }
        return;
      }

      float imagePatch2[ELEMS_PER_THREAD];
      float curr_patch_sum = 0.f;

      const float2 xy2Save = shifted_curr_xy;

      const float gxxRcp = __fdividef(1.0f, gxx);
      for (int iterations = 0; iterations < MAX_POINT_TRACK_ITERATIONS; iterations++) {
        xmin = shifted_curr_xy.x + (0.5f - PATCH_HALF);
        xmax = shifted_curr_xy.x + (0.5f - PATCH_HALF + (float)PATCH_DIM);
        ymin = shifted_curr_xy.y + (0.5f - PATCH_HALF);
        ymax = shifted_curr_xy.y + (0.5f - PATCH_HALF + (float)PATCH_DIM);

        if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
          good_patch = false;  // the patch bound box is out of the image
        }

        if (!good_patch) {
          break;
        }

        curr_patch_sum = 0.f;

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
            float curr_value = tex2D<float>(curr_image, xmin + ix_val[elem_id], ymin + iy_val[elem_id]);
            curr_patch_sum += curr_value;

            imagePatch2[elem_id] = curr_value;
          }
        }
        curr_patch_sum = reduce(curr_patch_sum);

        float delta = (curr_patch_sum - prev_patch_sum) * windowSizeRcp;

        float ex = 0.f;

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if ((elem_id < ELEMS_PER_THREAD - 1) || last_elem_valid) {
            float curr_value = imagePatch2[elem_id];
            float prev_value = imagePatch1[elem_id];
            float diff = curr_value - prev_value - delta;

            ex += weights[elem_id] * diff * gradPatchX[elem_id];
          }
        }
        ex = reduce(ex);

        const float v_x = ex * gxxRcp;
        const float v_norm = fabs(v_x);

        isConverged = (v_norm < (float)LK_CONVERGENCE_THRESHOLD);

        if (isConverged) {
          // TODO: investigate info matrix calculation,
          // now it's not correct and requires to use default info matrix
          // if ((lane_id == 0) || (lane_id == 3)) {
          //     info_mat[lane_id] = gxx;
          // }
          break;
        }

        shifted_curr_xy.x += v_x;
      }

      if (isConverged) {
        const float level_radius = search_radius_px / (float)(1 << level);
        const float dx = shifted_curr_xy.x - xy2Save.x;
        const float dy = shifted_curr_xy.y - xy2Save.y;
        if (dx * dx + dy * dy > level_radius * level_radius) {
          isConverged = false;
        }
      }

      if (isConverged) {
        float sum_prev_squared = 0.f;
        float sum_curr_squared = 0.f;
        float sum_prev_curr = 0.f;
        float sum_prev = 0.f;
        float sum_curr = 0.f;

        const float sizeRcp = 1.0f / (float)(NCC_DIM * NCC_DIM);

#pragma unroll
        for (int elem_id = 0; elem_id < ELEMS_PER_THREAD; ++elem_id) {
          if (ncc_mask[elem_id]) {
            float curr_value = imagePatch2[elem_id];
            float prev_value = imagePatch1[elem_id];

            sum_prev += prev_value;
            sum_curr += curr_value;

            sum_prev_squared += prev_value * prev_value;
            sum_curr_squared += curr_value * curr_value;
            sum_prev_curr += curr_value * prev_value;
          }
        }
        sum_prev = reduce(sum_prev);
        sum_curr = reduce(sum_curr);
        sum_prev_squared = reduce(sum_prev_squared);
        sum_curr_squared = reduce(sum_curr_squared);
        sum_prev_curr = reduce(sum_prev_curr);

        const float a1 = sum_prev * sizeRcp;
        const float a2 = sum_curr * sizeRcp;

        const float d1 = sum_prev_squared * sizeRcp - a1 * a1;
        const float d2 = sum_curr_squared * sizeRcp - a2 * a2;
        const float d12 = sum_prev_curr * sizeRcp - a1 * a2;

        const float d1d2 = d1 * d2;
        float ncc = d1d2 < kFloatEpsilon ? 0.f : d12 * __frsqrt_rn(d1d2);

        isConverged = ncc > ncc_threshold;
      }

      if (!isConverged && level > 0) {
        shifted_curr_xy.x = xy2Save.x;
        shifted_curr_xy.y = xy2Save.y;
      }

      isConverged = (level == 0) && isConverged;
    }  // can getGradXYPatches

    // go to next level
    if (i != levels - 1) {
      shifted_xy.x = shifted_xy.x * 2 - 0.5f;
      shifted_xy.y = shifted_xy.y * 2 - 0.5f;

      shifted_curr_xy.x = shifted_curr_xy.x * 2 - 0.5f;
      shifted_curr_xy.y = shifted_curr_xy.y * 2 - 0.5f;
    }
  }

  if (master) {
    if (isConverged) {
      track = shifted_curr_xy;
    }

    status = isConverged;
  }
}

cudaError_t lk_track(Pyramid prevFrameGradXPyramid, Pyramid prevFrameGradYPyramid, Pyramid prevFrameImagePyramid,
                     Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks, cudaStream_t stream) {
  size_t warps_per_block = 1;
  size_t threads = WARP_SIZE * warps_per_block;
  size_t blocks = (num_tracks + warps_per_block - 1) / warps_per_block;

  lk_track_kernel<<<blocks, threads, 0, stream>>>(prevFrameGradXPyramid, prevFrameGradYPyramid, prevFrameImagePyramid,
                                                  currentFrameImagePyramid, track_data, num_tracks);
  return cudaGetLastError();
}

cudaError_t lk_track_horizontal(Pyramid prevFrameGradXPyramid, Pyramid prevFrameImagePyramid,
                                Pyramid currentFrameImagePyramid, TrackData* track_data, int num_tracks,
                                cudaStream_t stream) {
  size_t warps_per_block = 1;
  size_t threads = WARP_SIZE * warps_per_block;
  size_t blocks = (num_tracks + warps_per_block - 1) / warps_per_block;

  lk_track_horizontal_kernel<<<blocks, threads, 0, stream>>>(prevFrameGradXPyramid, prevFrameImagePyramid,
                                                             currentFrameImagePyramid, track_data, num_tracks);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
