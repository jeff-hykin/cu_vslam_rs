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
#include <cmath>
#include <cstring>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

#define PATCH_LAST_ID (PATCH_DIM - 1)
#define PATCH_SIZE (PATCH_DIM * PATCH_DIM)

#define FULL_MAP_ORDER 6
#define PARTIAL_MAP_ORDER 2
#define ST_CONVERGENCE_THRESHOLD 0.04f

namespace cuvslam::cuda {

template <int Dim>
__device__ __forceinline__ void multiply(const float* M1, const float* M2, float* res) {
  for (int i = 0; i < Dim; i++) {
    for (int j = 0; j < Dim; j++) {
      res[i * Dim + j] = 0;
      for (int k = 0; k < Dim; k++) {
        res[i * Dim + j] += M1[i * Dim + k] * M2[k * Dim + j];
      }
    }
  }
}

/*Makes the AP = QR decomposition.
 * Q is orthogonal, R is upper triangular, P is the matrix of column transpositions.
 * The main idea is presented here: https://en.wikipedia.org/wiki/QR_decomposition.
 * Some details were taken from Eigen lib to mimic it as much as possible.
 * Reference code: https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/QR/ColPivHouseholderQR.h#L482*/
template <int Dim>
__device__ __forceinline__ bool ColPivHouseholderQR(const float* A, float* Q_T, float* P, float* R,
                                                    size_t& nonzero_pivots) {
  float Q_temp[Dim - 1][Dim * Dim];
  float A_temp[Dim * Dim];

  memcpy((void*)A_temp, (void*)A, Dim * Dim * sizeof(float));
  memset(P, 0, Dim * Dim * sizeof(float));

  float col_norms[Dim];
  float col_norms_direct[Dim];
  int transposition[Dim];

  float max_norm = -1;
  for (int col = 0; col < Dim; col++) {
    col_norms[col] = 0.f;
    for (int row = 0; row < Dim; row++) {
      float value = A_temp[row * Dim + col];
      col_norms[col] += value * value;
    }
    col_norms[col] = sqrt(col_norms[col]);
    col_norms_direct[col] = col_norms[col];
    if (col_norms[col] > max_norm) {
      max_norm = col_norms[col];
    }
    transposition[col] = col;
  }

  float threshold_helper = max_norm * kFloatEpsilon;
  threshold_helper = threshold_helper * threshold_helper / Dim;

  nonzero_pivots = Dim;

  for (int q_id = 0; q_id < Dim - 1; q_id++) {
    float max_norm_per_columns = -1;
    int max_col_id = -1;

    for (int col_id = q_id; col_id < Dim; col_id++) {
      if (col_norms[col_id] > max_norm_per_columns) {
        max_norm_per_columns = col_norms[col_id];
        max_col_id = col_id;
      }
    }

    if (nonzero_pivots == Dim && max_norm_per_columns * max_norm_per_columns < threshold_helper * (Dim - q_id)) {
      nonzero_pivots = q_id;
    }

    if (q_id != max_col_id) {
      int t = transposition[q_id];
      transposition[q_id] = transposition[max_col_id];
      transposition[max_col_id] = t;

      float norm = col_norms[q_id];
      col_norms[q_id] = col_norms[max_col_id];
      col_norms[max_col_id] = norm;

      norm = col_norms_direct[q_id];
      col_norms_direct[q_id] = col_norms_direct[max_col_id];
      col_norms_direct[max_col_id] = norm;

      for (int row = 0; row < Dim; row++) {
        float temp = A_temp[row * Dim + q_id];
        A_temp[row * Dim + q_id] = A_temp[row * Dim + max_col_id];
        A_temp[row * Dim + max_col_id] = temp;
      }
    }

    memcpy((void*)R, (void*)A_temp, Dim * Dim * sizeof(float));

    float c0 = A_temp[q_id * Dim + q_id];

    if (c0 >= 0) {
      max_norm_per_columns = -max_norm_per_columns;
    }
    A_temp[q_id * Dim + q_id] += max_norm_per_columns;

    float tau_devider = max_norm_per_columns * max_norm_per_columns + c0 * max_norm_per_columns;
    float tau = 1.f / tau_devider;
    if (fabs(tau_devider) < kFloatEpsilon) {
      tau = 0.f;
    }

    for (int i = 0; i < Dim; i++) {
      for (int j = 0; j < Dim; j++) {
        float I = (i == j) ? 1.f : 0.f;
        if (i < q_id || j < q_id) {
          Q_temp[q_id][i * Dim + j] = I;
          continue;
        }
        Q_temp[q_id][i * Dim + j] = I - tau * A_temp[i * Dim + q_id] * A_temp[j * Dim + q_id];
      }
    }

    multiply<Dim>(Q_temp[q_id], R, A_temp);
    memcpy((void*)R, (void*)A_temp, Dim * Dim * sizeof(float));

    for (int j = q_id + 1; j < Dim; j++) {
      if (col_norms[j] != 0) {
        float temp = fabs(A_temp[q_id * Dim + j]) / col_norms[j];
        temp = (1 + temp) * (1 - temp);
        temp = temp < 0 ? 0 : temp;

        float temp2 = col_norms[j] / col_norms_direct[j];
        temp2 = temp * temp2 * temp2;

        if (temp2 <= kFloatEpsilon) {
          // The updated norm has become too inaccurate so re-compute the column
          // norm directly.

          col_norms_direct[j] = 0.f;
          for (int row = q_id + 1; row < Dim; row++) {
            float v = A_temp[row * Dim + j];
            col_norms_direct[j] += v * v;
          }
          col_norms_direct[j] = sqrt(col_norms_direct[j]);

          col_norms[j] = col_norms_direct[j];
        } else {
          col_norms[j] *= sqrt(temp);
        }
      }
    }
  }

  for (int i = Dim - 2; i > 0; i--) {
    multiply<Dim>(Q_temp[i], Q_temp[i - 1], A_temp);
    memcpy((void*)Q_temp[i - 1], (void*)A_temp, Dim * Dim * sizeof(float));
  }

  memcpy((void*)Q_T, (void*)Q_temp[0], Dim * Dim * sizeof(float));

  for (int i = 0; i < Dim; i++) {
    P[i * Dim + transposition[i]] = 1;
  }

  return true;
}

/* Solve the triangular linear equation */
template <int Dim>
__device__ __forceinline__ bool ColPivHouseholderQRSolve(const float* A, const float* b, float* x) {
  float Q_T[Dim * Dim];
  float R[Dim * Dim];
  float P[Dim * Dim];
  float x_transposed[Dim];

  size_t nonzero_pivots;

  if (!ColPivHouseholderQR<Dim>(A, Q_T, P, R, nonzero_pivots)) {
    return false;
  }

  for (int i = 0; i < Dim; i++) {
    x_transposed[i] = 0;
    for (int j = 0; j < Dim; j++) {
      x_transposed[i] += Q_T[i * Dim + j] * b[j];
    }
  }

  // solve triangular equation
  for (int i = nonzero_pivots - 1; i >= 0; i--) {
    if (fabs(R[i * Dim + i]) < kFloatEpsilon) {
      if (fabs(x_transposed[i]) > kFloatEpsilon) {
        return false;  // no solution exists
      }
    } else {
      x_transposed[i] /= R[i * Dim + i];
    }

    for (int j = i - 1; j >= 0; j--) {
      x_transposed[j] -= x_transposed[i] * R[j * Dim + i];
    }
  }

  for (int row = 0; row < Dim; row++) {
    x[row] = 0;
    for (int col = 0; col < Dim; col++) {
      x[row] += P[col * Dim + row] * x_transposed[col];
    }
  }

  return true;
}

__constant__ float x_patch[PATCH_SIZE];
__constant__ float y_patch[PATCH_SIZE];
__constant__ float xx_patch[PATCH_SIZE];
__constant__ float xy_patch[PATCH_SIZE];
__constant__ float yy_patch[PATCH_SIZE];
__constant__ float gauss_coeffs[PATCH_SIZE];

constexpr float gauss_coeffs_cpu[9] = {0.0135196f, 0.0476622f, 0.11723f,   0.201168f, 0.240841f,
                                       0.201168f,  0.11723f,   0.0476622f, 0.0135196f};

static bool st_initialized = false;

cudaError_t init_st_tracker() {
  if (st_initialized) {
    return cudaSuccess;
  }
  float cpu_x_patch[PATCH_SIZE];
  float cpu_y_patch[PATCH_SIZE];
  float cpu_xx_patch[PATCH_SIZE];
  float cpu_xy_patch[PATCH_SIZE];
  float cpu_yy_patch[PATCH_SIZE];

  const int m = PATCH_LAST_ID / 2;
  for (int y = 0; y < PATCH_DIM; ++y) {
    for (int x = 0; x < PATCH_DIM; ++x) {
      int patch_idx = y * PATCH_DIM + x;
      float x_v = static_cast<float>(x - m);
      float y_v = static_cast<float>(y - m);
      cpu_x_patch[patch_idx] = x_v;
      cpu_y_patch[patch_idx] = y_v;
      cpu_xx_patch[patch_idx] = x_v * x_v;
      cpu_xy_patch[patch_idx] = x_v * y_v;
      cpu_yy_patch[patch_idx] = y_v * y_v;
    }
  }

  float gauss_coeffs_array[PATCH_DIM * PATCH_DIM];

  for (int i = 0; i < PATCH_DIM; i++) {
    for (int j = 0; j < PATCH_DIM; j++) {
      int idx = i * PATCH_DIM + j;
      gauss_coeffs_array[idx] = gauss_coeffs_cpu[i] * gauss_coeffs_cpu[j];
    }
  }

  cudaError_t error = cudaMemcpyToSymbol(x_patch, cpu_x_patch, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(y_patch, cpu_y_patch, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(xx_patch, cpu_xx_patch, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(xy_patch, cpu_xy_patch, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }
  error = cudaMemcpyToSymbol(yy_patch, cpu_yy_patch, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }

  error = cudaMemcpyToSymbol(gauss_coeffs, gauss_coeffs_array, PATCH_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    return error;
  }

  st_initialized = true;
  return cudaSuccess;
}

__device__ __forceinline__ bool is_identity(const float4& map) {
  if (map.x != 1.f || map.y != 0.f || map.z != 0.f || map.w != 1.f) {
    return false;
  }
  return true;
}

__device__ __forceinline__ bool isPointInImage(const float2& point, const Size& size) {
  return point.x >= 0 && point.x < (float)size.width && point.y >= 0 && point.y < (float)size.height;
}

__global__ void st_track_kernel(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid prevImagePyramid,
                                Pyramid currentImagePyramid, TrackData* track_data, int num_tracks,
                                unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= num_tracks) {
    return;
  }

  float2& track = track_data[x].track;
  const float2& offset = track_data[x].offset;
  float* info_mat = track_data[x].info;
  const float4& initial_guess_map = track_data[x].initial_guess_map;
  bool& status = track_data[x].track_status;
  float search_radius_px = track_data[x].search_radius_px;
  float ncc_threshold = track_data[x].ncc_threshold;
  float& ncc = track_data[x].ncc;

  if (search_radius_px < 0.f) {
    status = false;
    return;
  }

  const int coarsest_level = min(currentImagePyramid.levels - 1, (int)(log1pf(search_radius_px)));

  assert(coarsest_level >= 0);
  assert(coarsest_level < currentImagePyramid.levels);

  if (!isPointInImage(track, prevImagePyramid.level_sizes[0])) {
    status = false;
    return;
  }

  const float scale = 1.f / static_cast<float>(1 << coarsest_level);  // scale = pow(0.5, level);
  const float shift = 0.5f;

  float2 previous_uv = {(track.x - shift) * scale + shift, (track.y - shift) * scale + shift};

  float4 map = initial_guess_map;

  float2 xy = {(track.x + offset.x - shift) * scale + shift, (track.y + offset.y - shift) * scale + shift};

  bool isConverged = false;

  for (int level = coarsest_level; level >= 0; --level) {
    cudaTextureObject_t grad_x = currentGradXPyramid.level_tex[level];
    cudaTextureObject_t grad_y = currentGradYPyramid.level_tex[level];
    cudaTextureObject_t curr_image = currentImagePyramid.level_tex[level];
    cudaTextureObject_t prev_image = prevImagePyramid.level_tex[level];
    Size level_size = currentImagePyramid.level_sizes[level];

    if (!isPointInImage(previous_uv, level_size)) {
      status = false;
      return;
    }

    if (!isPointInImage(xy, level_size)) {
      status = false;
      return;
    }

    float xmin = previous_uv.x - PATCH_HALF + 0.5f;
    float xmax = xmin + PATCH_DIM;
    float ymin = previous_uv.y - PATCH_HALF + 0.5f;
    float ymax = ymin + PATCH_DIM;

    bool good_patch = true;
    if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
      good_patch = false;  // the patch bound box is out of the image
    }

    float previous_patch[PATCH_SIZE];
    float prev_patch_sum = 0.f;
    if (good_patch) {
#pragma unroll
      for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
        for (int ix = 0; ix < PATCH_DIM; ix++) {
          int patch_idx = iy * PATCH_DIM + ix;
          previous_patch[patch_idx] = tex2D<float>(prev_image, xmin + ix, ymin + iy);
          prev_patch_sum += previous_patch[patch_idx];
        }
      }
    } else {
      if (level > 0) {
        xy.x = xy.x * 2.f - 0.5f;
        xy.y = xy.y * 2.f - 0.5f;

        previous_uv.x = previous_uv.x * 2.f - 0.5f;
        previous_uv.y = previous_uv.y * 2.f - 0.5f;
        continue;
      }
    }

    bool refine_mapping_status = true;

    {
      float4 map_ = map;
      float2 xy_ = xy;

      // logic here comes from camera::GetDefaultObservationInfoUV()
      info_mat[0] = 1.f / 9.f;
      info_mat[1] = 0.f;
      info_mat[2] = 0.f;
      info_mat[3] = 1.f / 9.f;

      const unsigned n_iterations = n_shift_only_iterations + n_full_mapping_iterations;
      for (unsigned i = 0; i < n_iterations; ++i) {
        const bool full_mapping = i >= n_shift_only_iterations;

        if (!isPointInImage(xy_, level_size)) {
          refine_mapping_status = false;
          break;
        }

        float current_patch[PATCH_SIZE];
        float gx[PATCH_SIZE];
        float gy[PATCH_SIZE];

        float curr_patch_sum = 0.f;
        float gx_sum = 0.f;
        float gy_sum = 0.f;

        if (is_identity(map_)) {
          xmin = xy_.x - PATCH_HALF + 0.5f;
          xmax = xmin + PATCH_DIM;
          ymin = xy_.y - PATCH_HALF + 0.5f;
          ymax = ymin + PATCH_DIM;

          if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
            good_patch = false;  // the patch bound box is out of the image
          }

#pragma unroll
          for (int iy = 0; iy < PATCH_DIM && good_patch; iy++) {
#pragma unroll
            for (int ix = 0; ix < PATCH_DIM && good_patch; ix++) {
              int patch_idx = iy * PATCH_DIM + ix;
              current_patch[patch_idx] = tex2D<float>(curr_image, xmin + ix, ymin + iy);
              gx[patch_idx] = tex2D<float>(grad_x, xmin + ix, ymin + iy);
              gy[patch_idx] = tex2D<float>(grad_y, xmin + ix, ymin + iy);

              curr_patch_sum += current_patch[patch_idx];
              gx_sum += gx[patch_idx];
              gy_sum += gy[patch_idx];
            }
          }

        } else {
          const float2 xy00 = {xy_.x - 0.5f - PATCH_LAST_ID * (map_.x + map_.y) / 2.f,
                               xy_.y - 0.5f - PATCH_LAST_ID * (map_.z + map_.w) / 2.f};

          for (int iy = 0; iy < PATCH_DIM && good_patch; ++iy) {
            for (int ix = 0; ix < PATCH_DIM && good_patch; ++ix) {
              float _x = xy00.x + map_.x * (float)ix + map_.y * (float)iy;
              float _y = xy00.y + map_.z * (float)ix + map_.w * (float)iy;

              if (_x < 0 || _x >= level_size.width || _y < 0 || _y >= level_size.height) {
                good_patch = false;
                break;
              }

              _x += 0.5f;
              _y += 0.5f;

              int patch_idx = iy * PATCH_DIM + ix;
              current_patch[patch_idx] = tex2D<float>(curr_image, _x, _y);
              gx[patch_idx] = tex2D<float>(grad_x, _x, _y);
              gy[patch_idx] = tex2D<float>(grad_y, _x, _y);

              curr_patch_sum += current_patch[patch_idx];
              gx_sum += gx[patch_idx];
              gy_sum += gy[patch_idx];
            }
          }
        }

        if (!good_patch) {
          refine_mapping_status = false;
          break;
        }

        float delta = (curr_patch_sum - prev_patch_sum) / PATCH_SIZE;

        float w__gxgx = 0.f;
        float w__gxgy = 0.f;
        float w__gygy = 0.f;

        float wr_gx = 0.f;
        float wr_gy = 0.f;

        float residual[PATCH_SIZE];

#pragma unroll
        for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
          for (int ix = 0; ix < PATCH_DIM; ix++) {
            int patch_idx = iy * PATCH_DIM + ix;

            float gx_normed = gx[patch_idx] - gx_sum / PATCH_SIZE;
            float gy_normed = gy[patch_idx] - gy_sum / PATCH_SIZE;

            float weight = gauss_coeffs[patch_idx];
            float curr_value = current_patch[patch_idx];
            float prev_value = previous_patch[patch_idx];

            float diff = curr_value - prev_value - delta;

            gx[patch_idx] = gx_normed;
            gy[patch_idx] = gy_normed;

            w__gxgx += weight * gx_normed * gx_normed;
            w__gxgy += weight * gx_normed * gy_normed;
            w__gygy += weight * gy_normed * gy_normed;

            residual[patch_idx] = diff;
            wr_gx += weight * diff * gx_normed;
            wr_gy += weight * diff * gy_normed;
          }
        }

        {
          const float determinant = w__gxgx * w__gygy - w__gxgy * w__gxgy;

          const float2 v = {(w__gygy * wr_gx - w__gxgy * wr_gy) / determinant,
                            (w__gxgx * wr_gy - w__gxgy * wr_gx) / determinant};

          const float v_norm = sqrt(v.x * v.x + v.y * v.y);

          isConverged = (v_norm < ST_CONVERGENCE_THRESHOLD);

          if (isConverged) {
            // TODO: investigate info matrix calculation,
            // now it's not correct and requires to use default info matrix
            // info_mat[0] = w__gxgx;
            // info_mat[1] = w__gxgy;
            // info_mat[2] = w__gxgy;
            // info_mat[3] = w__gygy;
            break;
          }
        }

        if (full_mapping) {
          float wx_gxgx = 0.f;
          float wx_gxgy = 0.f;
          float wx_gygy = 0.f;

          float wy_gxgx = 0.f;
          float wy_gxgy = 0.f;
          float wy_gygy = 0.f;

          float wxxgxgx = 0.f;
          float wxxgxgy = 0.f;
          float wxxgygy = 0.f;

          float wxygxgx = 0.f;
          float wxygxgy = 0.f;
          float wxygygy = 0.f;

          float wyygxgx = 0.f;
          float wyygxgy = 0.f;
          float wyygygy = 0.f;

          float wrxgx = 0.f;
          float wrxgy = 0.f;
          float wrygx = 0.f;
          float wrygy = 0.f;

#pragma unroll
          for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
            for (int ix = 0; ix < PATCH_DIM; ix++) {
              int patch_idx = iy * PATCH_DIM + ix;

              float weight = gauss_coeffs[patch_idx];
              float gx_value = gx[patch_idx];
              float gy_value = gy[patch_idx];

              float gxx = gx_value * gx_value;
              float gxy = gx_value * gy_value;
              float gyy = gy_value * gy_value;

              float wx = x_patch[patch_idx] * weight;
              float wy = y_patch[patch_idx] * weight;
              float wxx = xx_patch[patch_idx] * weight;
              float wxy = xy_patch[patch_idx] * weight;
              float wyy = yy_patch[patch_idx] * weight;
              float r = residual[patch_idx];

              wx_gxgx += wx * gxx;
              wx_gxgy += wx * gxy;
              wx_gygy += wx * gyy;

              wy_gxgx += wy * gxx;
              wy_gxgy += wy * gxy;
              wy_gygy += wy * gyy;

              wxxgxgx += wxx * gxx;
              wxxgxgy += wxx * gxy;
              wxxgygy += wxx * gyy;

              wxygxgx += wxy * gxx;
              wxygxgy += wxy * gxy;
              wxygygy += wxy * gyy;

              wyygxgx += wyy * gxx;
              wyygxgy += wyy * gxy;
              wyygygy += wyy * gyy;

              wrxgx += r * wx * gx_value;
              wrxgy += r * wx * gy_value;
              wrygx += r * wy * gx_value;
              wrygy += r * wy * gy_value;
            }
          }

          float t[FULL_MAP_ORDER * FULL_MAP_ORDER] = {
              wxxgxgx, wxxgxgy, wxygxgx, wxygxgy, wx_gxgx, wx_gxgy, wxxgxgy, wxxgygy, wxygxgy,
              wxygygy, wx_gxgy, wx_gygy, wxygxgx, wxygxgy, wyygxgx, wyygxgy, wy_gxgx, wy_gxgy,
              wxygxgy, wxygygy, wyygxgy, wyygygy, wy_gxgy, wy_gygy, wx_gxgx, wx_gxgy, wy_gxgx,
              wy_gxgy, w__gxgx, w__gxgy, wx_gxgy, wx_gygy, wy_gxgy, wy_gygy, w__gxgy, w__gygy};

          float a[FULL_MAP_ORDER] = {wrxgx, wrxgy, wrygx, wrygy, wr_gx, wr_gy};

          float z[FULL_MAP_ORDER];

          if (!ColPivHouseholderQRSolve<FULL_MAP_ORDER>(t, a, z)) {
            refine_mapping_status = false;
            break;
          }

          const float& dxx = z[0];
          const float& dyx = z[1];
          const float& dxy = z[2];
          const float& dyy = z[3];
          const float& dx = z[4];
          const float& dy = z[5];

          map_ = {map_.x + dxx, map_.y + dyx, map_.z + dxy, map_.w + dyy};

          xy_.x += dx;
          xy_.y += dy;
        } else {
          float t[PARTIAL_MAP_ORDER * PARTIAL_MAP_ORDER] = {w__gxgx, w__gxgy, w__gxgy, w__gygy};

          float a[PARTIAL_MAP_ORDER] = {wr_gx, wr_gy};
          float z[PARTIAL_MAP_ORDER];

          if (!ColPivHouseholderQRSolve<PARTIAL_MAP_ORDER>(t, a, z)) {
            refine_mapping_status = false;
            break;
          }

          xy_.x += z[0];
          xy_.y += z[1];
        }
      }

      if (refine_mapping_status) {
        if (!isPointInImage(xy_, level_size)) {
          refine_mapping_status = false;
        }
      }
      if (refine_mapping_status) {
        map = map_;
        xy = xy_;
      }
    }

    if (!refine_mapping_status) {
      if (level > 0) {
        xy.x = xy.x * 2.f - 0.5f;
        xy.y = xy.y * 2.f - 0.5f;

        previous_uv.x = previous_uv.x * 2.f - 0.5f;
        previous_uv.y = previous_uv.y * 2.f - 0.5f;
        continue;
      } else {
        status = false;
        return;
      }
    }

    float current_patch[PATCH_SIZE];

    if (is_identity(map)) {
      xmin = xy.x - PATCH_HALF + 0.5f;
      xmax = xmin + PATCH_DIM;
      ymin = xy.y - PATCH_HALF + 0.5f;
      ymax = ymin + PATCH_DIM;

      if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
        good_patch = false;  // the patch bound box is out of the image
      }

#pragma unroll
      for (int iy = 0; iy < PATCH_DIM && good_patch; iy++) {
#pragma unroll
        for (int ix = 0; ix < PATCH_DIM && good_patch; ix++) {
          int patch_idx = iy * PATCH_DIM + ix;
          current_patch[patch_idx] = tex2D<float>(curr_image, xmin + ix, ymin + iy);
        }
      }

    } else {
      const float2 xy00 = {xy.x - 0.5f - PATCH_LAST_ID * (map.x + map.y) / 2.f,
                           xy.y - 0.5f - PATCH_LAST_ID * (map.z + map.w) / 2.f};

      for (int iy = 0; iy < PATCH_DIM && good_patch; ++iy) {
        for (int ix = 0; ix < PATCH_DIM && good_patch; ++ix) {
          float _x = xy00.x + map.x * (float)ix + map.y * (float)iy;
          float _y = xy00.y + map.z * (float)ix + map.w * (float)iy;

          if (_x < 0 || _x >= level_size.width || _y < 0 || _y >= level_size.height) {
            good_patch = false;
            break;
          }

          _x += 0.5f;
          _y += 0.5f;

          int patch_idx = iy * PATCH_DIM + ix;
          current_patch[patch_idx] = tex2D<float>(curr_image, _x, _y);
        }
      }
    }

    if (!good_patch) {
      status = false;
      return;
    }

    if (level == 0) {
      float ncc_value = 0.f;
      {
        int start = (PATCH_DIM - NCC_DIM) / 2;

        float curr_sum = 0;
        float prev_sum = 0;

        float cur_sum_squared = 0;
        float prev_sum_squared = 0;
        float curr_prev_sum = 0;

        const float size = NCC_DIM * NCC_DIM;

#pragma unroll
        for (int i = start; i < start + NCC_DIM; i++) {
#pragma unroll
          for (int k = start; k < start + NCC_DIM; k++) {
            size_t idx = i * PATCH_DIM + k;

            float curr_value = current_patch[idx];
            float prev_value = previous_patch[idx];

            curr_sum += curr_value;
            prev_sum += prev_value;

            cur_sum_squared += curr_value * curr_value;
            prev_sum_squared += prev_value * prev_value;
            curr_prev_sum += curr_value * prev_value;
          }
        }

        const float curr_mean = curr_sum / size;
        const float prev_mean = prev_sum / size;

        const float curr_var = cur_sum_squared / size - curr_mean * curr_mean;
        const float prev_var = prev_sum_squared / size - prev_mean * prev_mean;
        const float cov = curr_prev_sum / size - curr_mean * prev_mean;

        const float d1d2 = curr_var * prev_var;
        ncc_value = d1d2 < kFloatEpsilon ? 0 : cov / sqrt(d1d2);
      }

      if (ncc_value <= ncc_threshold || !isConverged) {
        status = false;
        return;
      }

      ncc = ncc_value;
    }

    if (level > 0) {
      xy.x = xy.x * 2.f - 0.5f;
      xy.y = xy.y * 2.f - 0.5f;

      previous_uv.x = previous_uv.x * 2.f - 0.5f;
      previous_uv.y = previous_uv.y * 2.f - 0.5f;
    }
  }

  track = xy;

  status = isPointInImage(track, currentImagePyramid.level_sizes[0]);
}

__global__ void st_track_with_cache_kernel(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid,
                                           Pyramid currentImagePyramid, PointCacheDataT* cache_data,
                                           TrackData* track_data, int num_tracks, unsigned n_shift_only_iterations,
                                           unsigned n_full_mapping_iterations) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= num_tracks) {
    return;
  }

  float2& track = track_data[x].track;
  const float2& offset = track_data[x].offset;
  float* info_mat = track_data[x].info;
  float search_radius_px = track_data[x].search_radius_px;
  float ncc_threshold = track_data[x].ncc_threshold;
  float& ncc = track_data[x].ncc;
  const float4& initial_guess_map = track_data[x].initial_guess_map;
  bool& status = track_data[x].track_status;
  int cache_index = track_data[x].cache_index;

  if (cache_index == -1) {
    status = false;
    return;
  }

  const PointCacheDataT& data = cache_data[cache_index];
  uint32_t levels_mask = data.level_mask;

  if (search_radius_px < 0.f) {
    status = false;
    return;
  }

  const int coarsest_level = min(currentImagePyramid.levels - 1, (int)(log1pf(search_radius_px)));

  assert(coarsest_level >= 0);
  assert(coarsest_level < currentImagePyramid.levels);

  const float scale = 1.f / static_cast<float>(1 << coarsest_level);  // scale = pow(0.5, level);
  const float shift = 0.5f;

  float2 previous_uv = {(track.x - shift) * scale + shift, (track.y - shift) * scale + shift};

  float4 map = initial_guess_map;

  float2 xy = {(track.x + offset.x - shift) * scale + shift, (track.y + offset.y - shift) * scale + shift};

  bool isConverged = false;

  for (int level = coarsest_level; level >= 0; --level) {
    cudaTextureObject_t grad_x = currentGradXPyramid.level_tex[level];
    cudaTextureObject_t grad_y = currentGradYPyramid.level_tex[level];
    cudaTextureObject_t curr_image = currentImagePyramid.level_tex[level];
    Size level_size = currentImagePyramid.level_sizes[level];

    if (!isPointInImage(xy, level_size)) {
      status = false;
      return;
    }

    if ((levels_mask & (1 << level)) == 0) {
      if (level > 0) {
        xy.x = xy.x * 2.f - 0.5f;
        xy.y = xy.y * 2.f - 0.5f;

        previous_uv.x = previous_uv.x * 2.f - 0.5f;
        previous_uv.y = previous_uv.y * 2.f - 0.5f;
        continue;
      }
    }

    float previous_patch[PATCH_SIZE];

    void* ptr = (void*)(data.patch + PATCH_SIZE * level);
    memcpy((void*)previous_patch, ptr, PATCH_SIZE * sizeof(float));
    float prev_patch_sum = data.patch_sums[level];

    bool refine_mapping_status = true;

    {
      float4 map_ = map;
      float2 xy_ = xy;

      // logic here comes from camera::GetDefaultObservationInfoUV()
      info_mat[0] = 1.f / 9.f;
      info_mat[1] = 0.f;
      info_mat[2] = 0.f;
      info_mat[3] = 1.f / 9.f;

      const unsigned n_iterations = n_shift_only_iterations + n_full_mapping_iterations;
      for (unsigned i = 0; i < n_iterations; ++i) {
        const bool full_mapping = i >= n_shift_only_iterations;

        if (!isPointInImage(xy_, level_size)) {
          refine_mapping_status = false;
          break;
        }

        float current_patch[PATCH_SIZE];
        float gx[PATCH_SIZE];
        float gy[PATCH_SIZE];

        float curr_patch_sum = 0.f;
        float gx_sum = 0.f;
        float gy_sum = 0.f;

        bool good_patch = true;

        if (is_identity(map_)) {
          float xmin = xy_.x - PATCH_HALF + 0.5f;
          float xmax = xmin + PATCH_DIM;
          float ymin = xy_.y - PATCH_HALF + 0.5f;
          float ymax = ymin + PATCH_DIM;

          if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
            good_patch = false;  // the patch bound box is out of the image
          }

#pragma unroll
          for (int iy = 0; iy < PATCH_DIM && good_patch; iy++) {
#pragma unroll
            for (int ix = 0; ix < PATCH_DIM && good_patch; ix++) {
              int patch_idx = iy * PATCH_DIM + ix;
              current_patch[patch_idx] = tex2D<float>(curr_image, xmin + ix, ymin + iy);
              gx[patch_idx] = tex2D<float>(grad_x, xmin + ix, ymin + iy);
              gy[patch_idx] = tex2D<float>(grad_y, xmin + ix, ymin + iy);

              curr_patch_sum += current_patch[patch_idx];
              gx_sum += gx[patch_idx];
              gy_sum += gy[patch_idx];
            }
          }

        } else {
          const float2 xy00 = {xy_.x - 0.5f - PATCH_LAST_ID * (map_.x + map_.y) / 2.f,
                               xy_.y - 0.5f - PATCH_LAST_ID * (map_.z + map_.w) / 2.f};

          for (int iy = 0; iy < PATCH_DIM && good_patch; ++iy) {
            for (int ix = 0; ix < PATCH_DIM && good_patch; ++ix) {
              float _x = xy00.x + map_.x * (float)ix + map_.y * (float)iy;
              float _y = xy00.y + map_.z * (float)ix + map_.w * (float)iy;

              if (_x < 0 || _x >= level_size.width || _y < 0 || _y >= level_size.height) {
                good_patch = false;
                break;
              }

              _x += 0.5f;
              _y += 0.5f;

              int patch_idx = iy * PATCH_DIM + ix;
              current_patch[patch_idx] = tex2D<float>(curr_image, _x, _y);
              gx[patch_idx] = tex2D<float>(grad_x, _x, _y);
              gy[patch_idx] = tex2D<float>(grad_y, _x, _y);

              curr_patch_sum += current_patch[patch_idx];
              gx_sum += gx[patch_idx];
              gy_sum += gy[patch_idx];
            }
          }
        }

        if (!good_patch) {
          refine_mapping_status = false;
          break;
        }

        float delta = (curr_patch_sum - prev_patch_sum) / PATCH_SIZE;

        float w__gxgx = 0.f;
        float w__gxgy = 0.f;
        float w__gygy = 0.f;

        float wr_gx = 0.f;
        float wr_gy = 0.f;

        float residual[PATCH_SIZE];

#pragma unroll
        for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
          for (int ix = 0; ix < PATCH_DIM; ix++) {
            int patch_idx = iy * PATCH_DIM + ix;

            float gx_normed = gx[patch_idx] - gx_sum / PATCH_SIZE;
            float gy_normed = gy[patch_idx] - gy_sum / PATCH_SIZE;

            float weight = gauss_coeffs[patch_idx];
            float curr_value = current_patch[patch_idx];
            float prev_value = previous_patch[patch_idx];

            float diff = curr_value - prev_value - delta;

            gx[patch_idx] = gx_normed;
            gy[patch_idx] = gy_normed;

            w__gxgx += weight * gx_normed * gx_normed;
            w__gxgy += weight * gx_normed * gy_normed;
            w__gygy += weight * gy_normed * gy_normed;

            residual[patch_idx] = diff;
            wr_gx += weight * diff * gx_normed;
            wr_gy += weight * diff * gy_normed;
          }
        }

        {
          const float determinant = w__gxgx * w__gygy - w__gxgy * w__gxgy;

          const float2 v = {(w__gygy * wr_gx - w__gxgy * wr_gy) / determinant,
                            (w__gxgx * wr_gy - w__gxgy * wr_gx) / determinant};

          const float v_norm = sqrt(v.x * v.x + v.y * v.y);

          isConverged = (v_norm < ST_CONVERGENCE_THRESHOLD);

          if (isConverged) {
            // TODO: investigate info matrix calculation,
            // now it's not correct and requires to use default info matrix
            // info_mat[0] = w__gxgx;
            // info_mat[1] = w__gxgy;
            // info_mat[2] = w__gxgy;
            // info_mat[3] = w__gygy;
            break;
          }
        }

        if (full_mapping) {
          float wx_gxgx = 0.f;
          float wx_gxgy = 0.f;
          float wx_gygy = 0.f;

          float wy_gxgx = 0.f;
          float wy_gxgy = 0.f;
          float wy_gygy = 0.f;

          float wxxgxgx = 0.f;
          float wxxgxgy = 0.f;
          float wxxgygy = 0.f;

          float wxygxgx = 0.f;
          float wxygxgy = 0.f;
          float wxygygy = 0.f;

          float wyygxgx = 0.f;
          float wyygxgy = 0.f;
          float wyygygy = 0.f;

          float wrxgx = 0.f;
          float wrxgy = 0.f;
          float wrygx = 0.f;
          float wrygy = 0.f;

#pragma unroll
          for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
            for (int ix = 0; ix < PATCH_DIM; ix++) {
              int patch_idx = iy * PATCH_DIM + ix;

              float weight = gauss_coeffs[patch_idx];
              float gx_value = gx[patch_idx];
              float gy_value = gy[patch_idx];

              float gxx = gx_value * gx_value;
              float gxy = gx_value * gy_value;
              float gyy = gy_value * gy_value;

              float wx = x_patch[patch_idx] * weight;
              float wy = y_patch[patch_idx] * weight;
              float wxx = xx_patch[patch_idx] * weight;
              float wxy = xy_patch[patch_idx] * weight;
              float wyy = yy_patch[patch_idx] * weight;
              float r = residual[patch_idx];

              wx_gxgx += wx * gxx;
              wx_gxgy += wx * gxy;
              wx_gygy += wx * gyy;

              wy_gxgx += wy * gxx;
              wy_gxgy += wy * gxy;
              wy_gygy += wy * gyy;

              wxxgxgx += wxx * gxx;
              wxxgxgy += wxx * gxy;
              wxxgygy += wxx * gyy;

              wxygxgx += wxy * gxx;
              wxygxgy += wxy * gxy;
              wxygygy += wxy * gyy;

              wyygxgx += wyy * gxx;
              wyygxgy += wyy * gxy;
              wyygygy += wyy * gyy;

              wrxgx += r * wx * gx_value;
              wrxgy += r * wx * gy_value;
              wrygx += r * wy * gx_value;
              wrygy += r * wy * gy_value;
            }
          }

          float t[FULL_MAP_ORDER * FULL_MAP_ORDER] = {
              wxxgxgx, wxxgxgy, wxygxgx, wxygxgy, wx_gxgx, wx_gxgy, wxxgxgy, wxxgygy, wxygxgy,
              wxygygy, wx_gxgy, wx_gygy, wxygxgx, wxygxgy, wyygxgx, wyygxgy, wy_gxgx, wy_gxgy,
              wxygxgy, wxygygy, wyygxgy, wyygygy, wy_gxgy, wy_gygy, wx_gxgx, wx_gxgy, wy_gxgx,
              wy_gxgy, w__gxgx, w__gxgy, wx_gxgy, wx_gygy, wy_gxgy, wy_gygy, w__gxgy, w__gygy};

          float a[FULL_MAP_ORDER] = {wrxgx, wrxgy, wrygx, wrygy, wr_gx, wr_gy};

          float z[FULL_MAP_ORDER];

          if (!ColPivHouseholderQRSolve<FULL_MAP_ORDER>(t, a, z)) {
            refine_mapping_status = false;
            break;
          }

          const float& dxx = z[0];
          const float& dyx = z[1];
          const float& dxy = z[2];
          const float& dyy = z[3];
          const float& dx = z[4];
          const float& dy = z[5];

          map_ = {map_.x + dxx, map_.y + dyx, map_.z + dxy, map_.w + dyy};

          xy_.x += dx;
          xy_.y += dy;
        } else {
          float t[PARTIAL_MAP_ORDER * PARTIAL_MAP_ORDER] = {w__gxgx, w__gxgy, w__gxgy, w__gygy};

          float a[PARTIAL_MAP_ORDER] = {wr_gx, wr_gy};
          float z[PARTIAL_MAP_ORDER];

          if (!ColPivHouseholderQRSolve<PARTIAL_MAP_ORDER>(t, a, z)) {
            refine_mapping_status = false;
            break;
          }

          xy_.x += z[0];
          xy_.y += z[1];
        }
      }

      if (refine_mapping_status) {
        if (!isPointInImage(xy_, level_size)) {
          refine_mapping_status = false;
        }
      }
      if (refine_mapping_status) {
        map = map_;
        xy = xy_;
      }
    }

    if (!refine_mapping_status) {
      if (level > 0) {
        xy.x = xy.x * 2.f - 0.5f;
        xy.y = xy.y * 2.f - 0.5f;

        previous_uv.x = previous_uv.x * 2.f - 0.5f;
        previous_uv.y = previous_uv.y * 2.f - 0.5f;
        continue;
      } else {
        status = false;
        return;
      }
    }

    float current_patch[PATCH_SIZE];

    bool good_patch = true;
    if (is_identity(map)) {
      float xmin = xy.x - PATCH_HALF + 0.5f;
      float xmax = xmin + PATCH_DIM;
      float ymin = xy.y - PATCH_HALF + 0.5f;
      float ymax = ymin + PATCH_DIM;

      if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
        good_patch = false;  // the patch bound box is out of the image
      }

#pragma unroll
      for (int iy = 0; iy < PATCH_DIM && good_patch; iy++) {
#pragma unroll
        for (int ix = 0; ix < PATCH_DIM && good_patch; ix++) {
          int patch_idx = iy * PATCH_DIM + ix;
          current_patch[patch_idx] = tex2D<float>(curr_image, xmin + ix, ymin + iy);
        }
      }

    } else {
      const float2 xy00 = {xy.x - 0.5f - PATCH_LAST_ID * (map.x + map.y) / 2.f,
                           xy.y - 0.5f - PATCH_LAST_ID * (map.z + map.w) / 2.f};

      for (int iy = 0; iy < PATCH_DIM && good_patch; ++iy) {
        for (int ix = 0; ix < PATCH_DIM && good_patch; ++ix) {
          float _x = xy00.x + map.x * (float)ix + map.y * (float)iy;
          float _y = xy00.y + map.z * (float)ix + map.w * (float)iy;

          if (_x < 0 || _x >= level_size.width || _y < 0 || _y >= level_size.height) {
            good_patch = false;
            break;
          }

          _x += 0.5f;
          _y += 0.5f;

          int patch_idx = iy * PATCH_DIM + ix;
          current_patch[patch_idx] = tex2D<float>(curr_image, _x, _y);
        }
      }
    }

    if (!good_patch) {
      status = false;
      return;
    }

    if (level == 0) {
      float ncc_value = 0.f;
      {
        int start = (PATCH_DIM - NCC_DIM) / 2;

        float curr_sum = 0;
        float prev_sum = 0;

        float cur_sum_squared = 0;
        float prev_sum_squared = 0;
        float curr_prev_sum = 0;

        const float size = NCC_DIM * NCC_DIM;

#pragma unroll
        for (int i = start; i < start + NCC_DIM; i++) {
#pragma unroll
          for (int k = start; k < start + NCC_DIM; k++) {
            size_t idx = i * PATCH_DIM + k;

            float curr_value = current_patch[idx];
            float prev_value = previous_patch[idx];

            curr_sum += curr_value;
            prev_sum += prev_value;

            cur_sum_squared += curr_value * curr_value;
            prev_sum_squared += prev_value * prev_value;
            curr_prev_sum += curr_value * prev_value;
          }
        }

        const float curr_mean = curr_sum / size;
        const float prev_mean = prev_sum / size;

        const float curr_var = cur_sum_squared / size - curr_mean * curr_mean;
        const float prev_var = prev_sum_squared / size - prev_mean * prev_mean;
        const float cov = curr_prev_sum / size - curr_mean * prev_mean;

        const float d1d2 = curr_var * prev_var;
        ncc_value = d1d2 < kFloatEpsilon ? 0 : cov / sqrt(d1d2);
      }

      if (ncc_value <= ncc_threshold || !isConverged) {
        status = false;
        return;
      }

      ncc = ncc_value;
    }

    if (level > 0) {
      xy.x = xy.x * 2.f - 0.5f;
      xy.y = xy.y * 2.f - 0.5f;

      previous_uv.x = previous_uv.x * 2.f - 0.5f;
      previous_uv.y = previous_uv.y * 2.f - 0.5f;
    }
  }

  track = xy;

  status = isPointInImage(track, currentImagePyramid.level_sizes[0]);
}

__global__ void st_build_cache_kernel(Pyramid previous_image, TrackData* tracks, PointCacheDataT* cache_data,
                                      size_t num_tracks) {
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= num_tracks) {
    return;
  }

  const float2& track = tracks[x].track;
  PointCacheDataT& data = cache_data[x];

  uint32_t& levels_mask = data.level_mask;
  int& cache_index = tracks[x].cache_index;

  int max_level = previous_image.levels;
  if (max_level == 0) {
    cache_index = -1;
    return;
  }

  int level = max_level - 1;

  const float scale = 1.f / static_cast<float>(1 << level);
  const float shift = 0.5f;
  float2 previous_uv = {(track.x - shift) * scale + shift, (track.y - shift) * scale + shift};

  levels_mask = 0;

  for (;; --level) {
    cudaTextureObject_t prev_image = previous_image.level_tex[level];
    Size level_size = previous_image.level_sizes[level];
    if (isPointInImage(previous_uv, level_size)) {
      float xmin = previous_uv.x - PATCH_HALF + 0.5f;
      float xmax = xmin + PATCH_DIM;
      float ymin = previous_uv.y - PATCH_HALF + 0.5f;
      float ymax = ymin + PATCH_DIM;

      bool good_patch = true;
      if (xmin < 0 || xmax >= level_size.width || ymin < 0 || ymax >= level_size.height) {
        good_patch = false;  // the patch bound box is out of the image
      }

      if (good_patch) {
        float patch[PATCH_SIZE];
        float patch_sum = 0.f;
#pragma unroll
        for (int iy = 0; iy < PATCH_DIM; iy++) {
#pragma unroll
          for (int ix = 0; ix < PATCH_DIM; ix++) {
            int patch_idx = iy * PATCH_DIM + ix;
            patch[patch_idx] = tex2D<float>(prev_image, xmin + ix, ymin + iy);
            patch_sum += patch[patch_idx];
          }
        }

        void* ptr = (void*)(data.patch + PATCH_SIZE * level);
        memcpy(ptr, (void*)patch, PATCH_SIZE * sizeof(float));

        data.patch_sums[level] = patch_sum;

        levels_mask |= 1 << level;
      }
    }

    if (level > 0) {
      previous_uv = {
          previous_uv.x * 2.f - 0.5f,
          previous_uv.y * 2.f - 0.5f,
      };
    }
    if (level == 0) break;
  }
  cache_index = x;
}

cudaError_t st_track_with_cache(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid currentImagePyramid,
                                PointCacheDataT* cache_data, TrackData* track_data, int num_tracks,
                                unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations,
                                cudaStream_t stream) {
  size_t threads = WARP_SIZE;
  size_t blocks = (num_tracks + WARP_SIZE - 1) / WARP_SIZE;

  st_track_with_cache_kernel<<<blocks, threads, 0, stream>>>(currentGradXPyramid, currentGradYPyramid,
                                                             currentImagePyramid, cache_data, track_data, num_tracks,
                                                             n_shift_only_iterations, n_full_mapping_iterations);
  return cudaGetLastError();
}

cudaError_t st_build_cache(Pyramid previous_image, TrackData* tracks, PointCacheDataT* cache_data, size_t num_tracks,
                           cudaStream_t stream) {
  assert(previous_image.levels <= PYRAMID_LEVELS);

  size_t threads = WARP_SIZE;
  size_t blocks = (num_tracks + WARP_SIZE - 1) / WARP_SIZE;

  st_build_cache_kernel<<<blocks, threads, 0, stream>>>(previous_image, tracks, cache_data, num_tracks);
  return cudaGetLastError();
}

cudaError_t st_track(Pyramid currentGradXPyramid, Pyramid currentGradYPyramid, Pyramid prevImagePyramid,
                     Pyramid currentImagePyramid, TrackData* track_data, int num_tracks,
                     unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations, cudaStream_t stream) {
  size_t threads = WARP_SIZE;
  size_t blocks = (num_tracks + WARP_SIZE - 1) / WARP_SIZE;

  st_track_kernel<<<blocks, threads, 0, stream>>>(currentGradXPyramid, currentGradYPyramid, prevImagePyramid,
                                                  currentImagePyramid, track_data, num_tracks, n_shift_only_iterations,
                                                  n_full_mapping_iterations);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
