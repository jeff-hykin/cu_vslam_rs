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

#include "cuda_modules/cuda_kernels/cuda_common.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

namespace {

__global__ void unproject_depth_kernel(cudaTextureObject_t depth_tex, float2 focal, float2 principal, int2 image_size,
                                       float depth_min, float depth_max, float3* out_points, int* out_count,
                                       int max_output, int stride) {
  const int grid_w = (image_size.x + stride - 1) / stride;
  const int grid_h = (image_size.y + stride - 1) / stride;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= grid_w * grid_h) {
    return;
  }

  const int px = (idx % grid_w) * stride + stride / 2;
  const int py = (idx / grid_w) * stride + stride / 2;
  if (px >= image_size.x || py >= image_size.y) {
    return;
  }

  const float d = tex2D<float>(depth_tex, static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f);
  if (d < depth_min || d > depth_max) {
    return;
  }

  const float inv_fx = __fdividef(1.f, focal.x);
  const float inv_fy = __fdividef(1.f, focal.y);
  // Pinhole backprojection: x = (px - cx)/fx * d, y = (py - cy)/fy * d, z = d
  const float x = (static_cast<float>(px) - principal.x) * inv_fx * d;
  const float y = (static_cast<float>(py) - principal.y) * inv_fy * d;

  const int write_idx = atomicAdd(out_count, 1);
  if (write_idx < max_output) {
    out_points[write_idx] = make_float3(x, y, d);
  }
}

// Count inliers for a given plane among active (non-removed) points.
// When gather_indices is non-null, also writes inlier point indices into that buffer.
// When gather_positions is non-null, writes the corresponding active-set position (tid)
// into that buffer in the same slot order, so callers can later mark removals on
// keep-flag arrays indexed by active-set position without a CPU-side reverse map.
__global__ void count_inliers_kernel(const float3* points, const int* active_indices, int num_active, float3 plane_n,
                                     float plane_d, float thresh, int* out_count, int* gather_indices,
                                     int* gather_positions, int max_gather) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_active) {
    return;
  }

  const int pt_idx = active_indices[tid];
  const float3 p = points[pt_idx];
  // Point-to-plane distance: |n . p + d| for plane equation n . x + d = 0
  const float dist = fabsf(plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z + plane_d);

  if (dist < thresh) {
    const int slot = atomicAdd(out_count, 1);
    if (slot < max_gather) {
      if (gather_indices) {
        gather_indices[slot] = pt_idx;
      }
      if (gather_positions) {
        gather_positions[slot] = tid;
      }
    }
  }
}

// Gather points: out[i] = points[indices[i]] for i < num.
__global__ void gather_inlier_points_kernel(const float3* points, const int* indices, int num, float3* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num) {
    return;
  }
  out[i] = points[indices[i]];
}

// Stride-subsample: out[i] = in[i * stride] for i < n_out.  Deterministic and
// preserves spatial diversity when the input is unprojected from a regular
// pixel grid (the upstream RANSAC depth pyramid).
__global__ void stride_subsample_indices_kernel(const int* in, int* out, int stride, int n_out, int n_in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_out) {
    return;
  }
  const int src = i * stride;
  out[i] = (src < n_in) ? in[src] : in[n_in - 1];
}

// Scatter zeros into keep_flags at the given active-set positions: marks those
// active entries for removal during the next compact_active pass.
__global__ void mark_positions_kernel(uint8_t* keep_flags, const int* positions, int num_positions) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_positions) {
    return;
  }
  keep_flags[positions[i]] = 0;
}

// Batched RANSAC: one block per hypothesis, threads cooperate to count inliers.
// Uses warp-shuffle reduction -- no shared memory needed beyond a small per-warp buffer.
__global__ void ransac_batch_kernel(const float3* points, const int* active_indices, int num_active,
                                    const float4* hypotheses, float thresh, int* out_counts, int num_hypotheses) {
  const int hyp_idx = blockIdx.x;
  if (hyp_idx >= num_hypotheses) {
    return;
  }

  const float4 h = hypotheses[hyp_idx];
  const float nx = h.x, ny = h.y, nz = h.z, d = h.w;

  int local_count = 0;
  for (int i = threadIdx.x; i < num_active; i += blockDim.x) {
    const int pt_idx = active_indices[i];
    const float3 p = points[pt_idx];
    const float dist = fabsf(nx * p.x + ny * p.y + nz * p.z + d);
    if (dist < thresh) {
      ++local_count;
    }
  }

  // Two-level parallel reduction: first within each warp via __shfl_down_sync,
  // then across warps via shared memory. Lane 0 of each warp stores its partial
  // sum; the first warp then reduces those partial sums.
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffff, local_count, offset);
  }

  // Lane 0 of each warp writes to shared memory
  __shared__ int warp_counts[32];
  const int lane = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    warp_counts[warp_id] = local_count;
  }
  __syncthreads();

  // First warp reduces across warps
  if (warp_id == 0) {
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    int val = (lane < num_warps) ? warp_counts[lane] : 0;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
      out_counts[hyp_idx] = val;
    }
  }
}

// Compute centroid and upper-triangle covariance of inlier points in a single pass.
// Uses warp-shuffle + shared memory two-level reduction (same pattern as ransac_batch_kernel).
// Output: out_centroid_cov[0..2] = centroid (sum_x, sum_y, sum_z),
//         out_centroid_cov[3..8] = cov upper triangle (xx, xy, xz, yy, yz, zz).
// Caller divides by num_inliers to get mean, and computes cov -= mean*mean^T.
__global__ void compute_plane_stats_kernel(const float3* points, const int* inlier_indices, int num_inliers,
                                           float* out_centroid_cov) {
  float sx = 0.f, sy = 0.f, sz = 0.f;
  float cxx = 0.f, cxy = 0.f, cxz = 0.f, cyy = 0.f, cyz = 0.f, czz = 0.f;

  // Accumulate sum(p_i) and upper triangle of sum(p_i * p_i^T) for PCA.
  // Caller computes: centroid = sum/N, cov_ij = raw_ij/N - centroid_i * centroid_j.
  // The smallest eigenvector of cov is the plane normal (least-variance direction).
  for (int i = threadIdx.x; i < num_inliers; i += blockDim.x) {
    const float3 p = points[inlier_indices[i]];
    sx += p.x;
    sy += p.y;
    sz += p.z;
    cxx += p.x * p.x;
    cxy += p.x * p.y;
    cxz += p.x * p.z;
    cyy += p.y * p.y;
    cyz += p.y * p.z;
    czz += p.z * p.z;
  }

  // Warp-level reduction for all 9 accumulators
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sx += __shfl_down_sync(0xffffffff, sx, offset);
    sy += __shfl_down_sync(0xffffffff, sy, offset);
    sz += __shfl_down_sync(0xffffffff, sz, offset);
    cxx += __shfl_down_sync(0xffffffff, cxx, offset);
    cxy += __shfl_down_sync(0xffffffff, cxy, offset);
    cxz += __shfl_down_sync(0xffffffff, cxz, offset);
    cyy += __shfl_down_sync(0xffffffff, cyy, offset);
    cyz += __shfl_down_sync(0xffffffff, cyz, offset);
    czz += __shfl_down_sync(0xffffffff, czz, offset);
  }

  __shared__ float warp_buf[32 * 9];
  const int lane = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    warp_buf[warp_id * 9 + 0] = sx;
    warp_buf[warp_id * 9 + 1] = sy;
    warp_buf[warp_id * 9 + 2] = sz;
    warp_buf[warp_id * 9 + 3] = cxx;
    warp_buf[warp_id * 9 + 4] = cxy;
    warp_buf[warp_id * 9 + 5] = cxz;
    warp_buf[warp_id * 9 + 6] = cyy;
    warp_buf[warp_id * 9 + 7] = cyz;
    warp_buf[warp_id * 9 + 8] = czz;
  }
  __syncthreads();

  if (warp_id == 0) {
    const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    float v[9];
    for (int k = 0; k < 9; ++k) {
      v[k] = (lane < num_warps) ? warp_buf[lane * 9 + k] : 0.f;
    }
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      for (int k = 0; k < 9; ++k) {
        v[k] += __shfl_down_sync(0xffffffff, v[k], offset);
      }
    }
    if (lane == 0) {
      for (int k = 0; k < 9; ++k) {
        out_centroid_cov[k] = v[k];
      }
    }
  }
}

// Generate plane hypotheses from random index triplets.
// Each thread processes one triplet: reads 3 points, computes cross product -> plane normal + d.
// Degenerate triplets (near-zero cross product) produce a zero-length normal (w=0).
__global__ void generate_hypotheses_kernel(const float3* points, const int* active_indices, const int3* triplets,
                                           float4* hypotheses, int num_hypotheses) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_hypotheses) {
    return;
  }

  const int3 tri = triplets[tid];
  const float3 p0 = points[active_indices[tri.x]];
  const float3 p1 = points[active_indices[tri.y]];
  const float3 p2 = points[active_indices[tri.z]];

  // Plane from 3 points: n = normalize((p1-p0) x (p2-p0)), d = -n . p0.
  // Degenerate (collinear) triplets produce |n| < 1e-8 and are zeroed out.
  const float3 v01 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
  const float3 v02 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
  float3 n = {v01.y * v02.z - v01.z * v02.y, v01.z * v02.x - v01.x * v02.z, v01.x * v02.y - v01.y * v02.x};
  const float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
  if (len < 1e-8f) {
    hypotheses[tid] = {0.f, 0.f, 0.f, 0.f};
    return;
  }
  n.x /= len;
  n.y /= len;
  n.z /= len;
  hypotheses[tid] = {n.x, n.y, n.z, -(n.x * p0.x + n.y * p0.y + n.z * p0.z)};
}

// Compact active indices: given a flags array (1=keep, 0=remove), write surviving
// indices from active_in to active_out via atomicAdd on out_count.
__global__ void compact_active_kernel(const int* active_in, const uint8_t* keep_flags, int num_active, int* active_out,
                                      int* out_count) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_active) {
    return;
  }
  if (keep_flags[tid]) {
    const int slot = atomicAdd(out_count, 1);
    active_out[slot] = active_in[tid];
  }
}

// Fill out[i] = i for i < n.  Trivial GPU iota, used to seed the RANSAC active-index
// array directly on the device so the host doesn't need a 76800-element loop + copy.
__global__ void fill_iota_kernel(int* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = i;
}

}  // namespace

cudaError_t unproject_depth_points(cudaTextureObject_t depth_tex, float2 focal, float2 principal, int2 image_size,
                                   float depth_min, float depth_max, float3* out_points, int* out_count, int max_output,
                                   int stride, cudaStream_t stream) {
  const int grid_w = (image_size.x + stride - 1) / stride;
  const int grid_h = (image_size.y + stride - 1) / stride;
  const int num_samples = grid_w * grid_h;
  const int threads = 256;
  const int blocks = (num_samples + threads - 1) / threads;

  unproject_depth_kernel<<<blocks, threads, 0, stream>>>(depth_tex, focal, principal, image_size, depth_min, depth_max,
                                                         out_points, out_count, max_output, stride);
  return cudaGetLastError();
}

cudaError_t count_plane_inliers(const float3* points, const int* active_indices, int num_active, float3 plane_n,
                                float plane_d, float thresh, int* out_count, int* gather_indices, int* gather_positions,
                                int max_gather, cudaStream_t stream) {
  if (num_active == 0) {
    return cudaSuccess;
  }

  const int threads = 256;
  const int blocks = (num_active + threads - 1) / threads;

  count_inliers_kernel<<<blocks, threads, 0, stream>>>(points, active_indices, num_active, plane_n, plane_d, thresh,
                                                       out_count, gather_indices, gather_positions, max_gather);
  return cudaGetLastError();
}

cudaError_t ransac_batch_evaluate(const float3* points, const int* active_indices, int num_active,
                                  const float4* hypotheses, float thresh, int* out_counts, int num_hypotheses,
                                  cudaStream_t stream) {
  if (num_hypotheses == 0 || num_active == 0) {
    return cudaSuccess;
  }

  const int threads = 256;
  const int blocks = num_hypotheses;

  ransac_batch_kernel<<<blocks, threads, 0, stream>>>(points, active_indices, num_active, hypotheses, thresh,
                                                      out_counts, num_hypotheses);
  return cudaGetLastError();
}

cudaError_t compute_plane_stats(const float3* points, const int* inlier_indices, int num_inliers,
                                float* out_centroid_cov, cudaStream_t stream) {
  if (num_inliers == 0) {
    return cudaSuccess;
  }

  const int threads = 256;
  compute_plane_stats_kernel<<<1, threads, 0, stream>>>(points, inlier_indices, num_inliers, out_centroid_cov);
  return cudaGetLastError();
}

cudaError_t generate_hypotheses(const float3* points, const int* active_indices, const int3* triplets,
                                float4* hypotheses, int num_hypotheses, cudaStream_t stream) {
  if (num_hypotheses == 0) {
    return cudaSuccess;
  }

  const int threads = 256;
  const int blocks = (num_hypotheses + threads - 1) / threads;
  generate_hypotheses_kernel<<<blocks, threads, 0, stream>>>(points, active_indices, triplets, hypotheses,
                                                             num_hypotheses);
  return cudaGetLastError();
}

cudaError_t compact_active_indices(const int* active_in, const uint8_t* keep_flags, int num_active, int* active_out,
                                   int* out_count, cudaStream_t stream) {
  if (num_active == 0) {
    return cudaSuccess;
  }

  const int threads = 256;
  const int blocks = (num_active + threads - 1) / threads;
  compact_active_kernel<<<blocks, threads, 0, stream>>>(active_in, keep_flags, num_active, active_out, out_count);
  return cudaGetLastError();
}

cudaError_t gather_inlier_points(const float3* points, const int* indices, int num, float3* out, cudaStream_t stream) {
  if (num <= 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = (num + threads - 1) / threads;
  gather_inlier_points_kernel<<<blocks, threads, 0, stream>>>(points, indices, num, out);
  return cudaGetLastError();
}

cudaError_t stride_subsample_indices(const int* in, int* out, int stride, int n_out, int n_in, cudaStream_t stream) {
  if (n_out <= 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = (n_out + threads - 1) / threads;
  stride_subsample_indices_kernel<<<blocks, threads, 0, stream>>>(in, out, stride, n_out, n_in);
  return cudaGetLastError();
}

cudaError_t mark_positions_for_removal(uint8_t* keep_flags, const int* positions, int num_positions,
                                       cudaStream_t stream) {
  if (num_positions <= 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = (num_positions + threads - 1) / threads;
  mark_positions_kernel<<<blocks, threads, 0, stream>>>(keep_flags, positions, num_positions);
  return cudaGetLastError();
}

cudaError_t fill_iota(int* out, int n, cudaStream_t stream) {
  if (n <= 0) {
    return cudaSuccess;
  }
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fill_iota_kernel<<<blocks, threads, 0, stream>>>(out, n);
  return cudaGetLastError();
}

}  // namespace cuvslam::cuda
