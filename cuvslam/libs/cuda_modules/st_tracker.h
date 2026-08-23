
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

#pragma once

#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/image_pyramid.h"

namespace cuvslam::cuda {

template <typename T, int Dim, int MaxPoints>
class GPUPatchData {
public:
  GPUPatchData() { CUDA_CHECK(cudaMalloc((void**)&data_, MaxPoints * sizeof(PointCacheData<T, Dim>))); }
  ~GPUPatchData() { CUDA_CHECK(cudaFree(data_)); }

  PointCacheData<T, Dim>* ptr() const { return data_; }

private:
  PointCacheData<T, Dim>* data_ = nullptr;
};

template <int MaxPoints>
using GPUPatchDataT = GPUPatchData<float, 9, MaxPoints>;

class GPUSTFeatureTracker {
public:
  GPUSTFeatureTracker(unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations);

  virtual bool track_points(const GPUGradientPyramid& previous_image_gradients,
                            const GPUGradientPyramid& current_image_gradients,
                            const GaussianGPUImagePyramid& previous_image, const GaussianGPUImagePyramid& current_image,
                            GPUArrayPinned<TrackData>& track_data, size_t num_tracks, cudaStream_t& stream);

protected:
  unsigned n_shift_only_iterations_;
  unsigned n_full_mapping_iterations_;
};

class GPUSTFeatureTrackerWithCache {
public:
  GPUSTFeatureTrackerWithCache(unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations);

  bool build_points_cache(const GaussianGPUImagePyramid& previous_image, const GPUArray<TrackData>& tracks,
                          GPUArray<PointCacheDataT>& cache_data, size_t num_tracks, cudaStream_t& stream);

  bool track_points_with_cache(const GPUGradientPyramid& current_image_gradients,
                               const GaussianGPUImagePyramid& current_image, GPUArray<TrackData>& track_data,
                               GPUArray<PointCacheDataT>& cache_data, size_t num_tracks, cudaStream_t& stream);

protected:
  unsigned n_shift_only_iterations_;
  unsigned n_full_mapping_iterations_;
};

}  // namespace cuvslam::cuda
