
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

#include "cuda_modules/st_tracker.h"

namespace cuvslam::cuda {

GPUSTFeatureTracker::GPUSTFeatureTracker(unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations)
    : n_shift_only_iterations_(n_shift_only_iterations), n_full_mapping_iterations_(n_full_mapping_iterations) {
  CUDA_CHECK(init_st_tracker());
}

bool GPUSTFeatureTracker::track_points(const GPUGradientPyramid&, const GPUGradientPyramid& current_image_gradients,
                                       const GaussianGPUImagePyramid& previous_image,
                                       const GaussianGPUImagePyramid& current_image,
                                       GPUArrayPinned<TrackData>& track_data, size_t num_tracks, cudaStream_t& stream) {
  if (num_tracks == 0) {
    return true;
  }
  Pyramid currentFrameGradXPyramid;
  Pyramid currentFrameGradYPyramid;
  Pyramid prevFrameImagePyramid;
  Pyramid currentFrameImagePyramid;

  auto& curr_grad_x = current_image_gradients.gradX();
  auto& curr_grad_y = current_image_gradients.gradY();

  int levels = current_image_gradients.getLevelsCount();

  currentFrameGradXPyramid.levels = levels;
  currentFrameGradYPyramid.levels = levels;
  prevFrameImagePyramid.levels = levels;
  currentFrameImagePyramid.levels = levels;
  for (int i = 0; i < levels; i++) {
    const GPUImageT& level_grad_x = curr_grad_x[i];
    currentFrameGradXPyramid.level_tex[i] = level_grad_x.get_texture_filter_linear();
    currentFrameGradXPyramid.level_sizes[i] = {static_cast<int>(level_grad_x.cols()),
                                               static_cast<int>(level_grad_x.rows())};

    const GPUImageT& level_grad_y = curr_grad_y[i];
    currentFrameGradYPyramid.level_tex[i] = level_grad_y.get_texture_filter_linear();
    currentFrameGradYPyramid.level_sizes[i] = {static_cast<int>(level_grad_y.cols()),
                                               static_cast<int>(level_grad_y.rows())};

    const GPUImageT& level_prev_img = previous_image[i];
    prevFrameImagePyramid.level_tex[i] = level_prev_img.get_texture_filter_linear();
    prevFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_prev_img.cols()),
                                            static_cast<int>(level_prev_img.rows())};

    const GPUImageT& level_cur_img = current_image[i];
    currentFrameImagePyramid.level_tex[i] = level_cur_img.get_texture_filter_linear();
    currentFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_cur_img.cols()),
                                               static_cast<int>(level_cur_img.rows())};
  }

  cudaError_t error =
      st_track(currentFrameGradXPyramid, currentFrameGradYPyramid, prevFrameImagePyramid, currentFrameImagePyramid,
               track_data.ptr(), num_tracks, n_shift_only_iterations_, n_full_mapping_iterations_, stream);

  CUDA_CHECK(error);

  return (error == cudaSuccess);
}

GPUSTFeatureTrackerWithCache::GPUSTFeatureTrackerWithCache(unsigned n_shift_only_iterations,
                                                           unsigned n_full_mapping_iterations)
    : n_shift_only_iterations_(n_shift_only_iterations), n_full_mapping_iterations_(n_full_mapping_iterations) {
  CUDA_CHECK(init_st_tracker());
}

bool GPUSTFeatureTrackerWithCache::build_points_cache(const GaussianGPUImagePyramid& previous_image,
                                                      const GPUArray<TrackData>& tracks,
                                                      GPUArray<PointCacheDataT>& cache_data, size_t num_tracks,
                                                      cudaStream_t& stream) {
  if (num_tracks == 0) {
    return true;
  }

  Pyramid prevFrameImagePyramid;
  prevFrameImagePyramid.levels = previous_image.getLevelsCount();

  for (int i = 0; i < prevFrameImagePyramid.levels; i++) {
    const GPUImageT& level_prev_img = previous_image[i];
    prevFrameImagePyramid.level_tex[i] = level_prev_img.get_texture_filter_linear();
    prevFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_prev_img.cols()),
                                            static_cast<int>(level_prev_img.rows())};
  }

  cudaError_t error = st_build_cache(prevFrameImagePyramid, tracks.ptr(), cache_data.ptr(), num_tracks, stream);

  CUDA_CHECK(error);

  return (error == cudaSuccess);
}

bool GPUSTFeatureTrackerWithCache::track_points_with_cache(const GPUGradientPyramid& current_image_gradients,
                                                           const GaussianGPUImagePyramid& current_image,
                                                           GPUArray<TrackData>& track_data,
                                                           GPUArray<PointCacheDataT>& cache_data, size_t num_tracks,
                                                           cudaStream_t& stream) {
  if (num_tracks == 0) {
    return true;
  }
  Pyramid currentFrameGradXPyramid;
  Pyramid currentFrameGradYPyramid;
  Pyramid currentFrameImagePyramid;

  auto& curr_grad_x = current_image_gradients.gradX();
  auto& curr_grad_y = current_image_gradients.gradY();

  int levels = current_image_gradients.getLevelsCount();
  assert(levels <= PYRAMID_LEVELS);

  currentFrameGradXPyramid.levels = levels;
  currentFrameGradYPyramid.levels = levels;
  currentFrameImagePyramid.levels = levels;
  for (int i = 0; i < levels; i++) {
    const GPUImageT& level_grad_x = curr_grad_x[i];
    currentFrameGradXPyramid.level_tex[i] = level_grad_x.get_texture_filter_linear();
    currentFrameGradXPyramid.level_sizes[i] = {static_cast<int>(level_grad_x.cols()),
                                               static_cast<int>(level_grad_x.rows())};

    const GPUImageT& level_grad_y = curr_grad_y[i];
    currentFrameGradYPyramid.level_tex[i] = level_grad_y.get_texture_filter_linear();
    currentFrameGradYPyramid.level_sizes[i] = {static_cast<int>(level_grad_y.cols()),
                                               static_cast<int>(level_grad_y.rows())};

    const GPUImageT& level_cur_img = current_image[i];
    currentFrameImagePyramid.level_tex[i] = level_cur_img.get_texture_filter_linear();
    currentFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_cur_img.cols()),
                                               static_cast<int>(level_cur_img.rows())};
  }

  cudaError_t error = st_track_with_cache(currentFrameGradXPyramid, currentFrameGradYPyramid, currentFrameImagePyramid,
                                          cache_data.ptr(), track_data.ptr(), num_tracks, n_shift_only_iterations_,
                                          n_full_mapping_iterations_, stream);

  CUDA_CHECK(error);

  return (error == cudaSuccess);
}

}  // namespace cuvslam::cuda
