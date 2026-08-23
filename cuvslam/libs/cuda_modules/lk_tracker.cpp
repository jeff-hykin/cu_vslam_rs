
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

#include "cuda_modules/lk_tracker.h"

namespace cuvslam::cuda {

GPULKFeatureTracker::GPULKFeatureTracker() { CUDA_CHECK(init_gauss_coeffs()); }

bool GPULKFeatureTracker::track_points(const GPUGradientPyramid& previous_image_gradients, const GPUGradientPyramid&,
                                       const GaussianGPUImagePyramid& previous_image,
                                       const GaussianGPUImagePyramid& current_image,
                                       GPUArrayPinned<TrackData>& track_data, size_t num_tracks, cudaStream_t& stream) {
  if (num_tracks == 0) {
    return true;
  }
  Pyramid prevFrameGradXPyramid;
  Pyramid prevFrameGradYPyramid;
  Pyramid prevFrameImagePyramid;
  Pyramid currentFrameImagePyramid;

  auto& prev_grad_x = previous_image_gradients.gradX();
  auto& prev_grad_y = previous_image_gradients.gradY();

  size_t levels = previous_image_gradients.getLevelsCount();

  prevFrameGradXPyramid.levels = levels;
  prevFrameGradYPyramid.levels = levels;
  prevFrameImagePyramid.levels = levels;
  currentFrameImagePyramid.levels = levels;
  for (size_t i = 0; i < levels; i++) {
    const GPUImageT& level_grad_x = prev_grad_x[i];
    prevFrameGradXPyramid.level_tex[i] = level_grad_x.get_texture_filter_linear();
    prevFrameGradXPyramid.level_sizes[i] = {static_cast<int>(level_grad_x.cols()),
                                            static_cast<int>(level_grad_x.rows())};

    const GPUImageT& level_grad_y = prev_grad_y[i];
    prevFrameGradYPyramid.level_tex[i] = level_grad_y.get_texture_filter_linear();
    prevFrameGradYPyramid.level_sizes[i] = {static_cast<int>(level_grad_y.cols()),
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

  cudaError_t error = lk_track(prevFrameGradXPyramid, prevFrameGradYPyramid, prevFrameImagePyramid,
                               currentFrameImagePyramid, track_data.ptr(), num_tracks, stream);

  CUDA_CHECK(error);

  return (error == cudaSuccess);
}

bool GPULKTrackerHorizontal::track_points(const GPUGradientPyramid& previous_image_gradients, const GPUGradientPyramid&,
                                          const GaussianGPUImagePyramid& previous_image,
                                          const GaussianGPUImagePyramid& current_image,
                                          GPUArrayPinned<TrackData>& track_data, size_t num_tracks,
                                          cudaStream_t& stream) {
  if (num_tracks == 0) {
    return true;
  }
  Pyramid prevFrameGradXPyramid;
  Pyramid prevFrameImagePyramid;
  Pyramid currentFrameImagePyramid;

  auto& prev_grad_x = previous_image_gradients.gradX();

  size_t levels = previous_image_gradients.getLevelsCount();
  prevFrameGradXPyramid.levels = levels;
  prevFrameImagePyramid.levels = levels;
  currentFrameImagePyramid.levels = levels;
  for (size_t i = 0; i < levels; i++) {
    const GPUImageT& level_grad_x = prev_grad_x[i];
    prevFrameGradXPyramid.level_tex[i] = level_grad_x.get_texture_filter_linear();
    prevFrameGradXPyramid.level_sizes[i] = {static_cast<int>(level_grad_x.cols()),
                                            static_cast<int>(level_grad_x.rows())};

    const GPUImageT& level_prev_img = previous_image[i];
    prevFrameImagePyramid.level_tex[i] = level_prev_img.get_texture_filter_linear();
    prevFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_prev_img.cols()),
                                            static_cast<int>(level_prev_img.rows())};

    const GPUImageT& level_cur_img = current_image[i];
    currentFrameImagePyramid.level_tex[i] = level_cur_img.get_texture_filter_linear();
    currentFrameImagePyramid.level_sizes[i] = {static_cast<int>(level_cur_img.cols()),
                                               static_cast<int>(level_cur_img.rows())};
  }
  cudaError_t error = lk_track_horizontal(prevFrameGradXPyramid, prevFrameImagePyramid, currentFrameImagePyramid,
                                          track_data.ptr(), num_tracks, stream);

  CUDA_CHECK(error);
  return (error == cudaSuccess);
}

}  // namespace cuvslam::cuda
