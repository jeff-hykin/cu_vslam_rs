
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

#include "common/image_matrix.h"

#include "sof/feature_tracker.h"
#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"

namespace cuvslam::sof {

using STFeaturePatch = Eigen::Matrix<float, 9, 9, Eigen::DontAlign | Eigen::RowMajor>;

class STTracker : public IFeatureTracker {
public:
  STTracker(unsigned n_shift_only_iterations, unsigned n_full_mapping_iterations);
  ~STTracker() override = default;

  bool trackPoint(const GradientPyramidT& previous_image_gradients, const GradientPyramidT& current_image_gradients,
                  const ImagePyramidT& previous_image, const ImagePyramidT& current_image, const Vector2T& previous_uv,
                  Vector2T& current_uv, Matrix2T& current_info, float search_radius_px,
                  float ncc_threshold) const override;

public:
  static bool BuildPointCache(const ImagePyramidT& previous_image, const Vector2T& track, uint32_t& levels_mask,
                              std::vector<STFeaturePatch>& image_patches);

  bool TrackPointWithCache(const GradientPyramidT& current_image_gradients, const ImagePyramidT& current_image,
                           uint32_t levels_mask, uint32_t image_patches_size, const STFeaturePatch* image_patches,
                           const float previous_uv[2], float current_uv[2], float& ncc, float current_info[4],
                           float search_radius_px, float ncc_threshold = 0.8f,
                           const Matrix2T& initial_guess_map = Matrix2T::Identity()) const;

private:
  const unsigned n_shift_only_iterations;
  const unsigned n_full_mapping_iterations;
};

}  // namespace cuvslam::sof
