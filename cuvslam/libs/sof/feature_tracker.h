
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

#include "common/types.h"
#include "common/vector_2t.h"

#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"

namespace cuvslam::sof {

class IFeatureTracker {
public:
  virtual ~IFeatureTracker() = default;

  // Track single point.
  // current_uv must contain a valid initial guess.
  // Output information matrix is optional.
  // search_radius_px how good the initial guess is.
  //
  // Set search_radius_px to image max of the image dimensions to indicate maximal
  // uncertainty.
  //
  // For negative search radius the function must return false: it is an indication
  // that the feature is not visible.
  //
  // Note:
  // - Information matrix should be computed assuming i.i.d standard normal image noise.
  // - Information matrix depends on the range of pixel intensities (consequence of
  //   the previous note)
  virtual bool trackPoint(const GradientPyramidT& previous_image_gradient,
                          const GradientPyramidT& current_image_gradient, const ImagePyramidT& previous_image,
                          const ImagePyramidT& current_image, const Vector2T& previous_uv, Vector2T& current_uv,
                          Matrix2T& current_info, float search_radius_px = 2048.f,
                          float ncc_threshold = 0.8f) const = 0;

  virtual bool isHorizontal() const { return false; }
};

}  // namespace cuvslam::sof
