
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
#include "sof/image_pyramid_float.h"
#include "sof/kernel_operator.h"

namespace cuvslam::sof {

/*
 This class implements a variant of LKT algorithm:
 https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker

 Key differences:
 - step control (LM as opposed to Gauss-Newton)
 - compensation for additive brightness changes:
   we minimize
     sum | f(x + u) - g(x) - 1/n sum (f(x + u) - g(x)) |^2
   as opposed to
     sum | f(x + u) - g(x) |^2

  We also estimate uncertainty in the position assuming i.i.d.
  standard normal image noise.
*/
class KLTTracker : public IFeatureTracker {
public:
  KLTTracker();
  ~KLTTracker() override = default;

  bool trackPoint(const GradientPyramidT& previous_image_gradients, const GradientPyramidT& current_image_gradients,
                  const ImagePyramidT& previous_image, const ImagePyramidT& current_image, const Vector2T& previous_uv,
                  Vector2T& current_uv, Matrix2T& current_info, float search_radius_px,
                  float ncc_threshold) const override;

private:
  using FeaturePatch = ImageMatrixPatch<float, 9, 9>;

  // Coarse-to-fine tracking
  bool track_point(const GradientPyramidT& current_image_gradient, const ImagePyramidT& previous_image,
                   const ImagePyramidT& current_image, Vector2T& track, const Vector2T& offset, Matrix2T& info,
                   float search_radius_px, float ncc_threshold) const;

  // Run KLT iterations on a single level
  virtual bool refine_position(const FeaturePatch& previous_patch, const GradientPyramidT& current_image_gradient,
                               const ImagePyramidT& current_image, int level, Vector2T& xy, Matrix2T& info) const;

  static void compute_residual(FeaturePatch& residual, const FeaturePatch& img1, const FeaturePatch& img2);
  static float compute_ncc(const FeaturePatch& patch1, const FeaturePatch& patch2);

protected:
  bool compute_cost(float* cost, FeaturePatch& current_patch, FeaturePatch& residual,
                    const ImagePyramidT& current_image, const FeaturePatch& previous_patch, const Vector2T& xy,
                    int level) const;

  // weights pixels within a patch
  const KernelOperator<FeaturePatch> weights_;
};

// class KLTTrackerHorizontal performs Lucas-Kanade tracking along
// horizontal lines assuming that L&R images are rectified

class KLTTrackerHorizontal : public KLTTracker {
public:
  ~KLTTrackerHorizontal() override = default;

private:
  using FeaturePatch = ImageMatrixPatch<float, 9, 9>;

  // Run KLT iterations on a single level
  bool refine_position(const FeaturePatch& previous_patch, const GradientPyramidT& current_image_gradient,
                       const ImagePyramidT& current_image, int level, Vector2T& xy, Matrix2T& info) const override;

  bool isHorizontal() const override { return true; }
};

}  // namespace cuvslam::sof
