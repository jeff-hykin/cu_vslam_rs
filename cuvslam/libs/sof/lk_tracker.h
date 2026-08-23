
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
#include "sof/kernel_operator.h"

namespace cuvslam::sof {

class LKFeatureTracker : public IFeatureTracker {
protected:
  using FeaturePatch = ImageMatrixPatch<float, 9, 9>;

private:
  using PatchRowVector = Eigen::internal::plain_row_type<FeaturePatch>::type;
  using PatchColVector = Eigen::internal::plain_col_type<FeaturePatch>::type;

protected:
  const KernelOperator<FeaturePatch> gauss2DKernel_;
  static void computeTGradient(FeaturePatch& tPatch, const FeaturePatch& img1, const FeaturePatch& img2);
  static float ncc(const FeaturePatch& patch1, const FeaturePatch& patch2);

public:
  LKFeatureTracker();
  ~LKFeatureTracker() override = default;

  bool trackPoint(const GradientPyramidT&, const GradientPyramidT& current_image_gradients,
                  const ImagePyramidT& previous_image, const ImagePyramidT& current_image, const Vector2T& previous_uv,
                  Vector2T& current_uv, Matrix2T& current_info, float search_radius_px,
                  float ncc_threshold) const override;

private:
  virtual bool trackPoint(const GradientPyramidT& prevFrameGradPyramid, const ImagePyramidT& prevFrameImagePyramid,
                          const ImagePyramidT& currentFrameImagePyramid, Vector2T& track, const Vector2T& offset,
                          Matrix2T& info, float search_radius_px, float ncc_threshold) const;
};

// class LKTrackerHorizontal performs Lucas-Kanade tracking along
// horizontal lines assuming that L&R images are rectified

class LKTrackerHorizontal : public LKFeatureTracker {
public:
  ~LKTrackerHorizontal() override = default;

private:
  // Run LK iterations on a single level
  bool trackPoint(const GradientPyramidT& prevFrameGradPyramid, const ImagePyramidT& prevFrameImagePyramid,
                  const ImagePyramidT& currentFrameImagePyramid, Vector2T& track, const Vector2T& offset,
                  Matrix2T& info, float search_radius_px, float ncc_threshold) const override;

  bool isHorizontal() const override { return true; }
};

}  // namespace cuvslam::sof
