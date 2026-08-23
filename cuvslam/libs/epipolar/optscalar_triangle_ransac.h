
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
#include "math/ransac.h"

namespace cuvslam::epipolar {

class OptimalScalarTriangle : public math::HypothesisBase<float, float, float, 1, 1> {
public:
  // threshold_ - threshold for deviation of the ration of triangulated in adjacent cameras pairs points Z (in the
  // camera space of the middle frame of 3 frames)
  void setOptions(const float t = 1.0f) { threshold_ = t; }

  OptimalScalarTriangle(const float t = 1.0f) : threshold_(t) {}

  float getTheshold() { return threshold_; }

private:
  float threshold_ = 0;

protected:
  // Required method for Ransac, called by Ransac operator()
  template <typename _ItType>
  bool evaluate(float& ScaleRatio, _ItType beginIt, _ItType /*endIt*/) const {
    ScaleRatio = *beginIt;
    return true;
  }

  // Required method for Ransac, called by main Ransac method.
  template <typename _ItType>
  size_t countInliers(const float& ScaleRatio, const _ItType beginIt, const _ItType endIt) const {
    size_t inliers = 0;
    assert(threshold_ > 0);

    for_each(beginIt, endIt, [&](const float& i) { inliers += (std::abs(i - ScaleRatio) < threshold_) ? 1 : 0; });
    return inliers;
  }
};

}  // namespace cuvslam::epipolar
