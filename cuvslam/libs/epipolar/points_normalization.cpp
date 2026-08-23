
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

#include "epipolar/points_normalization.h"

#include <algorithm>

#include "common/statistic.h"
#include "common/types.h"
#include "common/vector_2t.h"

namespace cuvslam::epipolar {

NormalizationTransform::NormalizationTransform(const Vector2TVector &srcPoints, float minDistLimit) {
  const size_t nPoints = srcPoints.size();
  const float fnPoints = static_cast<float>(nPoints);
  (void)fnPoints;
  assert(fnPoints > 0);

  auto statVar = std::for_each(srcPoints.cbegin(), srcPoints.cend(), Statistical<Vector2T>());
  assert(statVar.count() == srcPoints.size());

  const float radius = std::sqrt(statVar.variance().sum());

  if (radius > minDistLimit) {
    scale_ = std::sqrt(2.f) / radius;
    mean_ = statVar.mean();
  }
}

bool NormalizationTransform::isValid() const { return scale_ != 0; }

Matrix3T NormalizationTransform::calcNormMatrix() const {
  return (Eigen::UniformScaling<float>(scale_) * Eigen::Translation<float, 2>(-mean_)).matrix();
}

Matrix3T NormalizationTransform::calcDenormMatrix() const {
  assert(scale_ != 0);
  return (Eigen::Translation<float, 2>(mean_) * Eigen::UniformScaling<float>(1.f / scale_)).matrix();
}

Vector2T NormalizationTransform::operator()(const Vector2T &src) const { return (src - mean_) * scale_; }

}  // namespace cuvslam::epipolar
