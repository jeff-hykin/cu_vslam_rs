
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

#include "math/ransac.h"

namespace test::math {

struct LineTest {
  cuvslam::Vector2T first;
  cuvslam::Vector2T second;

  LineTest(const cuvslam::Vector2T& f, const cuvslam::Vector2T& s) : first(f), second(s) {}
  LineTest() = default;

  float distance(const cuvslam::Vector2T& pt) const {
    cuvslam::Vector2T diff = second - first;
    float denom = diff.norm();

    if (denom == 0) {
      return (second - pt).norm();
    }

    float nomin = std::fabs(diff.y() * pt.x() - diff.x() * pt.y() + second.x() * first.y() - second.y() * first.x());
    return nomin / denom;
  }
};

using InputType = std::pair<cuvslam::Vector2T, cuvslam::Vector2T>;

class LineTestRansacImpl : public cuvslam::math::HypothesisBase<float, InputType, LineTest, 1> {
  using Base = cuvslam::math::HypothesisBase<float, InputType, LineTest, 1>;
  float threshhold_ = 0;

public:
  void setThreshold(float t) {
    assert(t > 0);
    threshhold_ = t;
  }

protected:
  // Required method for RANSAC template
  template <typename _ItType>
  bool evaluate(LineTest& result, const _ItType beginIt, _ItType) const {
    result.first = cuvslam::Vector2T::Zero();
    result.second = static_cast<const InputType&>(*beginIt).first;
    return true;
  }

  // Required method for RANSAC template
  template <typename _ItType>
  size_t countInliers(const LineTest& result, const _ItType beginIt, const _ItType endIt) const {
    assert(threshhold_ > 0);
    size_t inliers = 0;
    for_each(beginIt, endIt, [&](const InputType& i) { inliers += (result.distance(i.first) < threshhold_); });
    return inliers;
  }
};

}  // namespace test::math
