
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

#include "math/test/ransac_test.h"

#include <random>

#include "common/include_gtest.h"

using namespace cuvslam;

namespace test::math {
using LineTestRansac = cuvslam::math::Ransac<LineTestRansacImpl>;

class RansacTest : public testing::Test {
protected:
  float r_random() {
    unsigned int r = static_cast<unsigned int>(gen_());
    return float(r) / float(std::numeric_limits<unsigned int>::max());
  }

  float r_random2(const float a, const float b) { return a + r_random() * (b - a); }

  cuvslam::math::DefaultRandomGenerator gen_;
};

TEST_F(RansacTest, RansacLineTest) {
  const size_t sample_size = 235;
  float ang = r_random2(0, float(2 * PI));
  InputType v({std::sin(ang), std::cos(ang)}, {});
  LineTest groundTruth(Vector2T::Zero(), v.first);

  const float noise = .02f;
  std::vector<InputType> data;

  for (size_t i = 0; i < sample_size; i++) {
    data.push_back({v.first * r_random2(-1, 1), Vector2T::Zero()});
    InputType outlierSample = {Vector2T(r_random2(-1, 1), r_random2(-1, 1)), Vector2T::Zero()};

    if (groundTruth.distance(outlierSample.first) <= noise) {
      outlierSample.first += Vector2T(-v.first.y(), v.first.x());  // add point from orthogonal to ground truth line
    }

    data.push_back(outlierSample);
  }

  std::vector<InputType> dataCopy(data);
  std::shuffle(dataCopy.begin(), dataCopy.end(), gen_);

  LineTest bestFit(Vector2T::Zero(), Vector2T::Zero());
  LineTestRansac rtd;
  rtd.setConfidence(0.995f);
  rtd.setThreshold(0.01f);
  const size_t numIterations = rtd(bestFit, dataCopy.cbegin(), dataCopy.cend());
  EXPECT_TRUE(numIterations > 0);

  // outliers should contain all odd numbers
  int indx = 0;
  std::vector<int> outliers;

  for (auto dit = data.begin(); dit != data.end(); dit++, indx++) {
    if (bestFit.distance(dit->first) > noise) {
      EXPECT_EQ(1, indx % 2);
      outliers.push_back(indx);
    }
  }

  EXPECT_EQ(sample_size, outliers.size());

  Vector2T checkIt(bestFit.second - bestFit.first);
  checkIt.normalize();
  v.first.normalize();
  float correlation = std::fabs(checkIt.dot(v.first));
  EXPECT_NEAR(correlation, 1, 0.000001);
}

}  // namespace test::math
