
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

#include <random>

#include "common/include_gtest.h"
#include "math/twist_to_quaternion.h"

namespace test::math {
using namespace cuvslam;

TEST(TwistToQuaternionTest, TwistToQuaternionTest) {
  Isometry3T cameraOrig = Translation3T(Vector3T::Random()) * Rotation3T(Vector3T::Random(), AngleUnits::Radian);
  cuvslam::math::VectorTwistT twist(cameraOrig.inverse());
  cuvslam::math::TwistToQuaternion tqTransform(twist);
  Vector7T tq = tqTransform.getQuatTran();

  std::cout << "m:" << tqTransform << std::endl;
  std::cout << "twist:" << twist << std::endl;
  std::cout << "q+t:" << tq << std::endl;

  const float& w = tq[3];
  const float& x = tq[0];
  const float& y = tq[1];
  const float& z = tq[2];
  QuaternionT qN(w, x, y, z);
  Isometry3T cameraExpectedInversed = Translation3T(Vector3T(tq.tail(3))) * qN;

  Isometry3T transform = cameraOrig * cameraExpectedInversed;
  EXPECT_TRUE(transform.isApprox(Isometry3T::Identity()));
}

}  // namespace test::math
