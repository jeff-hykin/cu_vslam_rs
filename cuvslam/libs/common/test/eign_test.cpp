
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

#include "Eigen/Eigenvalues"

#include "common/include_gtest.h"
#include "common/types.h"

using namespace cuvslam;

namespace Test {

TEST(TestEigen, DISABLED_TestEignValues) {
  MatrixXT svdData(6, 6);
  svdData << 0.0199687239f, 0.000839802087f, -0.0255170017f, 0.0103427283f, 0.0119441925f, -0.00481563760f,
      0.000839802087f, 0.00252064294f, -0.00604654755f, -0.00303798495f, 0.00480106333f, -0.00324882311f,
      -0.0255170017f, -0.00604654755f, 0.0326068103f, -0.0133496309f, -0.00983827561f, 0.00318189710f, 0.0103427283f,
      -0.00303798495f, -0.0133496309f, 0.00802240707f, 0.00928178150f, -0.00371615426f, 0.0119441925f, 0.00480106333f,
      -0.00983827561f, 0.00928178150f, 0.00124395895f, 0.00131855917f, -0.00481563760f, -0.00324882311f, 0.00318189710f,
      -0.00371615426f, 0.00131855917f, -0.00147585059f;

  EXPECT_TRUE(svdData.isApprox(svdData.transpose(), 0));
  auto eigv = svdData.eigenvalues();

  for (int i = 0; i < 6; ++i) {
    const float real = eigv[i].real();
    EXPECT_GT(real, 0);
    EXPECT_EQ(eigv[i].imag(), 0);
  }
}

TEST(TestEigen, TestVectorSum) {
  const int size = 10000;
  Vector<float, Eigen::Dynamic> a(size), b(size);
  a = a.setRandom().cwiseInverse();
  b = b.setRandom().cwiseInverse();

  /* don't know exactly, what is it, bug this make warning on linux
  float s = 0;

  s = a.sum();
  s = a.dot(b);
  */
}

}  // namespace Test
