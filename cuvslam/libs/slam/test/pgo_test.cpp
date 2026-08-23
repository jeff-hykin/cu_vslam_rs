
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

#include "Eigen/Geometry"

#include "common/include_gtest.h"
#include "common/vector_3t.h"

namespace test::slam {

TEST(SlamTest, PGO) {
  Eigen::AlignedBox3f observer_box;
  cuvslam::Vector3T v1(1, 2, 3);
  cuvslam::Vector3T v2(-1, -2, -3);
  observer_box.extend(v1);
  observer_box.extend(v2);

  auto scale = 1.2f;

  cuvslam::Vector3T center = observer_box.center();
  // pr
  double radius2 = observer_box.diagonal().squaredNorm() * (scale * scale);
  printf("%f, %f", center.norm(), radius2);
}

}  // namespace test::slam
