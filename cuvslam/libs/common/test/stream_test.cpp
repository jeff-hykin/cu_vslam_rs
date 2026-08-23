
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

#include <iostream>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/stream.h"
#include "common/types.h"

using namespace cuvslam;

TEST(Stream, Tum) {
  Isometry3T pose;
  pose.setIdentity();
  pose.translate(Vector3<float>(1, 2, 3));

  std::stringstream ss;
  ss << std::fixed << std::setprecision(0) << PoseIOManip(PoseFormat::TUM, false);
  ss << pose;
  ASSERT_EQ(ss.str(), "1 2 3 0 0 0 1");
}

TEST(Stream, MatrixAndConvert) {
  Isometry3T pose;
  pose.setIdentity();
  pose.translate(Vector3<float>(1, 2, 3));

  std::stringstream ss;
  ss << std::fixed << std::setprecision(0) << PoseIOManip(PoseFormat::MATRIX, true);
  ss << pose;
  ASSERT_EQ(ss.str(), "1 0 0 3 0 1 0 -1 0 0 1 -2");
}
