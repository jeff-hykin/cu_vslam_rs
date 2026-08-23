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

#include "slam/async_slam/tail.h"
#include "common/include_gtest.h"

namespace test::slam {
namespace {

cuvslam::Isometry3T PoseAtX(float x) {
  cuvslam::Isometry3T pose = cuvslam::Isometry3T::Identity();
  pose.translation().x() = x;
  return pose;
}

}  // namespace

TEST(Tail, RejectsSlamUpdatesOlderThanRetentionWindow) {
  cuvslam::slam::Tail tail(2'000);

  ASSERT_TRUE(tail.UpdateTimeByOdometry(0, PoseAtX(0)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(1'000, PoseAtX(1)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(2'000, PoseAtX(2)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(3'000, PoseAtX(3)));

  EXPECT_FALSE(tail.IsTimestampRetained(999));
  EXPECT_TRUE(tail.IsTimestampRetained(1'000));
  EXPECT_TRUE(tail.IsTimestampRetained(3'000));
  EXPECT_TRUE(tail.IsTimestampRetained(4'000));

  EXPECT_FALSE(tail.UpdatePoseBySLAM(999, PoseAtX(100)));
  const auto tip = tail.GetTip();
  ASSERT_TRUE(tip);
  EXPECT_EQ(tip->first, 3'000);
  EXPECT_NEAR(tip->second.translation().x(), 3.f, 1e-5f);
}

TEST(Tail, AppliesRetainedPastUpdateToLaterTailPoses) {
  cuvslam::slam::Tail tail(10'000);

  ASSERT_TRUE(tail.UpdateTimeByOdometry(0, PoseAtX(0)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(1'000, PoseAtX(1)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(2'000, PoseAtX(2)));

  ASSERT_TRUE(tail.UpdatePoseBySLAM(1'000, PoseAtX(10)));
  const auto tip = tail.GetTip();
  ASSERT_TRUE(tip);
  EXPECT_EQ(tip->first, 2'000);
  EXPECT_NEAR(tip->second.translation().x(), 11.f, 1e-5f);
}

TEST(Tail, ZeroRetentionKeepsOnlyTheLatestTimestampUsable) {
  cuvslam::slam::Tail tail(0);

  ASSERT_TRUE(tail.UpdateTimeByOdometry(0, PoseAtX(0)));
  ASSERT_TRUE(tail.UpdateTimeByOdometry(1'000, PoseAtX(1)));

  EXPECT_FALSE(tail.IsTimestampRetained(999));
  EXPECT_TRUE(tail.IsTimestampRetained(1'000));
  EXPECT_TRUE(tail.IsTimestampRetained(2'000));
}

}  // namespace test::slam
