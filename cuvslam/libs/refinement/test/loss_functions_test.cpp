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

#include "refinement/loss_functions.h"
#include <gtest/gtest.h>

namespace cuvslam::refinement {
namespace {

class BinaryLossTest : public ::testing::Test {
protected:
  void SetUp() override {}
};

TEST_F(BinaryLossTest, BelowCrossOverPoint) {
  BinaryLoss loss(1.0);
  double rho[3];

  // Test several values below cross-over point
  loss.Evaluate(0.5, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.5);  // rho
  EXPECT_DOUBLE_EQ(rho[1], 1.0);  // rho'
  EXPECT_DOUBLE_EQ(rho[2], 0.0);  // rho''

  loss.Evaluate(0.0, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.0);
  EXPECT_DOUBLE_EQ(rho[1], 1.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);

  loss.Evaluate(0.999, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.999);
  EXPECT_DOUBLE_EQ(rho[1], 1.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);
}

TEST_F(BinaryLossTest, AtCrossOverPoint) {
  BinaryLoss loss(1.0);
  double rho[3];

  loss.Evaluate(1.0, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.0);  // At cross-over point, should be zeroed out
  EXPECT_DOUBLE_EQ(rho[1], 0.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);
}

TEST_F(BinaryLossTest, AboveCrossOverPoint) {
  BinaryLoss loss(1.0);
  double rho[3];

  // Test several values above cross-over point
  loss.Evaluate(1.001, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.0);
  EXPECT_DOUBLE_EQ(rho[1], 0.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);

  loss.Evaluate(2.0, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.0);
  EXPECT_DOUBLE_EQ(rho[1], 0.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);

  loss.Evaluate(1000.0, rho);
  EXPECT_DOUBLE_EQ(rho[0], 0.0);
  EXPECT_DOUBLE_EQ(rho[1], 0.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);
}

TEST_F(BinaryLossTest, NegativeValues) {
  BinaryLoss loss(1.0);
  double rho[3];

  // Negative values should be treated as below cross-over point
  loss.Evaluate(-0.5, rho);
  EXPECT_DOUBLE_EQ(rho[0], -0.5);
  EXPECT_DOUBLE_EQ(rho[1], 1.0);
  EXPECT_DOUBLE_EQ(rho[2], 0.0);
}

}  // namespace
}  // namespace cuvslam::refinement
