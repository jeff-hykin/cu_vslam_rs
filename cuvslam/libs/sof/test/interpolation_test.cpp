
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

#include "common/include_gtest.h"

#include "sof/image_pyramid_float.h"

namespace test::sof {

TEST(ImagePyramid, ComputePatchInterpolatesCorrectly) {
  using namespace cuvslam;
  const int n = 512;
  ImageMatrixT image(n, n);

  // vertical linear ramp
  for (int i = 0; i < n; ++i) {
    image.row(i).setConstant(static_cast<float>(i));
  }

  cuvslam::sof::ImagePyramidT pyramid(image);
  EXPECT_TRUE(pyramid.buildPyramid());

  // center at pixel (10.25, 10.25) (pixel coordinates)
  // Quarter pixel offset should allow interpolation without any round-off errors.
  Vector2T xy(10.75f, 10.75f);

  // extra -0.5f to account for different coordinate systems
  const float expectedSmallestValue = xy.y() - 2.f - 0.5f;

  ImageMatrixPatch<float, 5, 5> patch;
  EXPECT_TRUE(pyramid.computePatch(patch, xy, 0));

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(patch(i, j), static_cast<float>(i) + expectedSmallestValue);
    }
  }

  //
  // Do the same in another direction
  //

  // horizontal linear ramp
  for (int i = 0; i < n; ++i) {
    image.col(i).setConstant(static_cast<float>(i));
  }

  pyramid.base() = image;
  EXPECT_TRUE(pyramid.buildPyramid());

  EXPECT_TRUE(pyramid.computePatch(patch, xy, 0));

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(patch(i, j), static_cast<float>(j) + expectedSmallestValue);
    }
  }
}

}  // namespace test::sof
