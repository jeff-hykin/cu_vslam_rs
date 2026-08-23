
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
#include "utils/image_loader.h"

namespace test {
using namespace cuvslam;

TEST(Loader, DISABLED_DepthLoader) {
  const std::string npy_file_path = std::string(CUVSLAM_TEST_ASSETS) + "test.npy";

  utils::DepthLoader loader(npy_file_path.c_str());

  const auto& depth_image = loader.getImage();

  const auto w = depth_image.cols();
  const auto h = depth_image.rows();

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      ASSERT_EQ(depth_image(i, j), 7 * i + j);
    }
  }
}

}  // namespace test
