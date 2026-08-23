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

#include "utils/cuvslam_yaml_config.h"

#include <fstream>
#include <string>

#include "common/include_gtest.h"
#include "cuvslam/cuvslam2_internal.h"

namespace test::utils {

TEST(CuvslamYamlConfig, LoadsPerFrameKeyframeOverrideInternals) {
  const std::string path = ::testing::TempDir() + "/internals_keyframe_override.yaml";
  {
    std::ofstream file(path);
    file << "internals:\n"
         << "  kf_override_frame_selection: true\n";
  }

  cuvslam::internal::Internals internals;
  ASSERT_TRUE(cuvslam::LoadInternalsFromFile(path.c_str(), internals));

  ASSERT_TRUE(internals.kf_override_frame_selection.has_value());
  EXPECT_TRUE(*internals.kf_override_frame_selection);
}

}  // namespace test::utils
