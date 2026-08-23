
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

#include "edex/edex.h"

#include "common/include_gtest.h"
#include "edex/file_name_tools.h"

namespace test::edex {

using EdexTest = testing::TestWithParam<const char*>;

TEST_P(EdexTest, DISABLED_ReaderWriterTest) {
  cuvslam::edex::EdexFile f1;
  cuvslam::edex::EdexFile f2;

  const std::string edexFolder = CUVSLAM_TEST_ASSETS;

  const std::string testEdexFile = GetParam();
  const std::string inTestJson = edexFolder + testEdexFile;
  const std::string outTestJson =
      edexFolder + cuvslam::edex::filepath::GetFilePath(testEdexFile) + std::to_string(rand()) + ".edex";
  ASSERT_TRUE(f1.read(inTestJson));
  ASSERT_TRUE(f1.write(outTestJson));
  ASSERT_TRUE(f2.read(outTestJson));
  ASSERT_EQ(remove(outTestJson.c_str()), 0) << "can't delete temp file";

  // compare data
  ASSERT_EQ(f1.cameras_.size(), f2.cameras_.size());

  for (size_t i = 0; i < f1.cameras_.size(); ++i) {
    const cuvslam::edex::Camera& c1 = f1.cameras_[i];
    const cuvslam::edex::Camera& c2 = f2.cameras_[i];
    EXPECT_EQ(c1.tracks2D, c2.tracks2D);
    EXPECT_EQ(c1.intrinsics.resolution, c2.intrinsics.resolution);
    EXPECT_EQ(c1.intrinsics.focal, c2.intrinsics.focal);
    EXPECT_EQ(c1.intrinsics.principal, c2.intrinsics.principal);
    EXPECT_EQ(c1.intrinsics.distortion_model, c2.intrinsics.distortion_model);
    EXPECT_EQ(c1.intrinsics.distortion_params, c2.intrinsics.distortion_params);
    EXPECT_TRUE(c1.transform.isApprox(c2.transform));
    EXPECT_EQ(c1.sequence, c2.sequence);
  }

  EXPECT_EQ(f1.rigPositions_.size(), f2.rigPositions_.size());
  for (const auto& p : f1.rigPositions_) {
    const cuvslam::FrameId id = p.first;
    const cuvslam::Isometry3T& position = p.second;
    EXPECT_TRUE(position.isApprox(f2.rigPositions_.at(id)));
  }

  EXPECT_EQ(f1.tracks3D_, f2.tracks3D_);
  EXPECT_EQ(f1.timeline_.firstFrame(), f2.timeline_.firstFrame());
  EXPECT_EQ(f1.timeline_.lastFrame(), f2.timeline_.lastFrame());
  EXPECT_EQ(f1.keyFrames_, f2.keyFrames_);
}

INSTANTIATE_TEST_SUITE_P(Edex, EdexTest,
                         testing::Values("edex/clean_empty_sequence.edex", "edex/clean_start_frame_sequence.edex",
                                         "edex/clean_some_frame_sequence.edex", "edex/clean_with_key_frames.edex",
                                         "edex/clean_multy_cameras.edex", "edex/distortions.edex"));

}  // namespace test::edex
