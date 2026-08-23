
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

#include "edex/file_name_tools.h"

#include "common/include_gtest.h"

namespace test::edex {

using namespace cuvslam::edex::filepath;

TEST(FileNameToolsTest, StripExt) {
  EXPECT_EQ(StripExt("simple.ext"), "simple");
  EXPECT_EQ(StripExt("simple.ext1.ext2"), "simple.ext1");
  EXPECT_EQ(StripExt("c:/some/path.ext1/simple.ext2.ext2"), "c:/some/path.ext1/simple.ext2");
  EXPECT_EQ(StripExt("c:/some/../../path.ext1/simple.ext2.ext2"), "c:/some/../../path.ext1/simple.ext2");
}

TEST(FileNameToolsTest, GetFilePath) {
  EXPECT_EQ(GetFilePath("c:/some/path/name.ext"), "c:/some/path/");
  EXPECT_EQ(GetFilePath("c:\\some\\path\\name.ext"), "c:\\some\\path\\");
  EXPECT_EQ(GetFilePath("\\\\server-name\\share\\some\\path\\name.ext"), "\\\\server-name\\share\\some\\path\\");
  EXPECT_EQ(GetFilePath("C:\\path1"), "C:\\");
  EXPECT_EQ(GetFilePath("C:\\path1\\path2"), "C:\\path1\\");
  EXPECT_EQ(GetFilePath("\\path1"), "\\");
  EXPECT_EQ(GetFilePath("/home/xx/share/sequences/v_3_undistorted/undistorted_0/image.0001.jpg"),
            "/home/xx/share/sequences/v_3_undistorted/undistorted_0/");
  EXPECT_EQ(GetFilePath("P:/repository/resources/sequences/scooby/matchmove.ca038BgRaw.0001.jpg"),
            "P:/repository/resources/sequences/scooby/");
  EXPECT_EQ(GetFilePath("matchmove.ca038BgRaw.0001.jpg"), "");
  EXPECT_EQ(GetFilePath("../../../../sequences/kitti/04/04.0.0001.png"), "../../../../sequences/kitti/04/");
  EXPECT_EQ(GetFilePath("name.ext"), "");
  EXPECT_EQ(GetFilePath("/file_in_linux_root.txt"), "/");
}

TEST(FileNameToolsTest, SplitSequinceFileName) {
  std::string prefix;
  cuvslam::FrameId frameId{1};
  std::string ext;

  EXPECT_TRUE(SplitSequinceFileName("/any/path/matchmove.ca038BgRaw.0004.jpg", &prefix, &frameId, &ext));
  EXPECT_EQ(prefix, "/any/path/matchmove.ca038BgRaw.");
  EXPECT_EQ(frameId, cuvslam::FrameId(4));
  EXPECT_EQ(ext, ".jpg");
}

TEST(FileNameToolsTest, ExtractFrameIdFromFileName) {
  cuvslam::FrameId frameId{1};

  EXPECT_TRUE(ExtractFrameIdFromFileName("/any/path/matchmove.ca038BgRaw.0004.jpg", frameId));
  EXPECT_EQ(frameId, cuvslam::FrameId(4));
}

TEST(FileNameToolsTest, ZeroPadFrameId) {
  EXPECT_EQ(ZeroPadFrameId(0), "0000");
  EXPECT_EQ(ZeroPadFrameId(1), "0001");
  EXPECT_EQ(ZeroPadFrameId(12), "0012");
  EXPECT_EQ(ZeroPadFrameId(123), "0123");
  EXPECT_EQ(ZeroPadFrameId(1234), "1234");
}

TEST(FileNameToolsTest, ReplaceFrameIdInFileName) {
  std::string replacedFileName;

  EXPECT_TRUE(ReplaceFrameIdInFileName("/any/path/matchmove.ca038BgRaw.0004.jpg", 123, replacedFileName));
  EXPECT_EQ(replacedFileName, "/any/path/matchmove.ca038BgRaw.0123.jpg");
}

}  // namespace test::edex
