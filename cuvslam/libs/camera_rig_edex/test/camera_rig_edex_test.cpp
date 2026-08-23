
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

#include "camera_rig_edex/camera_rig_edex.h"
#include "common/environment.h"
#include "common/image.h"
#include "common/include_gtest.h"
#include "utils/image_loader.h"

namespace test {
using namespace cuvslam;

TEST(CameraRigEdex, DISABLED_ReadSequence) {
  const std::string sequenceFolder = cuvslam::Environment::GetVar(cuvslam::Environment::CUVSLAM_DATASETS);
  const std::string absEdexFileName = sequenceFolder + "kitti/04/stereo.edex";

  edex::EdexFile f;
  ASSERT_TRUE(f.read(absEdexFileName));

  camera_rig_edex::CameraRigEdex rig(absEdexFileName);
  {
    ASSERT_TRUE(rig.start());
    Sources curr_sources;
    Sources masks_sources;
    DepthSources depth_sources;
    Metas curr_meta;
    size_t frameIndex = 0;

    utils::ImageLoaderT imgLoad;
    while (rig.getFrame(curr_sources, curr_meta, masks_sources, depth_sources)) {
      const size_t nCameras = curr_sources.size();

      TracePrint("[ ");

      // compare cameras pixels
      for (size_t i = 0; i < nCameras; ++i) {
        TracePrint("%zu ", static_cast<size_t>(curr_meta[i].frame_id));

        ImageSource& imgptr = curr_sources[i];

        const size_t h = curr_meta[i].shape.height;
        const size_t w = curr_meta[i].shape.width;
        ASSERT_EQ(frameIndex, curr_meta[i].frame_id);

        // build ImageMatrix over images[i].handle
        const ImageMatrixT eigenImageFromHandle = imgptr.as<uint8_t>(curr_meta[i].shape).cast<float>();

        // build ImageMatrix from file
        ASSERT_TRUE(imgLoad.load(sequenceFolder + f.cameras_[i].sequence[frameIndex]));
        const ImageMatrixT eigenImageFromFile = imgLoad.getImage().cast<float>();

        ASSERT_EQ(eigenImageFromFile.rows(), (Index)h);
        ASSERT_EQ(eigenImageFromFile.cols(), (Index)w);

        // verify pixel by pixel that we have same image in both cases
        for (size_t j = 0; j < h; j++) {
          for (size_t k = 0; k < w; k++) {
            ASSERT_EQ(eigenImageFromFile(j, k), eigenImageFromHandle(j, k));
          }
        }
      }

      TracePrint("]\n");

      ++frameIndex;
    }

    ASSERT_TRUE(rig.stop());
  }
}

}  // namespace test
