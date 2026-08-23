
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

#pragma once

#include "cuvslam/cuvslam2.h"

#include <string>

namespace cuvslam {

struct TestingSettings {
  std::string data_folder;
  std::vector<uint8_t> camera_ids;
  int start_frame = 0;
  float max_fps = 0.;
  bool use_gpu_mem = false;
  bool print_nan_on_failure = true;
  bool enable_slam = false;
  std::string odom_poses_file;
  std::string slam_poses_file;
};

bool DoesEdexExist(const std::string& data_folder);

bool DoesEdexHaveImu(const std::string& data_folder);

bool TrackEdexApi2(const TestingSettings& settings, const Odometry::Config& cfg);

}  // namespace cuvslam
