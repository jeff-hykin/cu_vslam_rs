
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

#include <optional>

#include "common/coordinate_system.h"
#include "common/isometry.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"

namespace cuvslam::edex {

enum class RotationStyle { EulerDegrees, Quaternion, RotationMatrix };

using Sequence = std::vector<std::string>;
using DepthSequence = std::vector<std::string>;
using DepthId = std::optional<uint8_t>;

// This is edex compatible structure
struct Intrinsics {
  Vector2T resolution;
  Vector2T focal;
  Vector2T principal;
  std::string distortion_model;
  std::vector<float> distortion_params;
};

struct Camera {
  Isometry3T transform;
  Sequence sequence;
  Intrinsics intrinsics;
  Tracks2DVectorsMap tracks2D;  // per frame
  bool has_depth = false;
  DepthId depth_id;
  DepthSequence depth_sequence;
};

struct IMU {
  Isometry3T transform = Isometry3T::Identity();
  std::vector<float> g = {0.0f, 9.81f, 0.0f};
  std::string imu_log_path_ = "";
  CoordinateSystem coordinate_system = CoordinateSystem::OPENCV;  // raw IMU body frame needs no re-axis
  float accelerometer_noise_density = 0.002;                      // default is euroc imu noise params
  float accelerometer_random_walk = 0.003;
  float gyroscope_noise_density = 0.00016968;
  float gyroscope_random_walk = 0.000019393;
  float frequency = 200;
};

using Cameras = std::vector<Camera>;

}  // namespace cuvslam::edex
