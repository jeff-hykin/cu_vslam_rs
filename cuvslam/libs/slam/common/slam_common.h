
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

#include <limits>

#include "common/isometry.h"
#include "common/log.h"

namespace cuvslam::slam {

using KeyFrameId = size_t;
using LandmarkId = size_t;

constexpr KeyFrameId InvalidKeyFrameId = std::numeric_limits<KeyFrameId>::max();
constexpr LandmarkId InvalidLandmarkId = std::numeric_limits<LandmarkId>::max();

// only rotation and translation
void RemoveScaleFromTransform(Isometry3T& pose);

}  // namespace cuvslam::slam

// for USE_SLAM_OUTPUT see src\CMakeLists.txt

#ifdef USE_SLAM_OUTPUT
#define SlamStdout(...)         \
  fprintf(stdout, __VA_ARGS__); \
  fflush(stdout)
#define SlamStderr(...)         \
  fprintf(stdout, __VA_ARGS__); \
  fflush(stdout);               \
  TraceError(__VA_ARGS__)
#else
#define SlamStdout(...) (void(0))
#define SlamStderr(...) TraceError(__VA_ARGS__)
#endif  // USE_SLAM_OUTPUT
