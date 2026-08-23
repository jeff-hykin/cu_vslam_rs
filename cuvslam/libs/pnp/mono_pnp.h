
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

#include <cstdint>

#include "common/isometry.h"
#include "common/types.h"
#include "common/unaligned_types.h"

namespace cuvslam::pnp {

struct PosePrior {
  using Pose = storage::Pose<float>;
  using Mat6 = storage::Mat6<float>;

  Pose mean = Pose::Identity();
  Mat6 precision = Mat6::Zero();  // inverse of covariance

  PosePrior() = default;
  PosePrior(PosePrior&&) = default;
  PosePrior& operator=(PosePrior&&) = default;
};

struct RotationPrior {
  using Pose = storage::Pose<float>;
  using Mat3 = storage::Mat3<float>;

  Pose mean = Pose::Identity();
  Mat3 precision = Mat3::Zero();  // inverse of covariance

  RotationPrior() = default;
  RotationPrior(RotationPrior&&) = default;
  RotationPrior& operator=(RotationPrior&&) = default;
};

struct PoseRefinementInput {
  // 2D observations, metric space (z = 1)
  // 2n elements
  float* uvs = nullptr;

  // 3D point coordinates, world space
  // 3n elements
  float* xyzs = nullptr;

  // information matrices,
  // 4 elements per observation, column-major
  float* infos = nullptr;

  // number of points/observations
  int32_t n = 0;

  // optimizer parameters
  float huber_delta = 1.35f;

  // if not nullptr, all 2d info matrices will be substituted by this matrix
  storage::Mat2<float>* defaultInfo = nullptr;
};

// Modifies pose of a pinhole camera to minimize
// norm(point2D - project(cameraFromWorld * point3D))
// input should be a 2 element array (data for L and R cameras)
bool RefinePose(Isometry3T& left_camera_from_world, Matrix6T& precision, const PoseRefinementInput* input,
                const PosePrior& prior, const cuvslam::Isometry3T& right_from_left_camera_transform);

// Modifies pose of a pinhole camera by applying rotations on top
// of the initial pose to minimize
// norm(point2D - project(cameraFromWorld * point3D))
bool RefineRotation(Isometry3T& left_camera_from_world, Matrix3T& precision, const PoseRefinementInput*,
                    const RotationPrior&, const cuvslam::Isometry3T& right_from_left_camera_transform);

}  // namespace cuvslam::pnp
