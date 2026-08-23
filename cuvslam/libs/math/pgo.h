
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

#include <unordered_set>
#include <vector>

#include "Eigen/Sparse"

#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/types.h"

namespace cuvslam::math {

struct PGOInput {
  struct PoseDelta {
    int pose1_id;
    int pose2_id;
    Isometry3T pose1_from_pose2;
    Matrix6T info = Matrix6T::Identity();
  };

  std::vector<PoseDelta> deltas;
  std::vector<Isometry3T> poses;
  std::unordered_set<int> constrained_pose_ids;

  bool use_planar_constraint = false;
  Vector4T plane_normal = {1, 0, 0,
                           0};  // [A, B, C, D], where norm([A, B, C]) = 1, D is shift, see wiki for plane details
  float planar_weight = 1;

  float robustifier = 0.1f;

  void clear();
};

class PGO {
public:
  bool run(PGOInput& inputs, int max_iterations = 10) const;

private:
  bool pgo_planar(PGOInput& inputs, int max_iterations) const;
  bool pgo_regular(PGOInput& inputs, int max_iterations) const;

  // cache
  mutable Eigen::MatrixXf H_;
  mutable Eigen::VectorXf rhs_;
  mutable Eigen::VectorXf step_;
  mutable Eigen::SparseMatrix<float> HS_;
};

}  // namespace cuvslam::math
