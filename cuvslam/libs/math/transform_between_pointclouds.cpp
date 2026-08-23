
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

#include "math/transform_between_pointclouds.h"

#include <numeric>

namespace cuvslam::math {

/*
The Kabsch algorithm
https://en.wikipedia.org/wiki/Kabsch_algorithm
*/

// find rigid transform for two point clouds
bool TransformBetweenPointclouds(const std::vector<Vector3T>& A, const std::vector<Vector3T>& B,
                                 Isometry3T& transform) {
  transform = Isometry3T::Identity();
  if (A.size() != B.size()) {
    return false;
  }

  Vector3T A_mean = std::accumulate(A.begin(), A.end(), Vector3T(0, 0, 0)) / A.size();
  Vector3T B_mean = std::accumulate(B.begin(), B.end(), Vector3T(0, 0, 0)) / B.size();

  Eigen::MatrixXf Am(3, A.size());
  Eigen::MatrixXf Bm(3, B.size());
  for (size_t i = 0; i < A.size(); i++) {
    Vector3T a = A[i] - A_mean;
    for (int x = 0; x < 3; x++) {
      Am(x, i) = a(x);
    }
    Vector3T b = B[i] - B_mean;
    for (int x = 0; x < 3; x++) {
      Bm(x, i) = b(x);
    }
  }
  Eigen::MatrixXf m = Am * Bm.transpose();

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Vector3f rhs(1, 0, 0);
  svd.solve(rhs);

  auto R = svd.matrixV() * svd.matrixU().transpose();

  transform = Isometry3T::Identity();
  transform.linear() = R;
  transform.translation() = -(R * A_mean) + B_mean;
  transform.makeAffine();

  return true;
}

}  // namespace cuvslam::math
