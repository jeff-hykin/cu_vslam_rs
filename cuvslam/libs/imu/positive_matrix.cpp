
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

// SelfAdjointEigenSolver may return inf/nan and uninitialized warning
// this file is for accurate wrapping it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "Eigen/Eigenvalues"  // should be before any Eigen include
#pragma GCC diagnostic pop

#include "imu/positive_matrix.h"

namespace cuvslam::sba_imu {

template <int Dim>
bool MakeMatrixPositive(SquareMatrix<float, Dim>& info) {
  using Matrix = SquareMatrix<float, Dim>;
  Eigen::SelfAdjointEigenSolver<Matrix> es(info);
  if (es.info() != Eigen::ComputationInfo::Success) {
    return false;
  }
  Eigen::Matrix<float, Dim, 1> eigen_values = es.eigenvalues();
  for (int i = 0; i < Dim; i++) {
    if (eigen_values[i] < 1e-12) {
      eigen_values[i] = 0;
    }
  }
  const Matrix& eigen_vectors = es.eigenvectors();
  info = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
  return true;
}

bool MakeMatrixPositive(Matrix3T& info) { return MakeMatrixPositive<3>(info); }
bool MakeMatrixPositive(Matrix9T& info) { return MakeMatrixPositive<9>(info); }
bool MakeMatrixPositive(Matrix15T& info) { return MakeMatrixPositive<15>(info); }

}  // namespace cuvslam::sba_imu
