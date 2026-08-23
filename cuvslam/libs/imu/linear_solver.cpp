
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

// qr.solve(B); may return inf/nan and uninitialized warning
// this file is for accurate wrapping it
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "Eigen/Eigenvalues"  // should be before any Eigen include
#pragma GCC diagnostic pop

#include "imu/linear_solver.h"

namespace cuvslam::sba_imu {

// This method finds a solution X to the equation AX=B using ColPivHouseholderQR
template <typename MatrixType>
bool SolveLinearEquation(const MatrixType& A, const MatrixType& B, MatrixType& X) {
  Eigen::ColPivHouseholderQR<MatrixType> qr(A);
  qr.setThreshold(1e-5f);
  if (!qr.isInvertible()) {
    return false;
  }
  // From the Eigen documentation:
  // The method below just tries to find as good a solution as possible.
  // If you want to check whether a solution exists or if it is accurate,
  // just call this function to get a result and then compute the error of this result,
  // or use MatrixBase::isApprox() directly
  X = qr.solve(B);

  const bool solution_exists = (A * X).isApprox(B, 1e-4f);
  return solution_exists;
}

bool SolveLinearEquation(const Matrix3T& A, const Matrix3T& B, Matrix3T& X) {
  return SolveLinearEquation<Matrix3T>(A, B, X);
}
bool SolveLinearEquation(const Matrix9T& A, const Matrix9T& B, Matrix9T& X) {
  return SolveLinearEquation<Matrix9T>(A, B, X);
}
bool SolveLinearEquation(const Matrix15T& A, const Matrix15T& B, Matrix15T& X) {
  return SolveLinearEquation<Matrix15T>(A, B, X);
}

}  // namespace cuvslam::sba_imu
