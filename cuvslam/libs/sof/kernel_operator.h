
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
#include "common/log.h"

namespace cuvslam::sof {

template <typename _KernelMatrix>
class KernelOperator {
public:
  using Scalar = typename _KernelMatrix::Scalar;

  enum { CArraySize = _KernelMatrix::SizeAtCompileTime };
  typedef Scalar ScalarArray[CArraySize];

  KernelOperator(const _KernelMatrix& kernel) : kernel_(kernel) {
    if ((kernel.rows() % 2) == 0 || (kernel.cols() % 2) == 0) {
      TraceError("Invalid kernel size, should be odd size in both dimensions.");
    }
  }

  Eigen::Index cols() const { return kernel_.cols(); }
  Eigen::Index rows() const { return kernel_.rows(); }

  template <typename _MatrixOther>
  Scalar operator()(const Eigen::MatrixBase<_MatrixOther>& other, const Scalar) const {
    return kernel_.cwiseProduct(other).sum();
  }

  template <typename _MatrixOther>
  Scalar operator()(const Eigen::MatrixBase<_MatrixOther>& other) const {
    return kernel_.cwiseProduct(other).sum();
  }

private:
  _KernelMatrix kernel_;
};

}  // namespace cuvslam::sof
