
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

#include "common/include_eigen.h"

#include "unsupported/Eigen/LevenbergMarquardt"
#include "unsupported/Eigen/NumericalDiff"

namespace cuvslam {

// Bridge between fixed-size functor InputType and LM's dynamic FVectorType.
// Stock Eigen's LM defines FVectorType = Matrix<Scalar, Dynamic, 1> regardless of the functor's InputType,
// so fixed-size vectors can't bind directly to minimize(FVectorType&).
template <typename LM, typename VectorType>
Eigen::LevenbergMarquardtSpace::Status lm_minimize(LM& lm, VectorType& x) {
  typename LM::FVectorType x_dyn = x;
  auto status = lm.minimize(x_dyn);
  x = x_dyn;
  return status;
}

}  // namespace cuvslam
