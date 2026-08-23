
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

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

namespace cuvslam::math {

namespace {
template <int Dim>
using Vec = Eigen::Matrix<float, Dim, 1>;

template <int Dim>
using Mat = Eigen::Matrix<float, Dim, Dim>;
}  // namespace

// If `lambda` is affine (or is close enough to affine for a caller purposes),
// the function can use less computations based on the fact that
// `lambda(mean) = 1/2 * (lambda(mean + delta) + lambda(mean - delta))`.
// `normalized = lambda(mean)` must be provided in this case.
template <int InDim, int OutDim>
Mat<OutDim> approximate(const Vec<InDim>& mean, const Vec<InDim>& normalized, const Mat<InDim>& cov,
                        const std::function<Vec<OutDim>(const Vec<InDim>& x)>& lambda, bool is_lambda_affine = false) {
  Mat<InDim> L = cov.llt().matrixL();
  // TODO(C++23): remove ternary when constexpr srqt() is available
  constexpr const float sqrt_L = InDim == 2 ? 1.414213562 : sqrt(InDim);

  Vec<OutDim> v_out;
  Vec<OutDim> mean_v = Vec<OutDim>::Zero();
  Mat<OutDim> mean_sst = Mat<OutDim>::Zero();

  for (int i = 0; i < InDim; i++) {
    auto delta = sqrt_L * L.col(i);
    Vec<OutDim> v_out = lambda(mean + delta);
    mean_v += v_out;
    mean_sst += v_out * v_out.transpose();

    Vec<OutDim> v_opp = is_lambda_affine ? 2 * normalized - v_out : lambda(mean - delta);
    mean_v += v_opp;
    mean_sst += v_opp * v_opp.transpose();
  }

  mean_v /= (2 * InDim);
  mean_sst /= (2 * InDim);

  return mean_sst - mean_v * mean_v.transpose();
}

}  // namespace cuvslam::math
