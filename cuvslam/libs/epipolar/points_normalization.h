
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

#include "Eigen/Eigenvalues"

#include "common/types.h"
#include "common/vector_2t.h"

namespace cuvslam::epipolar {

///-------------------------------------------------------------------------------------------------
/// @brief Class handles the 2d transformation for normalization (@see NormalizePoints).
///
/// The idea behind this class is optimization:
/// * normalize() method is faster then multiplication on general 3x3 matrix
/// * in some algorithms we require to have normalization matrix, in others reverse matrix
///   to skip general matrix inversion, calcDenorm matrix is provided
///-------------------------------------------------------------------------------------------------
class NormalizationTransform {
  Vector2T mean_ = Vector2T::Zero();
  float scale_ = 0;

public:
  NormalizationTransform(const Vector2TVector &srcPoints, float minDistLimit = epsilon());

  bool isValid() const;

  Matrix3T calcNormMatrix() const;

  Matrix3T calcDenormMatrix() const;

  Vector2T operator()(const Vector2T &src) const;
};

}  // namespace cuvslam::epipolar
