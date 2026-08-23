
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

#include <cassert>
#include <cmath>

namespace cuvslam::math {

float ComputeGemanMcClureLoss(float x_squared, const float delta, const float mu = 1.);
float ComputeDGemanMcClureLoss(float x_squared, const float delta, const float mu = 1.);

float ComputeTruncatedLeastSquaresLoss(float x_squared, const float delta, const float mu = 1.);
float ComputeDTruncatedLeastSquaresLoss(float x_squared, const float delta, const float mu = 1.);

float ComputeHuberLoss(float x_squared, const float delta);
float ComputeDHuberLoss(float x_squared, const float delta);

float ComputeStudentLoss(float x_squared, const float delta, const float nu = 1.f);
float ComputeDStudentLoss(float x_squared, const float delta, const float nu = 1.f);

}  // namespace cuvslam::math
