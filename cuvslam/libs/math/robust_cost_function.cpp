
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

#include "math/robust_cost_function.h"

namespace cuvslam::math {

float ComputeGemanMcClureLoss(float x_squared, const float delta, const float mu) {
  const float delta_squared = delta * delta;
  return mu * delta_squared * x_squared / (mu * delta_squared + x_squared);
}

float ComputeDGemanMcClureLoss(float x_squared, const float delta, const float mu) {
  const float delta_squared = delta * delta;
  float a = mu * delta_squared + x_squared;
  return mu * mu * delta_squared * delta_squared / (a * a);
}

float ComputeTruncatedLeastSquaresLoss(float x_squared, const float delta, const float mu) {
  const float delta_squared = delta * delta;
  if (x_squared < mu * delta_squared / (mu + 1)) {
    return x_squared;
  }

  if (x_squared > (mu + 1) * delta_squared / mu) {
    return delta_squared;
  }

  float r = sqrtf(x_squared);

  return 2 * delta * r * sqrtf(mu * (mu + 1)) - mu * (delta_squared + x_squared);
}

float ComputeDTruncatedLeastSquaresLoss(float x_squared, const float delta, const float mu) {
  const float delta_squared = delta * delta;
  if (x_squared < mu * delta_squared / (mu + 1)) {
    return 1.;
  }

  if (x_squared > (mu + 1) * delta_squared / mu) {
    return 0.;
  }

  float r = sqrtf(x_squared);
  return delta * sqrtf(mu * (mu + 1)) / r - mu;
}

float ComputeHuberLoss(float x_squared, const float delta) {
  // assert(x_squared >= 0);  TODO: restore
  if (delta <= 0.f) return 0.5f * x_squared;

  const auto delta_squared = delta * delta;

  if (x_squared < delta_squared) {
    return 0.5f * x_squared;
  }

  return delta * std::sqrt(x_squared) - 0.5f * delta_squared;
}

// Derivative of Huber loss
float ComputeDHuberLoss(float x_squared, const float delta) {
  // assert(x_squared >= 0);
  if (delta <= 0.f) return 0.5f;

  const auto delta_squared = delta * delta;

  if (x_squared < delta_squared) {
    return 0.5f;
  }

  return 0.5f * delta / std::sqrt(x_squared);
}

float ComputeStudentLoss(float x_squared, const float delta, const float nu) {
  return (delta + nu) * std::log(1.f + x_squared / delta);
}

float ComputeDStudentLoss(float x_squared, const float delta, const float nu) {
  return (delta + nu) / (delta + x_squared);
}

}  // namespace cuvslam::math
