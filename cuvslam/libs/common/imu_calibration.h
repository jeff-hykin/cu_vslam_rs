
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

namespace cuvslam::imu {

// Math according to Kalibr docs
// https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model
class ImuCalibration {
public:
  // euroc calibration
  ImuCalibration();

  ImuCalibration(const Isometry3T& rig_from_imu, float gyroscope_noise_density, float gyroscope_random_walk,
                 float accelerometer_noise_density, float accelerometer_random_walk, float frequency);

  const Isometry3T& rig_from_imu() const;
  const Matrix3T& gyro_noise() const;
  const Matrix3T& acc_noise() const;
  const Matrix3T& gyro_random_walk() const;
  const Matrix3T& acc_random_walk() const;
  float frequency() const;

private:
  Isometry3T rig_from_imu_ = Isometry3T::Identity();

  // euroc kalibr calibration data
  float gnd_ = 0.00016968;   // rad / (s * srqt(Hz))
  float grw_ = 0.000019393;  // rad / (s ^ 2 * srqt(Hz))
  float and_ = 0.002;        // m / (s ^ 2 * srqt(Hz))
  float arw_ = 0.003;        // m / (s ^ 3 * srqt(Hz))
  float freq_ = 200;         // Hz

  Matrix3T gyro_noise_cov_ = Matrix3T::Zero();
  Matrix3T acc_noise_cov_ = Matrix3T::Zero();

  Matrix3T gyro_random_walk_cov_ = Matrix3T::Zero();
  Matrix3T acc_random_walk_cov_ = Matrix3T::Zero();
};

}  // namespace cuvslam::imu
