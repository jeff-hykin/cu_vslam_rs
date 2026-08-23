
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

#include "common/imu_calibration.h"

namespace cuvslam::imu {

ImuCalibration::ImuCalibration() {
  rig_from_imu_.makeAffine();

  gyro_noise_cov_ = Matrix3T::Identity() * gnd_ * gnd_ * freq_;
  acc_noise_cov_ = Matrix3T::Identity() * and_ * and_ * freq_;

  gyro_random_walk_cov_ = Matrix3T::Identity() * grw_ * grw_ / freq_;
  acc_random_walk_cov_ = Matrix3T::Identity() * arw_ * arw_ / freq_;
}

ImuCalibration::ImuCalibration(const Isometry3T& rig_from_imu, float gyroscope_noise_density,
                               float gyroscope_random_walk, float accelerometer_noise_density,
                               float accelerometer_random_walk, float frequency)
    : rig_from_imu_(rig_from_imu),
      gnd_(gyroscope_noise_density),
      grw_(gyroscope_random_walk),
      and_(accelerometer_noise_density),
      arw_(accelerometer_random_walk),
      freq_(frequency) {
  gyro_noise_cov_ = Matrix3T::Identity() * gnd_ * gnd_ * freq_;
  acc_noise_cov_ = Matrix3T::Identity() * and_ * and_ * freq_;

  gyro_random_walk_cov_ = Matrix3T::Identity() * grw_ * grw_ / freq_;
  acc_random_walk_cov_ = Matrix3T::Identity() * arw_ * arw_ / freq_;
}

const Isometry3T& ImuCalibration::rig_from_imu() const { return rig_from_imu_; }

const Matrix3T& ImuCalibration::gyro_noise() const { return gyro_noise_cov_; }

const Matrix3T& ImuCalibration::acc_noise() const { return acc_noise_cov_; }

const Matrix3T& ImuCalibration::gyro_random_walk() const { return gyro_random_walk_cov_; }

const Matrix3T& ImuCalibration::acc_random_walk() const { return acc_random_walk_cov_; }

float ImuCalibration::frequency() const { return freq_; }

}  // namespace cuvslam::imu
