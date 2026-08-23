
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

#include <optional>

#include "common/imu_calibration.h"
#include "common/imu_measurement.h"
#include "common/isometry.h"
#include "common/state_serial.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_3t.h"
#include "math/twist.h"

namespace cuvslam::sba_imu {

const float GRAVITY_VALUE = 9.81;

class IMUPreintegration {
public:
  // covariance for [rotation, velocity, translation]
  Matrix9T cov_matrix_ = Matrix9T::Zero();
  Matrix3T acc_random_walk_accum_cov_matrix_ = Matrix3T::Zero();
  Matrix3T gyro_random_walk_accum_cov_matrix_ = Matrix3T::Zero();

  Matrix3T dR = Matrix3T::Identity();
  Vector3T dV = Vector3T::Zero();
  Vector3T dP = Vector3T::Zero();
  Matrix3T JRg = Matrix3T::Zero();
  Matrix3T JVg = Matrix3T::Zero();
  Matrix3T JVa = Matrix3T::Zero();
  Matrix3T JPg = Matrix3T::Zero();
  Matrix3T JPa = Matrix3T::Zero();
  float dT_s = 0;

public:
  IMUPreintegration();
  IMUPreintegration(const Vector3T& gyro_bias, const Vector3T& acc_bias);
  IMUPreintegration(const imu::ImuCalibration& calib, const imu::ImuMeasurementStorage& storage,
                    const Vector3T& gyro_bias, const Vector3T& acc_bias, int64_t start_time_ns, int64_t end_time_ns);

  void Initialize(const Vector3T& gyro_bias, const Vector3T& acc_bias);
  void IntegrateNewMeasurement(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m);
  void Reintegrate(const imu::ImuCalibration& calib);

  void SetNewBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias);
  void GetDeltaBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias, Vector3T& gyro_bias_delta,
                    Vector3T& acc_bias_delta) const;

  Matrix3T GetDeltaRotation(const Vector3T& new_gyro_bias) const;
  Vector3T GetDeltaVelocity(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const;
  Vector3T GetDeltaPosition(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const;

  float GetDeltaT_s() const;
  void GetDeltaBias(Vector3T& gyro_bias_delta, Vector3T& acc_bias_delta) const;

  const Vector3T& GetOriginalGyroBias() const;
  const Vector3T& GetOriginalAccBias() const;

  void GetUpdatedBias(Vector3T& gyro_bias_updated, Vector3T& acc_bias_updated) const;

  void InfoMatrix(Matrix9T& info) const;
  void InfoGyroRWMatrix(Matrix3T& info) const;
  void InfoAccRWMatrix(Matrix3T& info) const;

  bool empty() const;
  size_t size() const;

  const std::vector<imu::ImuMeasurement>& measurements() const { return measurements_; }

  void save_state(serial::Writer& w) const {
    w.write_tag(0x50494E54);  // "PINT"
    w.write_eigen(cov_matrix_);
    w.write_eigen(acc_random_walk_accum_cov_matrix_);
    w.write_eigen(gyro_random_walk_accum_cov_matrix_);
    w.write_eigen(dR);
    w.write_eigen(dV);
    w.write_eigen(dP);
    w.write_eigen(JRg);
    w.write_eigen(JVg);
    w.write_eigen(JVa);
    w.write_eigen(JPg);
    w.write_eigen(JPa);
    w.write_pod(dT_s);
    w.write_size(measurements_.size());
    for (const imu::ImuMeasurement& m : measurements_) {
      w.write_pod(m.time_ns);
      w.write_eigen(m.linear_acceleration);
      w.write_eigen(m.angular_velocity);
    }
    w.write_eigen(gyro_bias_);
    w.write_eigen(acc_bias_);
    w.write_eigen(gyro_bias_updated_);
    w.write_eigen(acc_bias_updated_);
    w.write_eigen(gyro_bias_diff_);
    w.write_eigen(acc_bias_diff_);
  }

  void load_state(serial::Reader& r) {
    r.expect_tag(0x50494E54, "IMUPreintegration");
    r.read_eigen(cov_matrix_);
    r.read_eigen(acc_random_walk_accum_cov_matrix_);
    r.read_eigen(gyro_random_walk_accum_cov_matrix_);
    r.read_eigen(dR);
    r.read_eigen(dV);
    r.read_eigen(dP);
    r.read_eigen(JRg);
    r.read_eigen(JVg);
    r.read_eigen(JVa);
    r.read_eigen(JPg);
    r.read_eigen(JPa);
    dT_s = r.read_pod<float>();
    measurements_.clear();
    const size_t num_measurements = r.read_size();
    measurements_.reserve(num_measurements);
    for (size_t i = 0; i < num_measurements; ++i) {
      imu::ImuMeasurement m;
      m.time_ns = r.read_pod<int64_t>();
      r.read_eigen(m.linear_acceleration);
      r.read_eigen(m.angular_velocity);
      measurements_.push_back(m);
    }
    r.read_eigen(gyro_bias_);
    r.read_eigen(acc_bias_);
    r.read_eigen(gyro_bias_updated_);
    r.read_eigen(acc_bias_updated_);
    r.read_eigen(gyro_bias_diff_);
    r.read_eigen(acc_bias_diff_);
    // Lazily recomputed from the restored covariances.
    info_matrix_ = std::nullopt;
    acc_random_walk_accum_info_matrix_ = std::nullopt;
    gyro_random_walk_accum_info_matrix_ = std::nullopt;
  }

private:
  std::vector<imu::ImuMeasurement> measurements_;

  void update_state(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m, float delta_t_s);

  // Values for the original bias (when integration was computed)
  Vector3T gyro_bias_ = Vector3T::Zero();
  Vector3T acc_bias_ = Vector3T::Zero();

  // Updated bias
  Vector3T gyro_bias_updated_ = Vector3T::Zero();
  Vector3T acc_bias_updated_ = Vector3T::Zero();

  // Dif between original and updated bias
  // This is used to compute the updated values of the preintegration
  Vector3T gyro_bias_diff_ = Vector3T::Zero();
  Vector3T acc_bias_diff_ = Vector3T::Zero();

  // internal cache
  mutable std::optional<Matrix9T> info_matrix_ = std::nullopt;
  mutable std::optional<Matrix3T> acc_random_walk_accum_info_matrix_ = std::nullopt;
  mutable std::optional<Matrix3T> gyro_random_walk_accum_info_matrix_ = std::nullopt;
};

}  // namespace cuvslam::sba_imu
