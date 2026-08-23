
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

#include "imu/imu_preintegration.h"

#include "common/rotation_utils.h"
#include "imu/linear_solver.h"
#include "imu/positive_matrix.h"

namespace cuvslam::sba_imu {

IMUPreintegration::IMUPreintegration() { Initialize(Vector3T::Zero(), Vector3T::Zero()); }

IMUPreintegration::IMUPreintegration(const Vector3T& gyro_bias, const Vector3T& acc_bias) {
  Initialize(gyro_bias, acc_bias);
}

IMUPreintegration::IMUPreintegration(const imu::ImuCalibration& calib, const imu::ImuMeasurementStorage& storage,
                                     const Vector3T& gyro_bias, const Vector3T& acc_bias, int64_t start_time_ns,
                                     int64_t end_time_ns) {
  Initialize(gyro_bias, acc_bias);

  storage.iterate_from_to(start_time_ns, end_time_ns,
                          [&](const imu::ImuMeasurement& m) { IntegrateNewMeasurement(calib, m); });
}

void IMUPreintegration::Initialize(const Vector3T& gyro_bias, const Vector3T& acc_bias) {
  gyro_bias_ = gyro_bias;
  acc_bias_ = acc_bias;

  gyro_bias_updated_ = gyro_bias;
  acc_bias_updated_ = acc_bias;

  gyro_bias_diff_.setZero();
  acc_bias_diff_.setZero();

  cov_matrix_.setZero();
  acc_random_walk_accum_cov_matrix_.setZero();
  gyro_random_walk_accum_cov_matrix_.setZero();

  info_matrix_ = std::nullopt;
  acc_random_walk_accum_info_matrix_ = std::nullopt;
  gyro_random_walk_accum_info_matrix_ = std::nullopt;

  dR.setIdentity();
  dV.setZero();
  dP.setZero();
  JRg.setZero();
  JVg.setZero();
  JVa.setZero();
  JPg.setZero();
  JPa.setZero();
  dT_s = 0;
}

void IMUPreintegration::update_state(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m, float delta_t_s) {
  Matrix9T A = Matrix9T::Identity();
  Eigen::Matrix<float, 9, 3> B = Eigen::Matrix<float, 9, 3>::Zero();
  Eigen::Matrix<float, 9, 3> C = Eigen::Matrix<float, 9, 3>::Zero();

  Vector3T lin_acc = m.linear_acceleration - acc_bias_;
  Vector3T ang_vel = m.angular_velocity - gyro_bias_;

  dP += dV * delta_t_s + 0.5f * dR * lin_acc * delta_t_s * delta_t_s;
  dV += dR * lin_acc * delta_t_s;

  A.block<3, 3>(3, 0) = -dR * delta_t_s * SkewSymmetric(lin_acc);
  A.block<3, 3>(6, 0) = -0.5f * dR * delta_t_s * delta_t_s * SkewSymmetric(lin_acc);
  A.block<3, 3>(6, 3) = Eigen::DiagonalMatrix<float, 3>(delta_t_s, delta_t_s, delta_t_s);

  B.block<3, 3>(3, 0) = dR * delta_t_s;
  B.block<3, 3>(6, 0) = 0.5f * dR * delta_t_s * delta_t_s;

  JPa += JVa * delta_t_s - 0.5f * dR * delta_t_s * delta_t_s;
  JPg += JVg * delta_t_s - 0.5f * dR * delta_t_s * delta_t_s * SkewSymmetric(lin_acc) * JRg;
  JVa -= dR * delta_t_s;
  JVg -= dR * delta_t_s * SkewSymmetric(lin_acc) * JRg;

  Matrix3T deltaR;
  math::Exp(deltaR, ang_vel * delta_t_s);
  Matrix3T rightJ = math::twist_right_jacobian(ang_vel * delta_t_s);

  dR = common::CalculateRotationFromSVD<float, 3>(dR * deltaR);

  A.block<3, 3>(0, 0) = deltaR.transpose();
  C.block<3, 3>(0, 0) = rightJ * delta_t_s;

  cov_matrix_ =
      A * cov_matrix_ * A.transpose() + B * calib.acc_noise() * B.transpose() + C * calib.gyro_noise() * C.transpose();

  acc_random_walk_accum_cov_matrix_ += calib.acc_random_walk();
  gyro_random_walk_accum_cov_matrix_ += calib.gyro_random_walk();

  JRg = deltaR.transpose() * JRg - rightJ * delta_t_s;
  dT_s += delta_t_s;

  info_matrix_ = std::nullopt;
  acc_random_walk_accum_info_matrix_ = std::nullopt;
  gyro_random_walk_accum_info_matrix_ = std::nullopt;
}

void IMUPreintegration::IntegrateNewMeasurement(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m) {
  float delta_t_s;
  if (measurements_.empty()) {
    delta_t_s = 1.f / calib.frequency();
  } else {
    delta_t_s = static_cast<float>(m.time_ns - measurements_.back().time_ns) * 1e-9f;
  }

  measurements_.push_back(m);

  update_state(calib, m, delta_t_s);
}

void IMUPreintegration::Reintegrate(const imu::ImuCalibration& calib) {
  Initialize(gyro_bias_updated_, acc_bias_updated_);

  float delta_t_s = 1.f / calib.frequency();
  for (size_t i = 0; i < measurements_.size(); i++) {
    const imu::ImuMeasurement& m = measurements_[i];
    if (i > 0) {
      const imu::ImuMeasurement& prev_m = measurements_[i - 1];
      delta_t_s = static_cast<float>(m.time_ns - prev_m.time_ns) * 1e-9f;
    }

    update_state(calib, m, delta_t_s);
  }
}

void IMUPreintegration::SetNewBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) {
  gyro_bias_updated_ = new_gyro_bias;
  acc_bias_updated_ = new_acc_bias;

  gyro_bias_diff_ = gyro_bias_updated_ - gyro_bias_;
  acc_bias_diff_ = acc_bias_updated_ - acc_bias_;
}

void IMUPreintegration::GetDeltaBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias,
                                     Vector3T& gyro_bias_delta, Vector3T& acc_bias_delta) const {
  gyro_bias_delta = new_gyro_bias - gyro_bias_;
  acc_bias_delta = new_acc_bias - acc_bias_;
}

Matrix3T IMUPreintegration::GetDeltaRotation(const Vector3T& new_gyro_bias) const {
  Matrix3T R;
  math::Exp(R, JRg * (new_gyro_bias - gyro_bias_));
  return common::CalculateRotationFromSVD<float, 3>(dR * R);
}

Vector3T IMUPreintegration::GetDeltaVelocity(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const {
  Vector3T dbg = new_gyro_bias - gyro_bias_;
  Vector3T dba = new_acc_bias - acc_bias_;
  return dV + JVg * dbg + JVa * dba;
}

Vector3T IMUPreintegration::GetDeltaPosition(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const {
  Vector3T dbg = new_gyro_bias - gyro_bias_;
  Vector3T dba = new_acc_bias - acc_bias_;
  return dP + JPg * dbg + JPa * dba;
}

float IMUPreintegration::GetDeltaT_s() const { return dT_s; }

void IMUPreintegration::GetDeltaBias(Vector3T& gyro_bias_delta, Vector3T& acc_bias_delta) const {
  gyro_bias_delta = gyro_bias_diff_;
  acc_bias_delta = acc_bias_diff_;
}

const Vector3T& IMUPreintegration::GetOriginalGyroBias() const { return gyro_bias_; }

const Vector3T& IMUPreintegration::GetOriginalAccBias() const { return acc_bias_; }

void IMUPreintegration::GetUpdatedBias(Vector3T& gyro_bias_updated, Vector3T& acc_bias_updated) const {
  gyro_bias_updated = gyro_bias_updated_;
  acc_bias_updated = acc_bias_updated_;
}

void IMUPreintegration::InfoMatrix(Matrix9T& info) const {
  if (info_matrix_) {
    info = *info_matrix_;
    return;
  }

  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(cov_matrix_, Matrix9T::Identity(), info) || !MakeMatrixPositive(info)) {
    info.setZero();
  }
  info_matrix_ = info;
}

void IMUPreintegration::InfoGyroRWMatrix(Matrix3T& info) const {
  if (gyro_random_walk_accum_info_matrix_) {
    info = *gyro_random_walk_accum_info_matrix_;
    return;
  }
  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(gyro_random_walk_accum_cov_matrix_, Matrix3T::Identity(), info) ||
      !MakeMatrixPositive(info)) {
    info.setZero();
  }
  gyro_random_walk_accum_info_matrix_ = info;
}

void IMUPreintegration::InfoAccRWMatrix(Matrix3T& info) const {
  if (acc_random_walk_accum_info_matrix_) {
    info = *acc_random_walk_accum_info_matrix_;
    return;
  }
  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(acc_random_walk_accum_cov_matrix_, Matrix3T::Identity(), info) ||
      !MakeMatrixPositive(info)) {
    info.setZero();
  }
  acc_random_walk_accum_info_matrix_ = info;
}

bool IMUPreintegration::empty() const { return measurements_.empty(); }

size_t IMUPreintegration::size() const { return measurements_.size(); }

}  // namespace cuvslam::sba_imu
