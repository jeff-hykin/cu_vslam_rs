
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

#include "math/twist.h"

namespace {

template <class T, int Dim>
void Exp(cuvslam::Isometry3T& result, const cuvslam::Vector<T, Dim>& twist) {
  using namespace cuvslam;

  const Vector3T w = twist.head(3);
  const float theta = (std::max)(w.norm(), epsilon());
  const float c = std::cos(theta);
  const float s_theta = std::sin(theta) / theta;
  const Vector3T n(w / theta);

  result.linear() = SkewSymmetric(w * s_theta, c) + (1 - c) * (n * n.transpose());
  if (Dim == 3) {
    result.translation() = Vector3T::Zero();
  } else {
    const Vector3T v = twist.tail(3);
    const Vector3T nxv = n.cross(v);
    result.translation() = v + ((1 - c) / theta * nxv + (1 - s_theta) * n.cross(nxv));
  }
  result.makeAffine();
}

template <class T, int Dim>
void Log(cuvslam::Vector<T, Dim>& result, const cuvslam::Isometry3T& trans) {
  using namespace cuvslam;

  const Matrix3T a(trans.linear() - Matrix3T::Identity());
  const Vector3T wvec = a.jacobiSvd(Eigen::ComputeFullV).matrixV().col(2);

  // assert((a * wvec).norm() < epsilon<float>() * 20);  // TODO: need more investigation
  assert(std::abs(wvec.norm() - 1) < epsilon() * 10);

  const Vector3T rvec(a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1));
  const float wmag = std::atan2(rvec.dot(wvec), a.trace() + 2);

  if (Dim == 3) {
    result.head(3) = wmag * wvec;
  } else {
    const float wmag_2 = wmag / 2;
    const Vector3T t = trans.translation();
    const Vector3T wxt = wvec.cross(t);

    Vector3T v =
        t + (std::abs(wmag) < epsilon() ? 0 : (1 - wmag_2 / std::tan(wmag_2))) * wvec.cross(wxt) - wmag_2 * wxt;

    result.head(3) = wmag * wvec;
    result.tail(3) = v;
  }
}

}  // namespace

namespace cuvslam::math {

void Exp(Matrix3T& result, const Vector3T& twist) {
  const float theta = (std::max)(twist.norm(), epsilon());
  const float c = std::cos(theta);
  const float s_theta = std::sin(theta) / theta;
  const Vector3T n(twist / theta);

  result = SkewSymmetric(twist * s_theta, c) + (1 - c) * (n * n.transpose());
}

void Log(Vector3T& result, const Matrix3T& m) {
  const Matrix3T a(m - Matrix3T::Identity());
  const Vector3T wvec = a.jacobiSvd(Eigen::ComputeFullV).matrixV().col(2);

  assert(std::abs(wvec.norm() - 1) < epsilon() * 10);

  const Vector3T rvec(a(2, 1) - a(1, 2), a(0, 2) - a(2, 0), a(1, 0) - a(0, 1));
  const float wmag = std::atan2(rvec.dot(wvec), a.trace() + 2);
  result = wmag * wvec;
}

void Exp(Isometry3T& result, const Vector6T& twist) { ::Exp(result, twist); }

void Exp(Isometry3T& result, const Vector3T& omega) { ::Exp(result, omega); }

void Log(Vector6T& result, const Isometry3T& m) { ::Log(result, m); }

void Log(Vector3T& result, const Isometry3T& m) { ::Log(result, m); }

Matrix3T twist_left_jacobian(const Vector3T& twist) {
  float phi = twist.norm();
  if (phi < sqrt_epsilon()) {
    return Matrix3T::Identity() + 0.5 * SkewSymmetric(twist);
  }
  Vector3T a = twist / phi;

  float wonderfull_limm = sinf(phi) / phi;
  float k = (1.f - cosf(phi)) / phi;

  return wonderfull_limm * Matrix3T::Identity() + (1 - wonderfull_limm) * a * a.transpose() + k * SkewSymmetric(a);
}

Matrix3T twist_left_inverse_jacobian(const Vector3T& twist) {
  float phi = twist.norm();
  if (phi < sqrt_epsilon()) {
    return Matrix3T::Identity() - 0.5 * SkewSymmetric(twist);
  }
  Vector3T a = twist / phi;
  float phi_half = phi * 0.5f;
  float cot_2 = phi_half / tanf(phi_half);

  return cot_2 * Matrix3T::Identity() + (1 - cot_2) * a * a.transpose() - phi_half * SkewSymmetric(a);
}

Matrix3T twist_right_jacobian(const Vector3T& twist) { return twist_left_jacobian(-twist); }

Matrix3T twist_right_inverse_jacobian(const Vector3T& twist) { return twist_left_inverse_jacobian(-twist); }

Matrix6T adjoint(const Isometry3T& pose) {
  Matrix6T out = Matrix6T::Zero();

  out.block<3, 3>(0, 0) = pose.linear();
  out.block<3, 3>(3, 3) = pose.linear();
  out.block<3, 3>(3, 0) = Skew(pose.translation()) * pose.linear();
  return out;
}

Matrix6T inv_adjoint(const Isometry3T& pose) {
  Matrix6T out = Matrix6T::Zero();

  Matrix3T Rt = pose.linear().transpose();
  out.block<3, 3>(0, 0) = Rt;
  out.block<3, 3>(3, 3) = Rt;
  out.block<3, 3>(3, 0) = -Rt * Skew(pose.translation());
  return out;
}

namespace {

Matrix3T compute_Q_left(const Vector6T& twist) {
  const auto w = twist.head(3);
  const auto v = twist.tail(3);
  const Matrix3T V = Skew(v);
  const Matrix3T W = Skew(w);

  Matrix3T Q;

  float phi = w.norm();
  const Matrix3T WV = W * V;
  const Matrix3T VW = V * W;
  const Matrix3T WVW = WV * W;

  float A = 1.f / 6.f;
  float B = 1.f / 24.f;
  float C = 1.f / 120.f;
  if (std::abs(phi) > 1e-2) {
    const float s = sinf(phi);
    const float c = cosf(phi);
    const float phi2 = phi * phi;
    const float phi3 = phi2 * phi;
    const float phi4 = phi3 * phi;
    const float phi5 = phi4 * phi;

    A = (phi - s) / phi3;
    B = (phi2 / 2 + c - 1) / phi4;
    C = 0.5 * ((2 + c) / phi4 - 3 * s / phi5);
  }
  Q = 0.5 * V + A * (WV + VW + WVW) + B * (W * WV + VW * W - 3 * WVW) + C * (WVW * W + W * WVW);
  return Q;
}

}  // namespace

Matrix6T se3_twist_left_jacobian(const Vector6T& twist) {
  Matrix6T out = Matrix6T::Zero();
  Matrix3T J_left = twist_left_jacobian(twist.head(3));
  out.block<3, 3>(0, 0) = J_left;
  out.block<3, 3>(3, 3) = J_left;
  out.block<3, 3>(3, 0) = compute_Q_left(twist);
  return out;
}

Matrix6T se3_twist_left_inverse_jacobian(const Vector6T& twist) {
  Matrix6T out = Matrix6T::Zero();
  Matrix3T J_inv_left = twist_left_inverse_jacobian(twist.head(3));
  out.block<3, 3>(0, 0) = J_inv_left;
  out.block<3, 3>(3, 3) = J_inv_left;
  out.block<3, 3>(3, 0) = -J_inv_left * compute_Q_left(twist) * J_inv_left;
  return out;
}

Matrix6T se3_twist_right_jacobian(const Vector6T& twist) { return se3_twist_left_jacobian(-twist); }

Matrix6T se3_twist_right_inverse_jacobian(const Vector6T& twist) { return se3_twist_left_inverse_jacobian(-twist); }

}  // namespace cuvslam::math
