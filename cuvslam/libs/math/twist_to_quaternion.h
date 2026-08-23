
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

#include "math/twist.h"

namespace cuvslam::math {

class TwistToQuaternion : public Eigen::Matrix<float, 7, 6> {
public:
  using Matrix4x3 = Eigen::Matrix<float, 4, 3>;

  using Base = Eigen::Matrix<float, 7, 6>;
  using Base::Base;

  TwistToQuaternion() = delete;
  TwistToQuaternion(const VectorTwistT& twist) : Base(SetMatrix(twist)), qt_(set_m_qt(twist)) {}

  Vector7T set_m_qt(const VectorTwistT& twist) {
    if (twist.norm() == 0)  // Transformation of Identity
    {
      qt_[0] = 0.0f;
      qt_[1] = 0.0f;
      qt_[2] = 0.0f;
      qt_[3] = 1.0f;
      qt_[4] = 0.0f;
      qt_[5] = 0.0f;
      qt_[6] = 0.0f;
    } else {
      qt_ = getBaseMatrix() * twist;
    }

    return qt_;
  }

  const Base& getBaseMatrix() const { return static_cast<const Base&>(*this); }

  Vector7T getQuatTran() { return qt_; }

private:
  static Base SetMatrix(const VectorTwistT& twist) {
    Base mat;
    mat.setZero();

    if (twist.norm() == 0) {
      return mat;
    }

    const Vector3T w = twist.head(3);

    // no rotation, i.e. Quaternion should be (0,0,0,1)
    if (w.norm() == 0) {
      mat.block<3, 3>(4, 3) = Matrix3T::Identity();  // to reproduce translations as they are in the twist

      Vector3T t = twist.tail(3);
      const float biggestElementOfT = t.maxCoeff();

      for (size_t i = 0; i < 3; i++) {
        if (t[i] == biggestElementOfT) {
          mat(3, 3 + i) = 1.0f / biggestElementOfT;
          break;
        }
      }

      return mat;
    }

    Matrix4x3 J1;
    J1.setZero();

    float theta = (std::max)(w.norm(), epsilon());
    J1(0, 0) = J1(1, 1) = J1(2, 2) = 1.0f / theta;
    const float biggestElementOfW = w.maxCoeff();

    for (size_t i = 0; i < 3; i++) {
      if (w[i] == biggestElementOfW) {
        J1(3, i) = theta / biggestElementOfW;
        break;
      }
    }

    Matrix4T J2;
    J2.setZero();
    J2(0, 0) = J2(1, 1) = J2(2, 2) = std::sin(theta / 2.0f);
    J2(3, 3) = std::cos(theta / 2.0f) / theta;

    mat.block<4, 3>(0, 0) = J2 * J1;

    theta = w.norm();
    const float theta2 = theta * theta;
    const float thetaThresh = sqrt_epsilon();
    const float cwc = (theta < thetaThresh) ? 0.5f - theta2 / 24.0f : (1.0f - std::cos(theta)) / theta2;
    const float swc = (theta < thetaThresh) ? 1.0f / 6.0f - theta2 / 120.0f : (1.0f - std::sin(theta) / theta) / theta2;

    Matrix3T M = Matrix3T::Identity();
    M = M + cwc * SkewSymmetric(w);
    M = M + swc * SkewSymmetric(w) * SkewSymmetric(w);

    mat.block<3, 3>(4, 3) = M;
    return mat;
  }

  Vector7T qt_;
};

}  // namespace cuvslam::math
