
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

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

namespace cuvslam::math {

void Exp(Isometry3T&, const Vector6T& twist);
void Log(Vector6T&, const Isometry3T& m);

// Returns a rotation: exp(skew(omega))
void Exp(Isometry3T&, const Vector3T& omega);
void Log(Vector3T&, const Isometry3T& m);

void Exp(Matrix3T& result, const Vector3T& omega);
void Log(Vector3T& result, const Matrix3T& m);

template <class Derived>
Eigen::Matrix3f Skew(const Eigen::MatrixBase<Derived>& x) {
  Eigen::Matrix3f s;
  s << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  return s;
}

template <class Derived, class Transform>
void Adjoint(Eigen::MatrixBase<Derived>& ad, const Transform& transformation) {
  const Matrix3T& rotation = transformation.linear();
  ad.template block<3, 3>(0, 0) = rotation;
  ad.template block<3, 3>(0, 3).setZero();
  ad.template block<3, 3>(3, 0) = Skew(transformation.translation()) * rotation;
  ad.template block<3, 3>(3, 3) = rotation;
}

template <int _Size>
class VectorTwist : public Vector<float, _Size> {
public:
  static_assert(_Size == 3 || _Size == 6, "Only vector of length 3 and 6 supported by this class.");

  using Base = Vector<float, _Size>;
  enum { Size = _Size };
  using Base::Base;

  VectorTwist() = default;
  VectorTwist(const Isometry3T& t) : Base(LogTransformation(t)) {}
  VectorTwist(const Vector<float, _Size>& t) : Base(t) {}

  VectorTwist operator+=(const VectorTwist& rhs) {
    const Isometry3T& t = rhs.transform() * transform();
    *this = VectorTwist(t);
    return *this;
  }
  VectorTwist operator-=(const VectorTwist& rhs) { return operator+=(-rhs); }
  VectorTwist operator+(const VectorTwist& rhs) const { return (VectorTwist(*this) += rhs); }
  VectorTwist operator-(const VectorTwist& rhs) const { return (VectorTwist(*this) += -rhs); }
  VectorTwist operator-() const { return VectorTwist(-static_cast<Base>(*this)); }

  Isometry3T transform() const { return ExpTransformation(static_cast<Base>(*this)); }

private:
  static Isometry3T ExpTransformation(const Base& twist) {
    Isometry3T m;
    Exp(m, twist);
    return m;
  }

  static Base LogTransformation(const Isometry3T& trans) {
    Base result;
    Log(result, trans);
    return result;
  }
};

using VectorTwistT = VectorTwist<6>;

template <int _Size>
class TwistDerivativesOld {
public:
  using PointDerivs = MatrixMN<float, 2, 3>;
  using PointDerivsTransposed = MatrixMN<float, 3, 2>;
  using CameraDerivs = MatrixMN<float, 2, _Size>;
  using CameraDerivsTransposed = MatrixMN<float, _Size, 2>;

private:
  using TempDerivs = MatrixMN<float, 3, _Size>;

  PointDerivs fm_;
  TempDerivs sm_;

public:
  TwistDerivativesOld() {
    fm_.leftCols(2) = Matrix2T::Identity();

    if (sm_.cols() == 6) {
      sm_.rightCols(3) = -1 * Matrix3T::Identity();
    }
  }

  void calcDerivs(const Vector3T& pt3DInWorldCoords, const Isometry3T& invCameraMat, CameraDerivs& derivsCam,
                  PointDerivs& derivsPoint) {
    const float invZ = calcDerivs(pt3DInWorldCoords, invCameraMat, derivsCam);
    derivsPoint = invZ * (fm_ * invCameraMat.linear());
  }

  float calcDerivs(const Vector3T& pt3DInWorldCoords, const Isometry3T& invCameraMat, CameraDerivs& derivsCam) {
    const Vector3T ptInLocalCoords = invCameraMat * pt3DInWorldCoords;
    const float invZ = 1 / AvoidZero(ptInLocalCoords.z());
    fm_.rightCols(1) = ptInLocalCoords.head(2) * -invZ;
    sm_.leftCols(3) = SkewSymmetric(ptInLocalCoords);
    derivsCam = invZ * (fm_ * sm_);
    return invZ;
  }
};

template <int _Size>
class TwistDerivatives {
public:
  using PointDerivs = MatrixMN<float, 2, 3>;
  using PointDerivsTransposed = MatrixMN<float, 3, 2>;
  using CameraDerivs = MatrixMN<float, 2, _Size>;
  using CameraDerivsTransposed = MatrixMN<float, _Size, 2>;

  inline static void calcPointDerivs(const Vector3T& pt3DInWorldCoords, const Isometry3T& invCameraMat,
                                     const Matrix3T& invCameraRot, PointDerivs& derivsPoint) {
    const Vector3T ptInLocalCoords = invCameraMat * pt3DInWorldCoords;
    const float invZ = 1 / AvoidZero(ptInLocalCoords.z());

    const float x = ptInLocalCoords.x() * invZ, y = ptInLocalCoords.y() * invZ;

    calcPointDerivs(x, y, invZ, invCameraRot, derivsPoint);
  }

  inline static void calcCamPointDerivs(const Vector3T& pt3DInWorldCoords, const Isometry3T& invCameraMat,
                                        const Matrix3T& invCameraRot, CameraDerivs& derivsCam,
                                        PointDerivs& derivsPoint) {
    const Vector3T ptInLocalCoords = invCameraMat * pt3DInWorldCoords;
    const float invZ = 1 / AvoidZero(ptInLocalCoords.z());
    const float x = ptInLocalCoords.x() * invZ, y = ptInLocalCoords.y() * invZ;

    calcCamDerivs(x, y, invZ, derivsCam);
    calcPointDerivs(x, y, invZ, invCameraRot, derivsPoint);
  }

  inline static void calcCamDerivs(const Vector3T& pt3DInWorldCoords, const Isometry3T& invCameraMat,
                                   CameraDerivs& derivsCam) {
    const Vector3T ptInLocalCoords = invCameraMat * pt3DInWorldCoords;
    const float invZ = 1 / AvoidZero(ptInLocalCoords.z());
    const float x = ptInLocalCoords.x() * invZ, y = ptInLocalCoords.y() * invZ;

    calcCamDerivs(x, y, invZ, derivsCam);
  }

private:
  inline static void calcCamDerivs(float x, float y, float invZ, CameraDerivs& derivsCam) {
    derivsCam(0, 0) = x * y;
    derivsCam(1, 0) = y * y + 1;
    derivsCam(0, 1) = -x * x - 1;
    derivsCam(1, 1) = -derivsCam(0, 0);  // -xy
    derivsCam(0, 2) = y;
    derivsCam(1, 2) = -x;

    if (derivsCam.cols() > 3) {
      derivsCam(0, 3) = -invZ;
      derivsCam(1, 3) = 0;
      derivsCam(0, 4) = 0;
      derivsCam(1, 4) = -invZ;
      derivsCam(0, 5) = x * invZ;
      derivsCam(1, 5) = y * invZ;
    }
  }

  inline static void calcPointDerivs(float x, float y, float invZ, const Matrix3T& invCameraRot,
                                     PointDerivs& derivsPoint) {
    const auto& r = invCameraRot;

    for (int j = 0; j < 3; ++j) {
      derivsPoint(0, j) = (r(0, j) - r(2, j) * x) * invZ;
      derivsPoint(1, j) = (r(1, j) - r(2, j) * y) * invZ;
    }
  }
};

using TwistDerivativesT = TwistDerivatives<6>;
using TwistDerivativesOldT = TwistDerivativesOld<6>;

Matrix3T twist_left_jacobian(const Vector3T& twist);

Matrix3T twist_left_inverse_jacobian(const Vector3T& twist);

Matrix3T twist_right_jacobian(const Vector3T& twist);

Matrix3T twist_right_inverse_jacobian(const Vector3T& twist);

Matrix6T adjoint(const Isometry3T& pose);
Matrix6T inv_adjoint(const Isometry3T& pose);

Matrix6T se3_twist_left_jacobian(const Vector6T& twist);

Matrix6T se3_twist_left_inverse_jacobian(const Vector6T& twist);

Matrix6T se3_twist_right_jacobian(const Vector6T& twist);

Matrix6T se3_twist_right_inverse_jacobian(const Vector6T& twist);

}  // namespace cuvslam::math
