
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

#include <map>

#include "common/frame_id.h"
#include "common/include_eigen.h"

namespace cuvslam {
enum class AngleUnits { Degree, Radian };

// class Angle stores angular values. Value always stored in radians but we provide mechanism to create it with and/or
// extract values in both radians and degrees.
template <typename _Scalar>
class Angle {
public:
  using Scalar = _Scalar;

  Angle() = default;
  Angle(const Angle&) = default;

  Angle(const Scalar& v, const AngleUnits e) : value_(v * ConversionCoeffFromTo(e, AngleUnits::Radian)) {
    // Sanity check: assert on values outside of the range (-4*PI, 4*PI)
    // most likely it indicated an error and either value in degrees was used as radian value or non-angular value or
    // result was used
    assert(value_ > Scalar(-EIGEN_PI * 4) && value_ < Scalar(EIGEN_PI * 4));
  }

  Angle(const int& i) : value_(Scalar(i)) {}
  Angle(const Scalar& v) : Angle(v, AngleUnits::Radian) {}

  const Scalar& getValue() const { return value_; }  // return angle in radians
  operator const Scalar&() const { return value_; }  // return angle in radians
  Scalar getValue(const AngleUnits e) const {
    return value_ * ConversionCoeffFromTo(AngleUnits::Radian, e);
  }  // return angle value in specified units

  const Angle& normalize() {
    value_ = Normalize(value_);
    return *this;
  }

  Angle getNormalized() const { return Normalize(value_); }

  const Angle& operator+=(const Scalar& v) {
    value_ += v;
    return *this;
  }
  const Angle& operator-=(const Scalar& v) {
    value_ -= v;
    return *this;
  }
  const Angle& operator*=(const Scalar& v) {
    value_ *= v;
    return *this;
  }
  const Angle& operator/=(const Scalar& v) {
    value_ /= v;
    return *this;
  }

  static Scalar ConversionCoeffFromTo(const AngleUnits from, const AngleUnits to) {
    return from == to ? Scalar(1) : from == AngleUnits::Degree ? Scalar(EIGEN_PI / 180) : Scalar(180 / EIGEN_PI);
  }

  // modulate angular value to be in [-PI, +PI) range
  static Scalar Normalize(const Scalar v) {
    const Scalar val = std::fmod(v + Scalar(EIGEN_PI), Scalar(EIGEN_PI * 2));
    return ((val < 0) ? val + Scalar(EIGEN_PI * 2) : val) - Scalar(EIGEN_PI);
  }

private:
  Scalar value_;
};

template <typename _Scalar, int _M, int _N>
using MatrixMN = Eigen::Matrix<_Scalar, _M, _N>;

template <typename _Scalar, int _Dim>
using RowVector = Eigen::Matrix<_Scalar, 1, _Dim>;
template <typename _Scalar, int _Dim>
using ColumnVector = Eigen::Matrix<_Scalar, _Dim, 1>;
template <typename _Scalar, int _Dim>
using Vector = ColumnVector<_Scalar, _Dim>;
template <typename _Scalar, int _Dim>
using Translation = Eigen::Translation<_Scalar, _Dim>;
template <typename _Scalar, int _Dim>
using SquareMatrix = Eigen::Matrix<_Scalar, _Dim, _Dim>;
template <typename _Scalar, int _Dim>
using MatrixXN = Eigen::Matrix<_Scalar, Eigen::Dynamic, _Dim>;

template <typename _Scalar>
using AngleVector3 = Vector<Angle<_Scalar>, 3>;
template <typename _Scalar>
using Vector2 = Vector<_Scalar, 2>;
template <typename _Scalar>
using Vector3 = Vector<_Scalar, 3>;
template <typename _Scalar>
using Vector4 = Vector<_Scalar, 4>;
template <typename _Scalar>
using Vector6 = Vector<_Scalar, 6>;
template <typename _Scalar>
using Vector7 = Vector<_Scalar, 7>;
template <typename _Scalar>
using Vector9 = Vector<_Scalar, 9>;
template <typename _Scalar>
using VectorX = Vector<_Scalar, Eigen::Dynamic>;
template <typename _Scalar>
using Translation3 = Translation<_Scalar, 3>;
template <typename _Scalar>
using Matrix2 = SquareMatrix<_Scalar, 2>;
template <typename _Scalar>
using Matrix3 = SquareMatrix<_Scalar, 3>;
template <typename _Scalar>
using Matrix4 = SquareMatrix<_Scalar, 4>;
template <typename _Scalar>
using Matrix5 = SquareMatrix<_Scalar, 5>;
template <typename _Scalar>
using Matrix6 = SquareMatrix<_Scalar, 6>;
template <typename _Scalar>
using Matrix7 = SquareMatrix<_Scalar, 7>;
template <typename _Scalar>
using Matrix9 = SquareMatrix<_Scalar, 9>;
template <typename _Scalar>
using Matrix15 = SquareMatrix<_Scalar, 15>;
template <typename _Scalar>
using MatrixX = SquareMatrix<_Scalar, Eigen::Dynamic>;
template <typename _Scalar>
using MatrixX9 = MatrixXN<_Scalar, 9>;

template <typename _VecType, typename _Scalar = typename _VecType::Scalar, int _Dim = _VecType::SizeAtCompileTime,
          typename _AngleVecType = Vector<Angle<_Scalar>, _Dim>>
_AngleVecType AngleVector(const _VecType& v, AngleUnits e) {
  return (v * Angle<_Scalar>::ConversionCoeffFromTo(e, AngleUnits::Radian)).template cast<Angle<_Scalar>>();
}

template <typename _VecType, typename _Scalar = typename _VecType::Scalar, int _Dim = _VecType::SizeAtCompileTime,
          typename _AngleVecType = Vector<Angle<_Scalar>, _Dim>>
_AngleVecType AngleVector(const _VecType& v) {
  return v.template cast<Angle<_Scalar>>();
}

template <typename _Scalar>
class Quaternion : public Eigen::Quaternion<_Scalar> {
  using Base = Eigen::Quaternion<_Scalar>;

public:
  using Base::Base;
  Quaternion() = default;
  Quaternion(const Eigen::MatrixBase<AngleVector3<_Scalar>>& v)
      : Base(Eigen::AngleAxis<_Scalar>(v.z(), cuvslam::Vector3<_Scalar>::UnitZ()) *
             Eigen::AngleAxis<_Scalar>(v.y(), cuvslam::Vector3<_Scalar>::UnitY()) *
             Eigen::AngleAxis<_Scalar>(v.x(), cuvslam::Vector3<_Scalar>::UnitX())) {}

  Quaternion(const cuvslam::Vector3<_Scalar>& v, const AngleUnits e) : Quaternion(AngleVector(v, e)) {}

  /// computes a quaternion from the 3-element small angle approximation theta
  static Quaternion FromSmallAngle(const cuvslam::Vector3<_Scalar>& theta) {
    const _Scalar q_squared = theta.squaredNorm() / _Scalar(4.0);

    if (q_squared < 1) {
      return Quaternion(sqrt(_Scalar(1) - q_squared), theta[0] * _Scalar(0.5), theta[1] * _Scalar(0.5),
                        theta[2] * _Scalar(0.5));
    } else {
      const _Scalar w = _Scalar(1.0) / sqrt(_Scalar(1) + q_squared);
      const _Scalar f = w * _Scalar(0.5);
      return Quaternion(w, theta[0] * f, theta[1] * f, theta[2] * f);
    }
  }
};

template <typename _Vector3, typename _Scalar = typename _Vector3::Scalar>
Matrix3<_Scalar> SkewSymmetric(const _Vector3& v, const _Scalar d = 0) {
  return (Matrix3<_Scalar>() << d, -v(2), +v(1), +v(2), d, -v(0), -v(1), +v(0), d).finished();
}

template <typename _Matrix, typename _Scalar = typename _Matrix::Scalar>
bool IsRotationalMatrix(const Eigen::MatrixBase<_Matrix>& rot, const _Scalar eps) {
  assert(rot.cols() == rot.rows());
  const bool isDetOne = Eigen::internal::isApprox(_Scalar(1), rot.determinant(), eps);
  assert(!Eigen::internal::isApprox(_Scalar(-1), rot.determinant(), eps) &&
         "determinant == -1 is indication of improper rotation matrix (rotoreflection)");
  const bool isRTRIdentity = (rot.transpose() * rot).isIdentity(eps);
  return isDetOne && isRTRIdentity;
}

template <typename _T = float>
constexpr _T epsilon() {
  return std::numeric_limits<float>::epsilon();
}

template <typename _T = float>
constexpr _T sqrt_epsilon() {
  return std::sqrt(epsilon<_T>());
}

template <typename _Scalar>
bool IsApprox(const _Scalar x, const _Scalar y, const _Scalar eps = epsilon<_Scalar>()) {
  return Eigen::internal::isApprox<_Scalar>(x, y, eps);
}

template <typename _Scalar>
_Scalar AvoidZero(const _Scalar a, const _Scalar eps = epsilon<_Scalar>()) {
  return (std::abs(a) > eps ? a : std::copysign(eps, a));
}

using AngleT = Angle<float>;
using QuaternionT = Quaternion<float>;
using AngleVector3T = AngleVector3<float>;
using Vector4T = Vector4<float>;
using Vector6T = Vector6<float>;
using Vector7T = Vector7<float>;
using Vector9T = Vector9<float>;
using VectorXT = VectorX<float>;
using Translation3T = Translation3<float>;
using Matrix2T = Matrix2<float>;
using Matrix3T = Matrix3<float>;
using Matrix4T = Matrix4<float>;
using Matrix5T = Matrix5<float>;
using Matrix6T = Matrix6<float>;
using Matrix7T = Matrix7<float>;
using Matrix9T = Matrix9<float>;
using Matrix15T = Matrix15<float>;
using MatrixXT = MatrixX<float>;
using MatrixX9T = MatrixX9<float>;
using Rotation3T = QuaternionT;
using Index = Eigen::Index;
using Vector2N = Vector2<Index>;  // used to store discrete image coordinates
using FrameVector = std::vector<FrameId>;

//////////////
// Iterative Solver class
template <typename _Solver>
struct IterativeSolver : public _Solver {
  using Solver = _Solver;
  using MatrixType = typename Solver::MatrixType;
  using RowVectorType = typename Eigen::internal::plain_row_type<MatrixType>::type;
  using ColVectorType = typename Eigen::internal::plain_col_type<MatrixType>::type;
  using Scalar = typename Solver::Scalar;

  template <typename... _Args>
  IterativeSolver(const Eigen::MatrixBase<MatrixType>& m, _Args... args) : _Solver(m, args...), m_(m) {
    Solver::setThreshold(Scalar(m.diagonalSize()) * epsilon<Scalar>());
  }

  RowVectorType solve(const ColVectorType& b, const size_t maxNumOfIterations = 5,
                      const Scalar eps = epsilon<Scalar>()) {
    RowVectorType x(m_.cols());
    RowVectorType x_temp(m_.cols());
    x_temp.setZero();

    Scalar minNorm = std::numeric_limits<Scalar>::max();
    Scalar norm = b.stableNorm();
    ColVectorType b_delta = b;

    for (size_t counter = 0; norm < minNorm && counter < maxNumOfIterations; counter++) {
      minNorm = norm;
      x = x_temp;
      x_temp += Solver::solve(b_delta).transpose();
      b_delta = b - m_ * x_temp.transpose();
      norm = b_delta.stableNorm();

      if (norm < eps) {
        x = x_temp;
        break;
      }
    }

    return x;
  }

private:
  const Eigen::MatrixBase<MatrixType>& m_;
};

static const float PI = float(EIGEN_PI);

template <typename _Scalar, size_t _Dim>
constexpr std::array<_Scalar, _Dim> ToArray(const Eigen::Matrix<_Scalar, _Dim, 1>& v) {
  std::array<_Scalar, _Dim> a;
  std::copy_n(v.data(), _Dim, a.begin());
  return a;
}

}  // namespace cuvslam

/*
namespace std
{
template <class T, class I>
struct hash<cuvslam::_TypeUnique<T, I>>
{
    using argument_type = cuvslam::_TypeUnique<T, I>;
    using result_type = size_t ;
    result_type operator()(const argument_type& o) const
    {
        static std::hash<T> hash;
        return hash(o);
    }
};
}

*/
