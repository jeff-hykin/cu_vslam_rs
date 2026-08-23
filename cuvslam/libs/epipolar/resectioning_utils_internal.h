
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

#include "common/log.h"
#include "math/nonlinear_optimization.h"
#include "math/twist.h"

#include "epipolar/camera_projection.h"

#ifdef TEST_NEW_DERIVS
#include <iostream>
#endif

namespace cuvslam::epipolar {

// Angular prealign: compare the 3D landmark direction in camera space to the image ray (x, y, 1) (OpenCV +Z forward).
template <typename _Vector, typename _VectorOther, typename _Scalar = typename _Vector::Scalar>
_Scalar AngularDistance(const _Vector& v0, const _VectorOther& v1) {
  const _Scalar norm = v0.norm() * v1.norm();

  if (norm < epsilon()) {
    assert(0 && "product of norms is too small, it will lead to inaccurate results so return result as 0");
    return 0;  // std::numeric_limits<_Scalar>::signaling_NaN();
  }

  const _Scalar sine = (v0.cross(v1)).norm() / norm;
  const _Scalar cosine = v0.dot(v1) / norm;
  TraceMessageIf(!IsApprox(_Scalar(1), sine * sine + cosine * cosine, 20 * epsilon()), "norm = %f", norm);
  assert(IsApprox(_Scalar(1), sine * sine + cosine * cosine, 20 * epsilon()));
  return (sine > 0.5f) ? std::acos(cosine) : (cosine > 0) ? std::asin(sine) : _Scalar(PI) - std::asin(sine);
}

static float costFunctionHuber(const float delta, const float huber_b) {
  float res = delta;
  const float deltaAbs = std::abs(delta);

  if (deltaAbs >= huber_b) {
    res = std::copysign(std::sqrt(2 * huber_b * deltaAbs - huber_b * huber_b), delta);
  }

  return res;
}

struct SurveyedTrackingParams {
  Vector3TVectorCIt p3DBegin;
  Vector3TVectorCIt p3DEnd;
  Vector2TVectorCIt p2DBegin;
  Vector2TVectorCIt p2DEnd;
  float huberDelta;
  SurveyedTrackingParams(const SurveyedTrackingParams&) = default;
  SurveyedTrackingParams(Vector3TVectorCIt b3D, Vector3TVectorCIt e3D, Vector2TVectorCIt b2D, Vector2TVectorCIt e2D,
                         float hDelta)
      : p3DBegin(b3D), p3DEnd(e3D), p2DBegin(b2D), p2DEnd(e2D), huberDelta(hDelta) {
    assert(b3D < e3D);
    assert(b2D < e2D);
    assert(std::distance(b3D, e3D) == std::distance(b2D, e2D));
  }

  int pointCount3D() const { return static_cast<int>(std::distance(p3DBegin, p3DEnd)); }
  int pointCount2D() const { return static_cast<int>(std::distance(p2DBegin, p2DEnd)); }
};

template <int _Size, typename _Base = Eigen::DenseFunctor<float>>
class SurveyedTracking : public _Base {
  using InputDerivsCalculator = math::TwistDerivatives<_Size>;
  using CameraDerivs = typename InputDerivsCalculator::CameraDerivs;

public:
  using Base = _Base;
  using InputType = math::VectorTwist<_Size>;
  using ValueType = typename Base::ValueType;
  using JacobianType = typename Base::JacobianType;

  using Base::Base;

  SurveyedTracking(const SurveyedTrackingParams& params) : Base(_Size, params.pointCount3D() * 2), params_(params) {}

  SurveyedTracking(Vector3TVectorCIt b3D, Vector3TVectorCIt e3D, Vector2TVectorCIt b2D, Vector2TVectorCIt e2D,
                   const float hDelta)
      : SurveyedTracking(SurveyedTrackingParams(b3D, e3D, b2D, e2D, hDelta)) {}

  int operator()(const InputType& input, ValueType& fvec) const {
    const Isometry3T cameraFromWorld = input.transform();
    auto observation = params_.p2DBegin;
    int residualRow = 0;

    for (auto point = params_.p3DBegin; point != params_.p3DEnd; ++point, ++observation) {
      const Vector2T residual = *observation - Project3DPoint<2>(cameraFromWorld * *point);
      float rx = costFunctionHuber(residual.x(), params_.huberDelta);
      float ry = costFunctionHuber(residual.y(), params_.huberDelta);
      fvec(residualRow++) = rx;
      fvec(residualRow++) = ry;
    }

    return 0;
  }

  int df(const InputType& input, JacobianType& fjac) const {
    CameraDerivs derivs;
    const Isometry3T cameraFromWorld = input.transform();
    auto point = params_.p3DBegin;

    for (int i = 0; i < Base::values(); point++) {
      InputDerivsCalculator::calcCamDerivs(*point, cameraFromWorld, derivs);
      fjac.row(i++) = derivs.row(0);
      fjac.row(i++) = derivs.row(1);

      TestDerivatives(*point, cameraFromWorld, derivs);
    }

    return 0;
  }

private:
  const SurveyedTrackingParams params_;

  void TestDerivatives(const Vector3T& point, const Isometry3T& cameraFromWorld, const CameraDerivs& derivs) const {
#ifdef TEST_NEW_DERIVS
    CameraDerivs goldDerivs;
    cuvslam::math::TwistDerivativesOld<_Size> derivsCalculator;
    derivsCalculator.calcDerivs(point, cameraFromWorld, goldDerivs);

    if (!derivs.isApprox(goldDerivs)) {
      std::cerr << __FILE__ << ':' << __LINE__ << ": old and new twist deriv don't match!" << std::endl;
    }
#else
    (void)point;
    (void)cameraFromWorld;
    (void)derivs;
#endif
  }

};  // class SurveyedTracking main template

class SurveyedTrackingScale : public Eigen::DenseFunctor<float> {
public:
  using Base = Eigen::DenseFunctor<float>;
  using Base::InputType;
  using Base::JacobianType;
  using Base::ValueType;

  using Base::Base;

  SurveyedTrackingScale(const SurveyedTrackingParams& params, const Isometry3T& transform, const Vector3T& v)
      : Base(1, params.pointCount3D() * 2), params_(params), transform_(transform), scaleVector_(v) {
    assert(v.norm() > sqrt_epsilon());
  }

  SurveyedTrackingScale(Vector3TVectorCIt b3D, Vector3TVectorCIt e3D, Vector2TVectorCIt b2D, Vector2TVectorCIt e2D,
                        const Isometry3T& transform, const Vector3T& v, const float hDelta)
      : SurveyedTrackingScale(SurveyedTrackingParams(b3D, e3D, b2D, e2D, hDelta), transform, v) {}

  int operator()(const InputType& input, ValueType& fvec) const {
    const Isometry3T camMatInverse = (Translation3T(scaleVector_ * input) * transform_).inverse();
    auto tt = params_.p2DBegin;
    int fvecIndx = 0;

    float flipPenalty = (input(0) > -1.0f ? 1.0f : 1.0f + std::abs(input(0)) * 10.0f);

    for (auto p3dIt = params_.p3DBegin; p3dIt != params_.p3DEnd; p3dIt++) {
      const Vector2T residual = *tt++ - Project3DPoint<2>(camMatInverse * *p3dIt);
      fvec(fvecIndx++) = residual.x() * flipPenalty;
      fvec(fvecIndx++) = residual.y() * flipPenalty;
    }

    return 0;
  }

private:
  SurveyedTrackingParams params_;
  Isometry3T transform_;
  Vector3T scaleVector_ = Vector3T::Zero();

};  // class SurveyedTrackingScale

class SurveyedTrackingScaleRot : public Eigen::DenseFunctor<float> {
public:
  using Base = Eigen::DenseFunctor<float>;
  using Base::InputType;
  using Base::JacobianType;
  using Base::ValueType;

  using Base::Base;

  SurveyedTrackingScaleRot(const SurveyedTrackingParams& params, const Isometry3T& transform, const Vector3T& v)
      : Base(1, params.pointCount3D()), params_(params), transform_(transform), scaleVector_(v) {
    assert(v.norm() > sqrt_epsilon());
  }

  SurveyedTrackingScaleRot(Vector3TVectorCIt b3D, Vector3TVectorCIt e3D, Vector2TVectorCIt b2D, Vector2TVectorCIt e2D,
                           const Isometry3T& transform, const Vector3T& v, const float hDelta)
      : SurveyedTrackingScaleRot(SurveyedTrackingParams(b3D, e3D, b2D, e2D, hDelta), transform, v) {}

  int operator()(const InputType& input, ValueType& fvec) const {
    const Isometry3T camMatInverse = (Translation3T(scaleVector_ * input) * transform_).inverse();
    size_t fvecIndx = 0;
    auto tt = params_.p2DBegin;
    auto p3dIt = params_.p3DBegin;

    while (p3dIt != params_.p3DEnd) {
      fvec(fvecIndx++) = AngularDistance(camMatInverse * *p3dIt++, (*tt++).homogeneous());
    }

    return 0;
  }

private:
  SurveyedTrackingParams params_;
  Isometry3T transform_;
  Vector3T scaleVector_ = Vector3T::Zero();

};  // class SurveyedTrackingScaleRot

template <int _Size, typename _Base = Eigen::DenseFunctor<float>>
class SurveyedTrackingRotational : public _Base {
public:
  using Base = _Base;
  using ValueType = typename Base::ValueType;
  using InputType = math::VectorTwist<_Size>;
  using JacobianType = typename Base::JacobianType;

  using Base::Base;

  SurveyedTrackingRotational(const SurveyedTrackingParams& params)
      : Base(_Size, params.pointCount3D()), params_(params) {}

  SurveyedTrackingRotational(Vector3TVectorCIt b3D, Vector3TVectorCIt e3D, Vector2TVectorCIt b2D, Vector2TVectorCIt e2D,
                             const float hDelta)
      : SurveyedTrackingRotational(SurveyedTrackingParams(b3D, e3D, b2D, e2D, hDelta)) {}

  int operator()(const InputType& input, ValueType& fvec) const {
    Isometry3T camMatInverse = input.transform();
    size_t fvecIndx = 0;
    auto tt = params_.p2DBegin;
    auto p3dIt = params_.p3DBegin;

    while (p3dIt != params_.p3DEnd) {
      fvec(fvecIndx++) = AngularDistance(camMatInverse * *p3dIt++, (*tt++).homogeneous());
    }

    return 0;
  }

private:
  SurveyedTrackingParams params_;

};  // class SurveyedTrackingRotational

}  // namespace cuvslam::epipolar
