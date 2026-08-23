
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

#include "epipolar/resectioning_utils.h"

#include "common/log.h"
#include "math/nonlinear_optimization.h"
#include "math/twist.h"

#include "epipolar/camera_projection.h"
#include "epipolar/resectioning_utils_internal.h"

namespace {

const auto RESIDUAL_NORM_THRESHOLD = 0.005f;
const auto PREALIGN_NORM_THRESHOLD = 0.01f;

}  // namespace

namespace cuvslam::epipolar {

template <int _Size, bool _useRotPrealign>
bool OptimizeCameraExtrinsicsExpMap(Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt,
                                    Vector2TVectorCIt points2DBeginIt, Vector2TVectorCIt points2DEndIt,
                                    Isometry3T& cameraMatrix, const float hDelta) {
  const SurveyedTrackingParams params(points3DBeginIt, points3DEndIt, points2DBeginIt, points2DEndIt, hDelta);

  if (params.pointCount3D() != params.pointCount2D()) {
    TraceError("The range of 2D and 3D points needs to be equal");
    return false;
  }

  if (params.pointCount3D() < 3) {
    TraceMessage("Surveyed tracking method operates on 3 points or more (points %u)", params.pointCount3D());
    return false;
  }

  using STRot = Eigen::NumericalDiff<SurveyedTrackingRotational<_Size>>;

  STRot functorRot(params);
  typename STRot::ValueType fvecRot(functorRot.values());
  typename STRot::InputType twistRot(cameraMatrix.inverse());

  functorRot(twistRot, fvecRot);
  const float normRot = fvecRot.stableNorm() / functorRot.values();

  auto retCode = Eigen::LevenbergMarquardtSpace::NotStarted;

  // if camera is too far, pre-align with rotational functor
  if (_useRotPrealign && params.pointCount3D() >= _Size && normRot > PREALIGN_NORM_THRESHOLD) {
    Eigen::LevenbergMarquardt<STRot> lm(functorRot);
    lm.setFtol(0.001f);
    lm.setXtol(0.001f);
    lm.setMaxfev(20);  // limit iterations
    retCode = lm_minimize(lm, twistRot);

    const float norm = lm.fnorm() / functorRot.values();

    TraceMessageIf(norm > PREALIGN_NORM_THRESHOLD,
                   "Failed to pre-align with rotational functor, norm = %f (%i), iters = %zu", norm,
                   functorRot.values(), lm.iterations());
  }

  using ST = SurveyedTracking<_Size>;
  ST functor(params);
  Eigen::LevenbergMarquardt<ST> lm(functor);

  lm.setMaxfev(_Size == 3 ? 80 : 20);  // rotation-only needs more iterations when the scene is very large
  retCode = lm_minimize(lm, twistRot);
  const float norm = lm.fnorm() / functor.values();

  if (norm > epsilon() && lm.info() != Eigen::Success &&
      retCode != Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation) {
    TraceError("LM failed with retCode %d and ComputationInfo %d LM norm: %e, fvec length: %d", retCode, lm.info(),
               norm, functor.values());
    return false;
  }

  cameraMatrix = twistRot.transform().inverse();

  if (norm > RESIDUAL_NORM_THRESHOLD) {
    // TraceMessage("LM norm: %f, fvec length: %d", norm, functor.values());
    return false;
  }

  return true;
}

template <bool _useRotPrealign>
bool OptimizeCameraExtrinsicsExpMapConstrained(Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt,
                                               Vector2TVectorCIt points2DBeginIt, Vector2TVectorCIt points2DEndIt,
                                               Isometry3T& cameraMatrix, const Vector3T& constraint,
                                               const float hDelta) {
  const SurveyedTrackingParams params(points3DBeginIt, points3DEndIt, points2DBeginIt, points2DEndIt, hDelta);

  if (params.pointCount3D() != params.pointCount2D()) {
    TraceError("The range of 2D and 3D points needs to be equal");
    return false;
  }

  if (params.pointCount3D() < 1) {
    TraceMessage("Constrained surveyed tracking method operates on 1 point or more (points %u)", params.pointCount3D());
    return false;
  }

  using STRot = Eigen::NumericalDiff<SurveyedTrackingScaleRot>;
  using ST = Eigen::NumericalDiff<SurveyedTrackingScale>;
  using InputType = typename STRot::InputType;

  InputType input = InputType::Constant(1, 0);
  auto retCode = Eigen::LevenbergMarquardtSpace::NotStarted;

  if (_useRotPrealign) {
    STRot functor(params, cameraMatrix, constraint);
    Eigen::LevenbergMarquardt<STRot> lm(functor);
    // lmRot.setFtol(0.001f);
    // lmRot.setXtol(0.001f);
    lm.setMaxfev(20);  // limit iterations
    retCode = lm_minimize(lm, input);
  }

  ST functor(params, cameraMatrix, constraint);
  Eigen::LevenbergMarquardt<ST> lm(functor);
  lm.setMaxfev(20);  // limit iterations
  retCode = lm_minimize(lm, input);

  const float norm = lm.fnorm() / functor.values();

  if (norm > epsilon() && lm.info() != Eigen::Success &&
      retCode != Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation) {
    TraceError("LM failed with retCode %d and ComputationInfo %d LM norm: %e, fvec length: %d", retCode, lm.info(),
               norm, functor.values());
    return false;
  } else if (norm > RESIDUAL_NORM_THRESHOLD) {
    // TraceMessage("LM norm: %f, fvec length: %d", norm, functor.values());
    return false;
  }

  // -1 = return camera to first, of two, key frame position
  if (input(0) > -1.0f) {
    cameraMatrix.pretranslate(constraint * input);
  }

  return true;
}

#define INSTANTIATE_RESECTIONING(Dim, UseRotPrealign)                                                        \
  template bool OptimizeCameraExtrinsicsExpMap<Dim, UseRotPrealign>(                                         \
      Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt, Vector2TVectorCIt points2DBeginIt, \
      Vector2TVectorCIt points2DEndIt, Isometry3T & cameraMatrix, const float hDelta)

INSTANTIATE_RESECTIONING(6, true);
INSTANTIATE_RESECTIONING(6, false);
INSTANTIATE_RESECTIONING(3, true);
INSTANTIATE_RESECTIONING(3, false);

#define INSTANTIATE_RESECTIONING_CONSTRAINED(UseRotPrealign)                                                 \
  template bool OptimizeCameraExtrinsicsExpMapConstrained<UseRotPrealign>(                                   \
      Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt, Vector2TVectorCIt points2DBeginIt, \
      Vector2TVectorCIt points2DEndIt, Isometry3T & cameraMatrix, const Vector3T& constraint, const float hDelta)

INSTANTIATE_RESECTIONING_CONSTRAINED(true);
INSTANTIATE_RESECTIONING_CONSTRAINED(false);

}  // namespace cuvslam::epipolar
