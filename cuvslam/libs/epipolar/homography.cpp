
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

#include "epipolar/homography.h"

#include "common/log.h"
#include "math/nonlinear_optimization.h"

#include "epipolar/camera_projection.h"
#include "epipolar/points_normalization.h"

namespace cuvslam::epipolar {

ComputeHomography::ReturnCode ComputeHomography::findHomography(Matrix3T& homography) {
  if (points1_.size() != points2_.size()) {
    return ReturnCode::ContainersOfDifferentSize;
  }

  if (points1_.size() < MINIMUM_NUMBER_OF_POINTS) {
    return ReturnCode::NotEnoughPoints;
  }

  const size_t numPoints = points1_.size();

  NormalizationTransform t1(points1_);
  NormalizationTransform t2(points2_);

  if (!t1.isValid() || !t2.isValid()) {
    return ReturnCode::InvalidScale;
  }

  // Normalize both 2D points containers.
  Vector2TVector scaledPoints1(numPoints), scaledPoints2(numPoints);

  std::transform(points1_.cbegin(), points1_.cend(), scaledPoints1.begin(), t1);
  std::transform(points2_.cbegin(), points2_.cend(), scaledPoints2.begin(), t2);

  // Build the A matrix
  MatrixX9T matA(numPoints * 2, 9);

  auto x1 = scaledPoints1.cbegin();
  auto x2 = scaledPoints2.cbegin();
  auto x1End = scaledPoints1.cend();

  for (size_t index = 0; x1End != x1; ++x1, ++x2) {
    matA.row(index++) << 0, 0, 0, -x1->x(), -x1->y(), -1, x2->y() * x1->x(), x2->y() * x1->y(), x2->y();

    matA.row(index++) << x1->x(), x1->y(), 1, 0, 0, 0, -x2->x() * x1->x(), -x2->x() * x1->y(), -x2->x();
  }

  const Matrix9T matATA = matA.transpose() * matA;
  Eigen::JacobiSVD<Matrix9T> svd(matATA, Eigen::ComputeFullV);
  assert(svd.info() == Eigen::Success && "Singular value decomposition failed");
  const float singularValue7 = float(svd.singularValues()(7));

  if (std::abs(singularValue7) < epsilon()) {
    return ReturnCode::InvalidHomographyMatrix;
  }

  const Vector9T v = svd.matrixV().rightCols(1).template cast<float>();
  Matrix3T homographyCandidate(v.data());
  homographyCandidate *= std::copysign(float(1), homographyCandidate(2, 2));
  homographyCandidate = t2.calcDenormMatrix() * homographyCandidate.transpose() * t1.calcNormMatrix();
  const Vector3T singulars = homographyCandidate.jacobiSvd().singularValues();

  // homography should be an invertible transformation
  if (std::abs(singulars.prod()) < epsilon()) {
    return ReturnCode::InvalidHomographyMatrix;
  }

  homography = homographyCandidate / singulars[1];

  return ReturnCode::Success;
}

// Gold Standard method, HZ Algorithm 4.3, p.114
class GoldStandardHomography : public Eigen::DenseFunctor<float> {
public:
  using Base = Eigen::DenseFunctor<float>;
  using Base::Base;
  using InputType = Vector9T;  // TODO: make InputType as Matrix3T

  GoldStandardHomography(Vector2TPairVectorCIt p2DBegin, Vector2TPairVectorCIt p2DEnd)
      : Base(9, (int)std::distance(p2DBegin, p2DEnd) * 4), p2DBegin_(p2DBegin), p2DEnd_(p2DEnd) {}

  int operator()(const InputType& input, ValueType& fvec) const {
    size_t fvecIndx = 0;
    const Matrix3T H(input.data());

    for (auto tt = p2DBegin_; tt != p2DEnd_; tt++) {
      Vector3T x1p3 = H.inverse() * tt->second.homogeneous();
      Vector3T x2p3 = H * tt->first.homogeneous();
      Vector2T x1p(x1p3.x() / x1p3.z(), x1p3.y() / x1p3.z());
      Vector2T x2p(x2p3.x() / x2p3.z(), x2p3.y() / x2p3.z());

      Vector2T x1corr = tt->first - x1p;
      Vector2T x2corr = tt->second - x2p;

      fvec(fvecIndx++) = x1corr.x();
      fvec(fvecIndx++) = x1corr.y();
      fvec(fvecIndx++) = x2corr.x();
      fvec(fvecIndx++) = x2corr.y();
    };

    return 0;
  }

private:
  Vector2TPairVectorCIt p2DBegin_, p2DEnd_;

};  // class GoldStandardHomography

bool GoldStandardHomographyEstimate(const Vector2TPairVector& sampleSequence, Matrix3T& homography) {
  using GSEN = Eigen::NumericalDiff<GoldStandardHomography>;
  GSEN functor(sampleSequence.cbegin(), sampleSequence.cend());
  Eigen::LevenbergMarquardt<GSEN> lm(functor);

  GSEN::InputType h(homography.data());
  lm.setMaxfev(20);  // limit iterations
  auto retCode = lm_minimize(lm, h);

  if (lm.fnorm() > 0.0001f * functor.values()) {
    TraceMessage("LM fnorm: %f", lm.fnorm() / functor.values());
  }

  if (retCode < 1 || retCode > 4) {
    return false;
  }

  Matrix3T homographyCandidate(h.data());
  const Vector3T singulars = homographyCandidate.jacobiSvd().singularValues();

  // homography should be a well-conditioned invertible transformation
  if (std::abs(singulars[2] / singulars[0]) < sqrt_epsilon()) {
    return false;
  }

  homography = homographyCandidate / singulars[1];

  return true;
}

}  // namespace cuvslam::epipolar
