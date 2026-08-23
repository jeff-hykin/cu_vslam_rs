
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

#include "epipolar/fundamental_matrix_utils.h"

#include "common/log.h"
#include "math/nonlinear_optimization.h"
#include "math/twist.h"

#include "epipolar/camera_selection.h"
#include "epipolar/points_normalization.h"

namespace cuvslam::epipolar {

// Gold Standard method, HZ Algorithm 11.3, p.285
class GoldStandardEssential : public Eigen::DenseFunctor<float> {
public:
  using Base = Eigen::DenseFunctor<float>;
  using Base::Base;
  using InputType = math::VectorTwist<6>;

  GoldStandardEssential(Vector2TPairVectorCIt p2DBegin, Vector2TPairVectorCIt p2DEnd);

  int operator()(const InputType& input, ValueType& fvec) const;

private:
  Vector2TPairVectorCIt p2DBegin_, p2DEnd_;
  const MatrixMN<float, 2, 3> S_ = MatrixMN<float, 2, 3>::Identity();
};

class RLM_Essential : public Eigen::DenseFunctor<float> {
public:
  using Base = Eigen::DenseFunctor<float>;
  using Base::Base;
  using InputType = math::VectorTwist<6>;

  RLM_Essential(Vector2TPairVectorCIt p2DBegin, Vector2TPairVectorCIt p2DEnd);

  int operator()(const InputType& input, ValueType& fvec) const;

private:
  Vector2TPairVectorCIt p2DBegin_, p2DEnd_;
};

ComputeFundamental::ComputeFundamental(const Vector2TVector& points1, const Vector2TVector& points2) {
  assert(points1.size() == points2.size() && "The number of points in image1 and image2 must be the same.");
  assert(points1.size() >= MINIMUM_NUMBER_OF_POINTS && "ComputeFundamental requires a minimum of 8 points.");
  returnCode_ = composeFundamentalSystem(points1, points2);
}

///-------------------------------------------------------------------------------------------------
/// @brief Computes the fundamental matrix from the set of 2D points. The fundamental matrix is
/// determined only up to a projective transform. The fundamental matrix is further scaled by its
/// last coefficient F(3,3) (if not too small) and normalize so that its Frobenius norm is 1.
///
/// @param [in,out] fundamental The fundamental matrix.
///
/// @return (@see ReturnCode).
///-------------------------------------------------------------------------------------------------
ComputeFundamental::ReturnCode ComputeFundamental::findFundamental(Matrix3T& fundamental) const {
  fundamental = fundamental_;

  Eigen::JacobiSVD<Matrix3T> svd(fundamental, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto s = svd.singularValues();

  if (!(std::abs(s(2)) < epsilon() && IsApprox(s(0), s(1), ComputeFundamental::EssentialThreshold()))) {
    return ReturnCode::NonEssential;
  }

  return returnCode_;
}

ComputeFundamental::ReturnCode ComputeFundamental::getStatus() const { return returnCode_; }

// This is for testing purposes only - will eventually be removed.
const Vector9T& ComputeFundamental::getMatATASingularValues() const { return singularValues_; }

bool ComputeFundamental::isPotentialHomography() const {
  return returnCode_ == ReturnCode::PotentialHomography || returnCode_ == ReturnCode::RuledQuadric;
}

constexpr float ComputeFundamental::EssentialThreshold() { return .005f; }

ComputeFundamental::ReturnCode ComputeFundamental::composeFundamentalSystem(const Vector2TVector& points1,
                                                                            const Vector2TVector& points2) {
  const size_t numPoints = points1.size();

  if (numPoints != points2.size()) {
    return ReturnCode::ContainersOfDifferentSize;
  }

  if (numPoints < MINIMUM_NUMBER_OF_POINTS) {
    return ReturnCode::NotEnoughPoints;
  }

  const bool isFromRansac = (numPoints == MINIMUM_NUMBER_OF_POINTS);

  NormalizationTransform t1(points1);
  NormalizationTransform t2(points2);

  if (!t1.isValid() || !t2.isValid()) {
    return ReturnCode::InvalidScale;
  }

  Vector2TVector normalizedPoints1(numPoints);
  Vector2TVector normalizedPoints2(numPoints);

  std::transform(points1.cbegin(), points1.cend(), normalizedPoints1.begin(), t1);
  std::transform(points2.cbegin(), points2.cend(), normalizedPoints2.begin(), t2);

  // Build the A matrix
  MatrixX9T matA(numPoints, 9);
  matA.setZero();

  for (size_t i = 0; i < numPoints; ++i) {
    matA.row(i) << normalizedPoints2[i].x() * normalizedPoints1[i].x(),
        normalizedPoints2[i].x() * normalizedPoints1[i].y(), normalizedPoints2[i].x(),
        normalizedPoints2[i].y() * normalizedPoints1[i].x(), normalizedPoints2[i].y() * normalizedPoints1[i].y(),
        normalizedPoints2[i].y(), normalizedPoints1[i].x(), normalizedPoints1[i].y(), 1;
  }

  auto svd = (numPoints == MINIMUM_NUMBER_OF_POINTS ? matA.transpose() * matA : matA).jacobiSvd(Eigen::ComputeFullV);
  singularValues_ = svd.singularValues();
  Matrix3T candidate = Matrix3T(svd.matrixV().rightCols(1).data()).transpose();
  const Index rank = svd.rank();

  auto svdF1 = candidate.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto singulars1 = svdF1.singularValues();

  singulars1(2) = 0.0f;  // Nullify the third eigen value as it is 0 by construction.
  candidate = svdF1.matrixU() * singulars1.asDiagonal() * svdF1.matrixV().transpose();

  candidate = t2.calcNormMatrix().transpose() * candidate * t1.calcNormMatrix();
  fundamental_ = candidate.normalized();

  if (isFromRansac) {
    return ReturnCode::Success;
  }

  const float thresh = singularValues_[0] * rank * epsilon();

  if (singularValues_[6] < thresh) {
    return ReturnCode::PotentialHomography;
  }

  if (singularValues_[7] < 2 * thresh) {
    // by Tian Lan, YiHong Wu and Zhanyi HU "Twisted Cubic: ..." rank 7 is usually contaminated Homography,
    // i.e. in reality rank 6. Clean way would be to report RuledQuadric and investigate is it really so
    // by calculating E matrix not in RANSAC and calculating matrix S using (2) of F.Kahl and R.Hartley
    // "Critical Curves ..." and checking that S is a) symmetric and b) points triangulated by pair of camera
    // matrices decomposed from E define a "quadric" S, i.e. PT*S*P = 0, where P - homogenious 3d coordinates
    // of the triangulated point.
    return ReturnCode::RuledQuadric;
  }

  auto svdF2 = candidate.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto singulars2 = svdF2.singularValues();

  // if (!Eigen::internal::isApprox(singulars2(0), singulars2(1), ComputeFundamental::EssentialThreshold()))
  //{
  //     TraceError(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Bad Essential computed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<");
  //     return ReturnCode::InvalidScale; // @TODO: need it's own return code
  // }

  singulars2(2) = 0.0f;  // Nullify the third eigen value as it is 0 by construction.
  candidate = svdF2.matrixU() * singulars2.asDiagonal() * svdF2.matrixV().transpose();

  // make bottom-right element always positive
  candidate *= std::copysign(float(1), candidate(2, 2));
  fundamental_ = candidate.normalized();

  return ReturnCode::Success;
}

void ExtractRotationTranslationFromEssential(const Matrix3T& e, Matrix3T& rot1, Matrix3T& rot2, Vector3T& t) {
  Eigen::JacobiSVD<Matrix3T> svd(e, Eigen::ComputeFullU | Eigen::ComputeFullV);

#ifdef RUNTIME_VERIFICATION  // test for essential
  auto s = svd.singularValues();
  // @TODO: need additional normalization of essential. For now set very lose threshold
  TraceErrorIf(
      !(std::abs(s(2)) < epsilon() * s.norm() && IsApprox(s(0), s(1), ComputeFundamental::EssentialThreshold())),
      "%f %f %f", s(0), s(1), s(2));
#endif

  const Matrix3T u = svd.matrixU() * std::copysign(float(1), svd.matrixU().determinant());
  const Matrix3T v = svd.matrixV() * std::copysign(float(1), svd.matrixV().determinant());
  const Matrix3T w = (Matrix3T() << 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f).finished();

  rot1 = u * w * v.transpose();
  rot2 = u * w.transpose() * v.transpose();
  t = u.col(2).normalized();
}

float ComputeQuadraticResidual(const Vector2T& pt1, const Vector2T& pt2, const Matrix3T& fundamental1To2) {
#if 1
  return (std::fabs(pt2.homogeneous().dot(fundamental1To2 * pt1.homogeneous())));
#else
  // compute geometric distance: HZ (11.9)
  float nominator = pt2.homogeneous().dot(fundamental1To2 * pt1.homogeneous());
  nominator *= nominator;

  const Vector2T l1 = (fundamental1To2 * pt1.homogeneous()).head(2);
  const Vector2T l2 = (fundamental1To2.transpose() * pt2.homogeneous()).head(2);
  float n1 = l1.squaredNorm();
  float n2 = l2.squaredNorm();
  float denominator = std::max(n1 + n2, epsilon());
  return std::sqrt(nominator / denominator);
#endif
}

///-------------------------------------------------------------------------------------------------
/// @brief This method finds the optimal camera matrix from all possible combinations of
/// rotations and translations provided by the user. This is the relative transform in
/// Hartley/Zisserman terms. The camera matrix is composed of a rotation and a translation. The
/// optimal camera matrix is defined as the camera matrix for which the inverse maximizes number of
/// reconstructed points are in front of the camera (in the camera z-negative half- space) and in
/// front of the reference camera (z-negative half-space of the world coordinate system). The
/// points are reconstructed from the pairs of 3D rays.
///-------------------------------------------------------------------------------------------------
bool FindOptimalCameraMatrixFromEssential(Vector2TPairVectorCIt points2DStart, Vector2TPairVectorCIt points2DEnd,
                                          const Matrix3T& essential, Isometry3T& relativeTransform,
                                          const size_t minCount) {
  Matrix3T rot1, rot2;
  Translation3T trans;

  ExtractRotationTranslationFromEssential(essential, rot1, rot2, trans.vector());

  const Isometry3T candidates[4] = {trans * Isometry3T(rot1), trans.inverse() * Isometry3T(rot1),
                                    trans * Isometry3T(rot2), trans.inverse() * Isometry3T(rot2)};

  size_t bestCount = minCount;
  int bestIdx = -1;

  for (int i = 0; i < 4; i++) {
    const size_t count = CountPointsInFrontOfCameras(points2DStart, points2DEnd, candidates[i].inverse());

    if (count > bestCount) {
      bestCount = count;
      bestIdx = i;
    }
  }

  assert(bestIdx < 4);

  if (bestIdx >= 0) {
    relativeTransform = candidates[bestIdx];
    return true;
  }

  return false;
}

GoldStandardEssential::GoldStandardEssential(Vector2TPairVectorCIt p2DBegin, Vector2TPairVectorCIt p2DEnd)
    : Base(6, (int)std::distance(p2DBegin, p2DEnd) * 4), p2DBegin_(p2DBegin), p2DEnd_(p2DEnd) {}

int GoldStandardEssential::operator()(const InputType& input, ValueType& fvec) const {
  // twist.transform() = inverse camera matrix. As we get 2nd->1st frame CV (=inversecuVSLAM) cameras transformation
  // the inverse of this transformation is what we want, i.e. 1st->2nd CV cameras transfer and Essential that
  // describes it. Which means we don't need to essential.transposeInPlace() after essential formed.
  size_t fvecIndx = 0;
  const Isometry3T transform = input.transform();
  const Matrix3T e = essential(transform);

  for (auto tt = p2DBegin_; tt != p2DEnd_; tt++) {
    Vector2T n1 = S_ * e * tt->second.homogeneous();
    Vector2T n2 = S_ * e.transpose() * tt->first.homogeneous();

    const Matrix2T essential2by2 = S_ * e * S_.transpose();

    const float a = n1.dot(essential2by2 * n2);
    const float b = 0.5f * (n1.squaredNorm() + n2.squaredNorm());
    const float c = tt->first.homogeneous().dot(e * tt->second.homogeneous());
    const float bbac = b * b - a * c;
    // assert(bbac > -float(10 * sqrt_epsilon<float>()));
    const float d = (bbac > 0) ? std::sqrt(bbac) : 0;

    const float failureThreshold = 2.0f * epsilon();
    const float bd = std::max(b + d, failureThreshold);

    const float lambda = c / bd;
    Vector2T dx1 = lambda * n1;
    Vector2T dx2 = lambda * n2;

    n1 -= essential2by2 * dx2;
    n2 -= essential2by2.transpose() * dx1;

    const float n1sqN = std::max(n1.squaredNorm(), failureThreshold);
    const float n2sqN = std::max(n2.squaredNorm(), failureThreshold);

#if 0
        // niter2 - faster but lower quality
        lambda = lambda * 2.0f * d / (n1sqN + n2sqN);
        dx1 = lambda * n1;
        dx2 = lambda * n2;
#else
    // niter1 - slower but higher quality
    dx1 = dx1.dot(n1) / n1sqN * n1;
    dx2 = dx2.dot(n2) / n2sqN * n2;
#endif
    const Vector3T x1m = tt->first.homogeneous() - S_.transpose() * dx1;
    const Vector3T x2m = tt->second.homogeneous() - S_.transpose() * dx2;

    const Vector2T x1corr = x1m.head(2) - tt->first;
    const Vector2T x2corr = x2m.head(2) - tt->second;

    fvec(fvecIndx++) = x1corr.x();
    fvec(fvecIndx++) = x1corr.y();
    fvec(fvecIndx++) = x2corr.x();
    fvec(fvecIndx++) = x2corr.y();
  };

  return 0;
}

bool GoldStandardEssentialEstimate(const Vector2TPairVector& sampleSequence, Isometry3T& relative2To1CameraTransform) {
  using GSEN = Eigen::NumericalDiff<GoldStandardEssential>;
  GSEN functor(sampleSequence.cbegin(), sampleSequence.cend());
  Eigen::LevenbergMarquardt<GSEN> lm(functor);
  GSEN::InputType twist(relative2To1CameraTransform);
  lm.setMaxfev(20);  // limit iterations
  auto retCode = lm_minimize(lm, twist);

  if (lm.fnorm() > 0.0001f * functor.values()) {
    TraceMessage("LM fnorm: %f", lm.fnorm() / functor.values());
  }

  if (retCode < 1 || retCode > 4) {
    return false;
  }

  relative2To1CameraTransform = twist.transform();

  return true;
}

RLM_Essential::RLM_Essential(Vector2TPairVectorCIt p2DBegin, Vector2TPairVectorCIt p2DEnd)
    : Base(6, (int)std::distance(p2DBegin, p2DEnd)), p2DBegin_(p2DBegin), p2DEnd_(p2DEnd) {}

int RLM_Essential::operator()(const InputType& input, ValueType& fvec) const {
  // twist.transform() = inverse camera matrix. As we get 2nd->1st frame CV (=inversecuVSLAM) cameras transformation
  // the inverse of this transformation is what we want, i.e. 1st->2nd CV cameras transfer and Essential that
  // describes it. Which means we don't need to essential.transposeInPlace() after essential formed.
  size_t fvecIndx = 0;
  const Isometry3T transform = input.transform();
  const Matrix3T e = essential(transform);

  for (auto tt = p2DBegin_; tt != p2DEnd_; tt++) {
    fvec(fvecIndx++) = ComputeQuadraticResidual(tt->first, tt->second, e);
  };

  return 0;
}

bool RLM_EssentialEstimate(const Vector2TPairVector& sampleSequence, Isometry3T& relative2To1CameraTransform) {
  using RLM = Eigen::NumericalDiff<RLM_Essential>;
  RLM functor(sampleSequence.cbegin(), sampleSequence.cend());
  Eigen::LevenbergMarquardt<RLM> lm(functor);
  RLM::InputType twist(relative2To1CameraTransform);
  lm.setMaxfev(20);  // limit iterations
  auto retCode = lm_minimize(lm, twist);

  if (lm.fnorm() > 0.0001f * functor.values()) {
    TraceMessage("LM fnorm: %f", lm.fnorm() / functor.values());
  }

  if (retCode < 1 || retCode > 4) {
    TraceMessage("RLM failed");
    return false;
  }

  relative2To1CameraTransform = twist.transform();

  return true;
}

}  // namespace cuvslam::epipolar
