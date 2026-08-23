
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

#include <array>

#include "common/isometry.h"
#include "common/unaligned_types.h"
#include "common/vector_3t.h"

#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "epipolar/points_normalization.h"

namespace cuvslam::epipolar {

const float ROTATIONAL_THRESHOLD = 0.002f;

struct CameraMatrixNormal {
  CameraMatrixNormal() : cameraMatrix(Isometry3T::Identity()), normal(Vector3T::Zero()) {}
  CameraMatrixNormal(const Isometry3T& t, const Vector3T& n = Vector3T::Zero()) : cameraMatrix(t), normal(n) {}

  Isometry3T cameraMatrix;
  Vector3T normal;
};

using CameraMatrixNormalVector = std::array<CameraMatrixNormal, 4>;

/// @brief Class computing the homography from points in 2 images.
class ComputeHomography {
  const size_t MINIMUM_NUMBER_OF_POINTS = 4;

  static float clampIfZeroNeg(const float value) {
    assert(value >= 0 || std::abs(value) < 10 * epsilon());
    return (std::max)(float(0), value);
  }

  static float oppositeOfMinor(const Matrix3T& m, const size_t row, const size_t col) {
    assert(row < 3 && col < 3);

    const size_t x1 = (col == 0) ? 1 : 0;
    const size_t x2 = (col == 2) ? 1 : 2;
    const size_t y1 = (row == 0) ? 1 : 0;
    const size_t y2 = (row == 2) ? 1 : 2;

    return m(y1, x2) * m(y2, x1) - m(y1, x1) * m(y2, x2);
  }

  // computes R = H( I - (2/v)*te_star*ne_t ) - Malis, (20)
  static Matrix3T findRmatFrom_tstar_n(const Matrix3T& h, const Vector3T& tstar, const Vector3T& n, const float v) {
    assert(std::abs(v) > epsilon());
    return h * (Matrix3T::Identity() - (2 / v) * tstar * n.transpose());
  }

public:
  /// @brief Contains the enumeration of all valid return codes.
  enum class ReturnCode {
    Success,                    // The operation succeeded
    NotEnoughPoints,            // The operation failed because the object was not instantiated with enough points
    ContainersOfDifferentSize,  // The operation failed because the object was not instantiated with container of
                                // different size
    InvalidHomographyMatrix,    // The operation failed because a valid homography could not be generated
    InvalidScale  // The operation failed because it could not determine the proper scale of the input data
  };

  ComputeHomography(const Vector2TVector& points1, const Vector2TVector& points2)
      : points1_(points1), points2_(points2) {
    assert(points1_.size() == points2_.size() && "The number of points in image1 and image2 must be the same.");
    assert(points1_.size() >= MINIMUM_NUMBER_OF_POINTS && "ComputeHomography requires a minimum of 3 points.");
  }

  // returns normalized homography by second singular value
  ReturnCode findHomography(Matrix3T& homography);

  static bool IsRotationalHomography(Vector2TPairVectorCIt samplesBegin, Vector2TPairVectorCIt samplesEnd,
                                     const Matrix3T& homography) {
    Isometry3T camMat = DecomposeHomography(samplesBegin, samplesEnd, homography);
    const float rotNorm = (homography - camMat.linear()).norm();

    return (rotNorm < ROTATIONAL_THRESHOLD);

    // we don't use homography as Rotation matrix in our code anywhere but, if we decide to use it later, the
    // orthonormalization of homography will be prudent to make if IsRotationalHomography() returns true:
    // homography = Quaternion<float>(homography).normalized().matrix();
  }

  static bool IsNormalizedHomography(const Matrix3T& homography) {
    Eigen::JacobiSVD<Matrix3T> svd(homography);
    const float s1 = svd.singularValues()[1];
    const bool result = Eigen::internal::isApprox(s1, float(1), sqrt_epsilon());
    assert(result);
    return result;
  }

  /// DecomposeHomography() method below needed to check Malis math disregarding 2d
  /// points, just RT -> H is done correct, if 'yes' than H->RT will produce initial RT.
  static void DecomposeHomography(const Matrix3T& hNorm, CameraMatrixNormalVector& camMotions) {
    assert(ComputeHomography::IsNormalizedHomography(hNorm));
    // S = H'H - I
    Matrix3T s = hNorm.transpose() * hNorm - Matrix3T::Identity();

    // Compute nvectors
    const float m00 = clampIfZeroNeg(oppositeOfMinor(s, 0, 0));
    const float m11 = clampIfZeroNeg(oppositeOfMinor(s, 1, 1));
    const float m22 = clampIfZeroNeg(oppositeOfMinor(s, 2, 2));

    const float rtM00 = std::sqrt(m00);
    const float rtM11 = std::sqrt(m11);
    const float rtM22 = std::sqrt(m22);

    const float m01 = oppositeOfMinor(s, 0, 1);
    const float m12 = oppositeOfMinor(s, 1, 2);
    const float m02 = oppositeOfMinor(s, 0, 2);

    const float n0 = std::abs(s(0, 0));
    const float n1 = std::abs(s(1, 1));
    const float n2 = std::abs(s(2, 2));

    // find max index of |Sii|, i = 0, 1, 2
    const size_t indx = (n0 < n1) ? ((n1 < n2) ? 2 : 1) : ((n0 < n2) ? 2 : 0);

    Vector3T vbase, vdelta;

    switch (indx) {
      case 0:
        vbase << s(0, 0), s(0, 1), s(0, 2);
        vdelta << 0, rtM22, std::copysign(rtM11, m12);
        break;

      case 1:
        vbase << s(0, 1), s(1, 1), s(1, 2);
        vdelta << rtM22, 0, -std::copysign(rtM00, m02);
        break;

      case 2:
        vbase << s(0, 2), s(1, 2), s(2, 2);
        vdelta << std::copysign(rtM11, m01), rtM00, 0;
        break;

      default:
        assert(false);
        return;
    }

    const Vector3T npa(vbase + vdelta);
    const Vector3T npb(vbase - vdelta);

    const float traceS = s.trace();
    const float c = 1 + traceS - (m00 + m11 + m22);
    assert(c >= 0);
    const float v = 2 * std::sqrt(c);

    const float r_2 = 2 + traceS + v;
    const float nt_2 = 2 + traceS - v;

    assert(r_2 >= 0 && nt_2 >= 0);

    const float r = std::sqrt(r_2);
    const float n_t = std::sqrt(nt_2);

    const Vector3T na = (npa.norm() == 0) ? Vector3T::Zero() : npa.normalized();
    const Vector3T nb = (npb.norm() == 0) ? Vector3T::Zero() : npb.normalized();

    const float half_nt = 0.5f * n_t;
    const float esii_t_r = std::copysign(r, s(indx, indx));

    const Vector3T ta_star = half_nt * (esii_t_r * nb - n_t * na);  // (97) of Malis
    const Vector3T tb_star = half_nt * (esii_t_r * na - n_t * nb);  // (98) of Malis

    // Ra, ta, na
    const Isometry3T ra(findRmatFrom_tstar_n(hNorm, ta_star, na, v));
    const Translation3T ta(ra * ta_star);

    // Rb, tb, nb
    const Isometry3T rb(findRmatFrom_tstar_n(hNorm, tb_star, nb, v));
    const Translation3T tb(rb * tb_star);

    camMotions[0] = CameraMatrixNormal(ta * ra, na);
    camMotions[1] = CameraMatrixNormal(ta.inverse() * ra, -na);

    camMotions[2] = CameraMatrixNormal(tb * rb, nb);
    camMotions[3] = CameraMatrixNormal(tb.inverse() * rb, -nb);

    // IMPORTANT! IN THE CASE OF MOTION BEING TRANSLATION+ROTATION ABOVE THE PLANE
    // n_t is value of our translation (i.e. tran.norm()) divided by distance to the plane in the first of two
    // frames that we use to calculate Homography between - d*. I.E. THIS WAY WE CAN CALCULATE SCALE WITHOUT
    // ACTUALLY FINDING PLANE THROUGH THE CLOUD OF 3D TRIANGULATED POINTS (THAT CAN BE ABSENT) ON THE GROUND AND
    // KEEP SCALE OF OUR SOLUTION STABLE IN THE CASE OF CAR MOTION. ta.norm() = tb.norm = n_t.
  }

  static Isometry3T DecomposeHomography(Vector2TPairVectorCIt samplesBegin, Vector2TPairVectorCIt samplesEnd,
                                        const Matrix3T& homography) {
    assert(IsNormalizedHomography(homography));

    CameraMatrixNormalVector camMotions;
    DecomposeHomography(homography, camMotions);
    return FindBestPlanarHomographyFromSolutions(samplesBegin, samplesEnd, camMotions);
  }

  static Matrix3T ComposeHomograhpyForPlanarScene(const float originToPlane, const Matrix3T& rotation,
                                                  const Vector3T& planeNormal, Vector3T& translation) {
    // This is Malis definition, (3) of Malis. HZ defines it as R - tn^{T}.
    translation /= originToPlane;
    return rotation + (translation * (planeNormal.transpose()));
  }

private:
  static Isometry3T FindBestPlanarHomographyFromSolutions(Vector2TPairVectorCIt samplesBegin,
                                                          Vector2TPairVectorCIt samplesEnd,
                                                          const CameraMatrixNormalVector& homographySolutions) {
    assert(homographySolutions.size() == 4);

    auto makePair = [&](const size_t idx) -> std::pair<size_t, size_t> {
      return std::make_pair(
          idx, CountPointsInFrontOfCameras(samplesBegin, samplesEnd, homographySolutions[idx].cameraMatrix.inverse()));
    };

    std::array<std::pair<size_t, size_t>, 4> idxs = {{makePair(0), makePair(1), makePair(2), makePair(3)}};
    auto pred = [&](const std::pair<size_t, size_t> v1, const std::pair<size_t, size_t> v2) {
      return v1.second > v2.second;
    };
    std::partial_sort(idxs.begin(), idxs.begin() + 1, idxs.end(), pred);

    const size_t idx0 = idxs[0].first;
    Isometry3T camMat = homographySolutions[idx0].cameraMatrix;
    // float scaleToOne = camMat.translation().norm();
    // scaleToOne = (scaleToOne < epsilon()) ? epsilon() : scaleToOne;
    // camMat.translation() /= scaleToOne;

    return camMat;
  }

  const Vector2TVector& points1_;
  const Vector2TVector& points2_;
};

///-------------------------------------------------------------------------------------------------
/// @brief Decompose the homography induced by the transformation of a planar scene. This method
/// decomposes the homography as a set of solutions, each solution giving a candidate for the
/// rotation, translation induced by the homography and plane normal. The current implementation
/// uses the analytical algorithm developed by Malis (and implemented by OpenCV 3.0.0). The
/// method returns 4 solutions in the general case.
///
/// @{Malis07, author = { Ezio Malis and Manuel Vargas and Th�me Num and Ezio Malis and Manuel
/// Vargas }, title = { Deeper understanding of the homography decomposition for vision - based
/// control.INRIA Research Report #6303 }, year = { 2007 } }
///
/// @param homography         The homography induced by the transformation of a planar scene.
/// @param [in,out] solutions The solutions.
///-------------------------------------------------------------------------------------------------

inline float ComputeHomographyResidual(const Vector2T& pt1, const Vector2T& pt2, const Matrix3T& homography) {
  //    assert(ComputeHomography::IsNormalizedHomography(homography));
  return (Project3DPoint<2>(homography * pt1.homogeneous()) - pt2).norm();
}

bool GoldStandardHomographyEstimate(const Vector2TPairVector& sampleSequence, Matrix3T& homography);

}  // namespace cuvslam::epipolar
