
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

#include "epipolar/test/camera_points_reconstruction_integ_test.h"

#include "common/include_gtest.h"
#include "common/rotation_utils.h"
#include "epipolar/camera_projection.h"
#include "epipolar/camera_selection.h"
#include "epipolar/homography_ransac.h"
#include "epipolar/test/generate_visible_points_test.h"

namespace test::epipolar {
using namespace ::testing;

struct TestParam {
  const float noiseScale;
  const float ratioErrorStdVsOpt;

  TestParam(const float noise, const float ratio) : noiseScale(noise), ratioErrorStdVsOpt(ratio) {}
};

using TriangulationTest = TestWithParam<TestParam>;

TEST_P(TriangulationTest, OptimalVsStandart) {
  const Vector3T& points3DMinRange = Vector3T(3, 2, 5);
  const Vector3T& points3DMaxRange = Vector3T(13, 10, 16);

  const TestParam& p = GetParam();
  const Isometry3T cam1 = Isometry3T::Identity();
  Isometry3T cam2;

  if (!TryToGenerateCamera2Matrix(500, points3DMinRange, points3DMaxRange, cam2)) {
    EXPECT_TRUE(false);  // i.e. failed to generate camMatrix
  }

  // generate points
  Vector2TVectorVector points2D;  // per cam
  Vector3TVector points3D;
  generateVisiblePoints({cam1, cam2}, 100, 50, points3DMinRange, points3DMaxRange, points2D, &points3D);
  const size_t nPoints = points3D.size();
  assert(points2D.size() == 2 && points2D[0].size() == nPoints && points2D[1].size() == nPoints);

  // add noise
  for (size_t i = 0; i < nPoints; ++i) {
    points2D[0][i] += Vector2T::Random() * p.noiseScale;
    points2D[1][i] += Vector2T::Random() * p.noiseScale;
  }

  // run test
  Vector3T loc3dOpt, loc3dStd;
  float distOpt = 0;
  float distStd = 0;
  size_t nTriangulatedStd = 0;
  size_t nTriangulatedOpt = 0;

  for (size_t i = 0; i < nPoints; ++i) {
    const Vector2T& v0 = points2D[0][i];
    const Vector2T& v1 = points2D[1][i];
    const Vector3T& trueLoc = points3D[i];
    float measureStd, measureOpt;

    // use of measureStd/Opt present weighting of 3D residual by influence on reprojection error

    if (IntersectRaysInReferenceSpace(cam2, v0.homogeneous(), v1.homogeneous(), loc3dStd)) {
      ++nTriangulatedStd;

      const Vector3T p0 = v0.homogeneous();
      const Vector3T p1 = v1.homogeneous();
      measureStd = VectorsParallelMeasure(p0, common::CalculateRotationFromSVD(cam2.matrix()) * p1);
      distStd += (loc3dStd - trueLoc).norm() * measureStd;
    }

    TriangulationState ts;

    if (OptimalTriangulation(cam2, v0, v1, loc3dOpt, measureOpt, ts)) {
      ++nTriangulatedOpt;
      distOpt += (loc3dOpt - trueLoc).norm() * measureOpt;
    }
  }

  EXPECT_GT(nTriangulatedStd, nPoints / 4);
  EXPECT_GT(nTriangulatedOpt, nPoints / 3);
  distStd /= nTriangulatedStd;
  distOpt /= nTriangulatedOpt;

  EXPECT_LT(distOpt * p.ratioErrorStdVsOpt, distStd);
}

TEST_P(TriangulationTest, DISABLED_OptimaHomographyVsStandart) {
  const Vector3T& points3DMinRange = Vector3T(3, 2, 5);
  const Vector3T& points3DMaxRange = Vector3T(13, 10, 15.99f);

  const TestParam p(0.0f, 1.0f);
  const Isometry3T cam1 = Isometry3T::Identity();
  Isometry3T cam2;

  if (!TryToGenerateCamera2Matrix(500, points3DMinRange, points3DMaxRange, cam2)) {
    EXPECT_TRUE(false);  // i.e. failed to generate camMatrix
  }

  // generate points
  Vector2TVectorVector points2D;  // per cam
  Vector3TVector points3D;
  generateVisiblePoints({cam1, cam2}, 100, 50, points3DMinRange, points3DMaxRange, points2D, &points3D);
  const size_t nPoints = points3D.size();
  assert(points2D.size() == 2 && points2D[0].size() == nPoints && points2D[1].size() == nPoints);

  Vector2TPairVector points2DPairs;

  for (size_t i = 0; i < nPoints; ++i) {
    std::pair<Vector2T, Vector2T> pp(points2D[0][i], points2D[1][i]);
    points2DPairs.push_back(pp);
  }

  HomographyRansac homographyRansac;
  homographyRansac.setThreshold(0.001f);

  Matrix3T homography;
  homographyRansac(homography, points2DPairs.begin(), points2DPairs.end());

  size_t numInliers = homographyRansac.countInliers(homography, points2DPairs.cbegin(), points2DPairs.cend());
  EXPECT_TRUE(numInliers > nPoints / 2);

  // add noise
  for (size_t i = 0; i < nPoints; ++i) {
    points2D[0][i] += Vector2T::Random() * p.noiseScale;
    points2D[1][i] += Vector2T::Random() * p.noiseScale;
  }

  // run test
  Vector3T loc3dOpt, loc3dStd;
  float distOpt = 0;
  float distStd = 0;
  size_t nTriangulatedStd = 0;
  size_t nTriangulatedOpt = 0;

  OptimalPlanarTriangulation opt;
  const Isometry3T relativeTransform = cam1.inverse() * cam2;

  Vector3TVector optTriangulated;
  Vector3TVector standardTriangulated;

  for (size_t i = 0; i < nPoints; ++i) {
    const Vector2T& v0 = points2D[0][i];
    const Vector2T& v1 = points2D[1][i];
    const Vector3T& trueLoc = points3D[i];
    float measureStd, measureOpt;

    // use of measureStd/Opt present weighting of 3D residual by influence on reprojection error

    if (IntersectRaysInReferenceSpace(relativeTransform, v0.homogeneous(), v1.homogeneous(), loc3dStd)) {
      ++nTriangulatedStd;

      const Vector3T p0 = v0.homogeneous();
      const Vector3T p1 = v1.homogeneous();
      measureStd = VectorsParallelMeasure(p0, common::CalculateRotationFromSVD(cam2.matrix()) * p1);
      distStd += (loc3dStd - trueLoc).norm() * measureStd;

      standardTriangulated.push_back(loc3dStd);
    }

    TriangulationState ts = TriangulationState::None;

    if (opt.triangulate(homography, relativeTransform, v0, v1, loc3dOpt, measureOpt, ts)) {
      ++nTriangulatedOpt;
      distOpt += (loc3dOpt - trueLoc).norm() * measureOpt;

      optTriangulated.push_back(loc3dOpt);
    }
  }

  EXPECT_GT(nTriangulatedStd, nPoints / 2);

  if (nTriangulatedStd >= 3) {
    MatrixXT svdStandardTriangulated(standardTriangulated.size(), 3);
    size_t ptsCntr = 0;
    for (const auto& pt3D : standardTriangulated) {
      svdStandardTriangulated(ptsCntr, 0) = pt3D.x() - standardTriangulated[0].x();
      svdStandardTriangulated(ptsCntr, 1) = pt3D.y() - standardTriangulated[0].y();
      svdStandardTriangulated(ptsCntr, 2) = pt3D.z() - standardTriangulated[0].z();
      ptsCntr++;
    }

    // if we place all triangulated points on a  plane then 3rd singulat value should be a small one but with standard
    // triangulation it is not guaranteed
    IterativeSolver<Eigen::JacobiSVD<MatrixXT>> solverStandard(svdStandardTriangulated,
                                                               Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto sStandard = solverStandard.singularValues();
    TraceMessage("Standard Triangulated singulars: %f, %f, %f\n", sStandard[0], sStandard[1], sStandard[2]);
  }

  EXPECT_GT(nTriangulatedOpt, nPoints / 2);

  if (nTriangulatedOpt >= 3) {
    MatrixXT svdOptTriangulated(optTriangulated.size(), 3);
    size_t ptsCntr = 0;
    for (const auto& pt3D : optTriangulated) {
      svdOptTriangulated(ptsCntr, 0) = pt3D.x() - optTriangulated[0].x();
      svdOptTriangulated(ptsCntr, 1) = pt3D.y() - optTriangulated[0].y();
      svdOptTriangulated(ptsCntr, 2) = pt3D.z() - optTriangulated[0].z();
      ptsCntr++;
    }

    // if we place all triangulated points on a  plane then 3rd singulat value should be a small one
    IterativeSolver<Eigen::JacobiSVD<MatrixXT>> solverOpt(svdOptTriangulated,
                                                          Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto sOpt = solverOpt.singularValues();
    TraceMessage("Opt Triangulated singulars: %f, %f, %f\n", sOpt[0], sOpt[1], sOpt[2]);
  }

  distStd /= nTriangulatedStd;
  distOpt /= nTriangulatedOpt;

  EXPECT_LT(distOpt * p.ratioErrorStdVsOpt, distStd);
}

INSTANTIATE_TEST_SUITE_P(Epipolar, TriangulationTest,
                         Values(
                             /*0*/ TestParam(0.0f, 1.f),
                             /*1*/ TestParam(0.1f, 1.f)));

}  // namespace test::epipolar
