
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

#include "common/include_gtest.h"
#include "common/rotation_utils.h"
#include "epipolar/camera_projection.h"
#include "epipolar/fundamental_matrix_utils.h"
#include "epipolar/point_reconstruction.h"
#include "epipolar/resectioning_utils.h"
#include "epipolar/resectioning_utils_internal.h"
#include "epipolar/test/generate_points_in_cube_test.h"
#include "math/twist.h"

#define USE_OPENCV_RESECTIONING 0

#if USE_OPENCV_RESECTIONING
#include "epipolar/test/resectioning_utils_test.h"
#endif

namespace {
using namespace cuvslam;

///-------------------------------------------------------------------------------------------------
/// @brief Generates points on a spiral pattern. The spiral's axis is the z-axis.
///
/// @param numPoints       Number of points to generate.
/// @param tangentialSpeed The speed at which we travel along the tangent (e.g.
/// tangentialSpeed=2*PI/N means that the method will generate N points per revolution.
/// @param radialSpeed     The speed at which the radius is increased (e.g. if point (N-1) is at
/// distance r from the z-axis, point N will be at (r + radialSpeed) from the z- axis).
/// @param verticalSpeed   The speed at which the z-coordinate of a point moves (e.g. if point
/// (N-1) has z = a, point N has z = (a + vertialSpeed).
/// @param initialRadius   The initial radius, i.e. r(0).
///
/// @return The points generated points in a spiral pattern.
///-------------------------------------------------------------------------------------------------
Vector3TVector GeneratePointsOnSpiral(size_t numPoints, float tangentialSpeed, float radialSpeed, float verticalSpeed,
                                      float initialRadius) {
  Vector3TVector points(numPoints);
  float radius = initialRadius;
  float angle = 0.0f;
  float height = 0.0f;

  for (auto& point : points) {
    point.x() = radius * std::cos(angle);
    point.y() = radius * std::sin(angle);
    point.z() = height;
    radius += radialSpeed;
    angle += tangentialSpeed;
    height += verticalSpeed;
  }

  return points;
}

///-------------------------------------------------------------------------------------------------
/// @brief Generates spiral of cameras with all cameras pointing towards the lookingAt point.
///
/// @param lookingAt        The point all cameras are looking at, i.e. all cameras z-axis is
/// pointing away from.
/// @param numCameras       Number of cameras.
/// @param tangentialSpeed  The tangential speed - (@see GeneratePointsOnSpiral).
/// @param radialSpeed      The radial speed - (@see GeneratePointsOnSpiral).
/// @param verticalSpeed    The vertical speed - (@see GeneratePointsOnSpiral).
/// @param initialRadius    The initial radius - (@see GeneratePointsOnSpiral).
/// @param upSweepSpeed     The speed at which the up vector sweeps. Refer to actual
/// implementation for more description. In summary, the parameter is used to make the camera
/// rotated around its z-axis.
/// @param [in,out] cameras The cameras in a spiral pattern.
///
/// @return false if the method could not generate a camera, true otherwise. If false, a
/// mechanism to re-try with a different "up" vector should be implemented.
///-------------------------------------------------------------------------------------------------
bool GenerateCameraSpiral(const Vector3T& lookingAt, size_t numCameras, float tangentialSpeed, float radialSpeed,
                          float verticalSpeed, float initialRadius, Isometry3TVector& cameras) {
  cameras.resize(numCameras);

  Vector3TVector cameraPositions =
      GeneratePointsOnSpiral(numCameras, tangentialSpeed, radialSpeed, verticalSpeed, initialRadius);

  auto positionsIt = cameraPositions.cbegin();
  const auto positionsItEnd = cameraPositions.cend();
  auto camerasIt = cameras.begin();

  for (float upAngle = 0.0f; positionsItEnd != positionsIt; ++positionsIt, ++camerasIt) {
    if (!epipolar::CreateCameraMatrix(*positionsIt, lookingAt,
                                      Vector3T(0.0f, std::abs(std::cos(upAngle)), std::abs(std::sin(upAngle))),
                                      *camerasIt)) {
      return false;
    }
  }

  return true;
}

Vector2TPairVector GenerateVectorOfPairsFromPairsOfVectors(const Vector2TVector& vector1,
                                                           const Vector2TVector& vector2) {
  const size_t size = vector1.size();
  assert(size == vector2.size());
  Vector2TPairVector pairs(size);

  for (size_t idx = 0; size > idx; ++idx) {
    pairs[idx].first = vector1[idx];
    pairs[idx].second = vector2[idx];
  }

  return pairs;
}

}  // namespace

namespace test::epipolar {
using namespace cuvslam;
using namespace cuvslam::epipolar;

class ResectioningIntegTestGroundTruth3D : public testing::Test {
public:
  void SetUp() override {
    // Generate the camera choreography
    ASSERT_TRUE(GenerateCameraChoreography(expectedCameras_));

    bool is3DPointReconstructed = false;

    for (int i = 0; i < 3; i++) {
      // Generate the ground truth 3D points
      expected3DPoints_ = GeneratePointsInCube(num3DPoints_, minRange_, maxRange_);

      // Generate the 2D points for each camera for a normalized camera
      Generate2DPointsForAllCameras(expected3DPoints_, expectedCameras_, points2D_);

      // Reconstruct the 3D points
      is3DPointReconstructed = Reconstruct3DPoints(actual3DPoints_);

      if (is3DPointReconstructed) {
        break;
      }
    }

    ASSERT_TRUE(is3DPointReconstructed);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Generates a camera choreography. One should overridethis method if one wants to test
  /// for a different choreography.
  ///-------------------------------------------------------------------------------------------------
  virtual bool GenerateCameraChoreography(Isometry3TVector& cameras) const {
    return GenerateCameraSpiral((minRange_ + maxRange_) / 2.0f, numCameras_, 2.0f * float(PI) / 19.7f, 0.04f, 0.07f,
                                30.0f, cameras);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Generates a 2D points for all cameras, i.e. for the entire sequence
  ///
  /// @param points3D                    The vector of 3D points that will be projected onto each camera.
  /// @param cameras                     The vector vector of cameras representing the sequence.
  /// @param [in,out] points2DAllCameras The vector of vector of 2D points.
  ///-------------------------------------------------------------------------------------------------
  void Generate2DPointsForAllCameras(const Vector3TVector& points3D, const Isometry3TVector& cameras,
                                     Vector2TVectorVector& points2DAllCameras) {
    points2DAllCameras.resize(cameras.size());
    auto cameraIt = cameras.cbegin();
    const auto cameraItEnd = cameras.cend();
    auto points2DCamerasIt = points2DAllCameras.begin();

    for (; cameraItEnd != cameraIt; ++cameraIt, ++points2DCamerasIt) {
      Project3DPointsInLocalCoordinates(cameraIt->inverse(), points3D, *points2DCamerasIt);
    }
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Solve the sequence of cameras, i.e. the choreography. This method optimizes the camera
  /// parameters, sequentially, from a camera starting point, 3D points and 2D points.
  ///
  /// @param startingPoint The camera starting point or initial guess.
  ///
  /// @return false if the method failed to optimize for one (or more) camera(s).
  ///-------------------------------------------------------------------------------------------------
  bool SolveSequence(const Isometry3T& startingPoint) {
    actualCameras_.clear();
    Isometry3T actualCamera = startingPoint;

    auto points2DIt = points2D_.cbegin();
    const auto points2DItEnd = points2D_.cend();
    bool res(true);

    for (; points2DItEnd != points2DIt; ++points2DIt) {
#if USE_OPENCV_RESECTIONING
      res &= OptimizeCameraExtrinsics(actual3DPoints_.cbegin(), actual3DPoints_.cend(), points2DIt->cbegin(),
                                      points2DIt->cend(), actualCamera);
#else
      res &= OptimizeCameraExtrinsicsExpMap<6>(actual3DPoints_.cbegin(), actual3DPoints_.cend(), points2DIt->cbegin(),
                                               points2DIt->cend(), actualCamera);
#endif
      actualCameras_.push_back(actualCamera);
    }

    return res;
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Reconstruct the 3D points. The base implementation does not reconstruct the 3D points,
  /// it simply return a reference to the ground truth. One should override this implementation for
  /// a different behavior (e.g. reconstruct 3D points from epipolar geometry).
  ///
  /// @param [in,out] actual3DPoints The vector of reconstructed 3D points.
  ///
  /// @return true.
  ///-------------------------------------------------------------------------------------------------
  virtual bool Reconstruct3DPoints(Vector3TVector& actual3DPoints) const {
    actual3DPoints = expected3DPoints_;
    return true;
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Evaluate that the sequence has converged to the ground truth in terms of Frobenius
  /// distance for each estimated camera.
  ///
  /// @param errorThreshold The error threshold determining whether the camera is deemed to have
  /// converged or not.
  ///-------------------------------------------------------------------------------------------------
  void EvaluateConvergence(float errorThreshold) {
    assert(expectedCameras_.size() == actualCameras_.size());

    auto expectedCameraIt = expectedCameras_.cbegin();
    const auto expectedCameraItEnd = expectedCameras_.cend();
    auto actualCameraIt = actualCameras_.cbegin();

    for (int idx = 0; expectedCameraItEnd != expectedCameraIt; ++expectedCameraIt, ++actualCameraIt, ++idx) {
      std::stringstream ss;

      ss << "Unexpected result occurred while evaluating camera number " << idx;
      EXPECT_LE(CalculateMatricesFrobeniusDistance(expectedCameraIt->matrix(), actualCameraIt->matrix()),
                errorThreshold)
          << ss.str();
    }
  }

protected:
  const size_t num3DPoints_ = 400;
  const size_t numCameras_ = 200;
  const Vector3T minRange_ = Vector3T(-5.0f, -5.0f, 2.0f);
  const Vector3T maxRange_ = Vector3T(5.0f, 5.0f, 15.0f);
  Vector2TVectorVector points2D_;
  Isometry3TVector expectedCameras_;
  Isometry3TVector actualCameras_;
  Vector3TVector expected3DPoints_;
  Vector3TVector actual3DPoints_;
};

TEST_F(ResectioningIntegTestGroundTruth3D, DISABLED_EvaluateConvergenceOnChoreography) {
  const float errorThreshold = 5.e-3f;
  SolveSequence(expectedCameras_[0]);
  EvaluateConvergence(errorThreshold);
}

class ResectioningIntegTestReconstructed3D : public ResectioningIntegTestGroundTruth3D {
public:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Reconstruct the 3D points from 2 frames containing matching 2D points and epipolar
  /// geometry. In the case of calibrated cameras, a similarity reconstruction is obtained. In
  /// order to be able to have exact reconstruction, we assume (only) (i) that the first camera
  /// rotation and translations are known and (ii) that the first 3D point of the ground truth is
  /// known in order to recover scale.
  ///
  /// @param [in,out] actual3Dpoints The 3D points reconstructed.
  ///
  /// @return true if it succeeds, false if it fails.
  ///-------------------------------------------------------------------------------------------------
  virtual bool Reconstruct3DPoints(Vector3TVector& actual3Dpoints) const override {
    const Vector2TVector& frame1 = points2D_[firstFrame_];
    const Vector2TVector& frame2 = points2D_[secondFrame_];

    // Find the essential matrix
    Matrix3T essential;
    ComputeFundamental computeEpipolar(frame1, frame2);

    if (ComputeFundamental::ReturnCode::Success != computeEpipolar.findFundamental(essential)) {
      return false;
    }

    // Do some data formatting
    Vector2TPairVector pairs = GenerateVectorOfPairsFromPairsOfVectors(frame1, frame2);

    Isometry3T relativeTransform;

    if (!FindOptimalCameraMatrixFromEssential(pairs.cbegin(), pairs.cend(), essential, relativeTransform)) {
      return false;
    }

    // line below added to force triangulated 3D points beyond hither of -1.0f (i.e. < -1)
    relativeTransform.translation() *= 4.0f;

    // Reconstruct the actual points from epipolar geometry.
    if (!Reconstruct3DPointsFrom2DPointsAndRelativeTransform(pairs, relativeTransform, actual3Dpoints)) {
      return false;
    }

    // Find the transform scale - the essential matrix provides a similarity reconstruction. Aside
    // from rotation and translation (given by the camera matrix for frame 1), the scale needs to
    // be determined. This is not possible without prior knowledge of the scene. We therefore
    // assume here that the first 3D point of the ground truth is known and we calculate the scale of our
    // reconstruction from it.
    Isometry3T aboluteTransformFirstFrame = expectedCameras_[firstFrame_];
    Vector3T point3InLocalSpace = aboluteTransformFirstFrame.inverse() * expected3DPoints_[0];
    const float scale = point3InLocalSpace.norm() / (actual3Dpoints[0].norm());

    // Apply the transform to the reconstructed points in order to place them in world coordinates.
    for (auto& point3D : actual3Dpoints) {
      point3D = aboluteTransformFirstFrame * Eigen::Scaling(scale) * point3D;
    }

    return true;
  }

protected:
  const size_t firstFrame_ = 20;    // First frame for epipolar geometry
  const size_t secondFrame_ = 190;  // Second frame for epipolar geometry
};

TEST_F(ResectioningIntegTestReconstructed3D, DISABLED_EvaluateConvergenceOnChoreography) {
  const float errorThreshold = 5.e-3f;
  SolveSequence(expectedCameras_[0]);
  EvaluateConvergence(errorThreshold);
}

class ExpMapIntegTest : public ResectioningIntegTestReconstructed3D {
public:
  inline Vector3T LogRotationPGO(const Matrix3T& rot) {
    float tr = rot.trace();

    // special cases (when tr_3 >= -epsilon()) I did not check logic

    // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    // we do something special
    if (std::abs(tr + 1.0f) < epsilon()) {
      if (std::abs(rot(2, 2) + 1.0f) > epsilon())
        return ((float(PI) / std::sqrt(2.0f + 2.0f * rot(2, 2))) * Vector3T(rot(0, 2), rot(1, 2), 1.0f + rot(2, 2)));
      else if (std::abs(rot(1, 1) + 1.0f) > epsilon())
        return ((float(PI) / std::sqrt(2.0f + 2.0f * rot(1, 1))) * Vector3T(rot(0, 1), 1.0f + rot(1, 1), rot(2, 1)));
      else  // if(std::abs(R.r1_.x()+1.0f) > epsilon())  This is implicit
        return ((float(PI) / std::sqrt(2.0f + 2.0f * rot(0, 0))) * Vector3T(1.0f + rot(0, 0), rot(1, 0), rot(2, 0)));
    } else {
      float magnitude;
      float tr_3 = tr - 3.0f;  // always negative

      if (tr_3 < -epsilon()) {
        float theta = std::acos((tr - 1.0f) / 2.0f);
        magnitude = theta / (2.0f * std::sin(theta));
      } else {
        // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0f)
        // use Taylor expansion: magnitude \approx 1/2-(t-3)/12 + O((t-3)^2)
        magnitude = 0.5f - tr_3 * tr_3 / 12.0f;
      }

      return magnitude * Vector3T(rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1));
    }
  }
};

TEST_F(ExpMapIntegTest, DISABLED_EvaluateExpMapValidity) {
  const float corrThresh = 0.99f;
  const float percentCorrError = epsilon();
  std::vector<float> testCorr;

  for (auto expectedCameraIt = expectedCameras_.cbegin(); expectedCameraIt != expectedCameras_.cend();
       ++expectedCameraIt) {
    Isometry3T invCamMat = (*expectedCameraIt).inverse();

    Vector2TVector projectedPoints;
    Project3DPointsInLocalCoordinates(invCamMat, expected3DPoints_.cbegin(), expected3DPoints_.cend(), projectedPoints);

    SurveyedTracking<6> derivsAnalytical(expected3DPoints_.cbegin(), expected3DPoints_.cend(), projectedPoints.cbegin(),
                                         projectedPoints.cend(), DEFAULT_HUBER_DELTA);
    SurveyedTracking<6> derivsNumericalBase(expected3DPoints_.cbegin(), expected3DPoints_.cend(),
                                            projectedPoints.cbegin(), projectedPoints.cend(), DEFAULT_HUBER_DELTA);
    Eigen::NumericalDiff<SurveyedTracking<6>> derivsNumerical(derivsNumericalBase);

    SurveyedTracking<6>::InputType twist(invCamMat);
    EXPECT_TRUE((LogRotationPGO(common::CalculateRotationFromSVD(invCamMat.matrix())) - twist.head(3)).squaredNorm() <
                10 * epsilon());

    const Isometry3T testIsometry = twist.transform();
    EXPECT_TRUE(testIsometry.isApprox(invCamMat, sqrt_epsilon()));

    size_t goodCorr = 0;
    size_t badCorr = 0;

    SurveyedTracking<6>::JacobianType JacobiAnalytical(projectedPoints.size() * 2, 6);
    SurveyedTracking<6>::JacobianType JacobiNumerical(projectedPoints.size() * 2, 6);
    derivsAnalytical.df(twist, JacobiAnalytical);
    derivsNumerical.df(twist, JacobiNumerical);

    for (int i = 0; i < JacobiNumerical.rows(); i += 2) {
      Vector<float, 12> da;
      da << JacobiAnalytical.row(i).transpose(), JacobiAnalytical.row(i + 1).transpose();

      Vector<float, 12> dn;
      dn << JacobiNumerical.row(i).transpose(), JacobiNumerical.row(i + 1).transpose();
      float corrMe = (da.dot(dn)) / (da.norm() * dn.norm());

      if (corrMe >= corrThresh) {
        goodCorr++;
      } else {
        badCorr++;
      }

      testCorr.push_back(corrMe);
      EXPECT_GE(corrMe, corrThresh);
    }

    EXPECT_TRUE(goodCorr > 0 && float(badCorr) / float(goodCorr + badCorr) < percentCorrError);
  }

  std::sort(testCorr.begin(), testCorr.end());
  float se = testCorr.front();
  EXPECT_GE(se, corrThresh);
}

}  // namespace test::epipolar
