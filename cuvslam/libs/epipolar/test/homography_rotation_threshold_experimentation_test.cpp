
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

#include "registration/test/homography_test.h"

#include "epipolar/fundamental_matrix_utils.h"

namespace test::epipolar {
using namespace cuvslam;

// #define PRINT_OUT_DATA

template <typename Type>
std::vector<Type> GenerateVectorInRange(Type min, Type max, size_t numSamples) {
  Type increment = std::abs(max - min) / static_cast<Type>(numSamples);
  Type current = min;
  std::vector<Type> samples(numSamples);
  std::generate(samples.begin(), samples.end(), [&increment, &current]() { return current += increment; });
  return samples;
}

class HomographyRotationExperimentationTest : public HomographyTest {
protected:
  HomographyRotationExperimentationTest() : HomographyTest() {
    ss << std::endl << "translation,residual,eigen7" << std::endl;
  }
  void logData(float translation, float residual, float eigen7) {
    ss << translation << "," << residual;
    ss << "," << eigen7 << std::endl;
  }

  void printOutData() {
#ifdef PRINT_OUT_DATA
    std::cout << ss.str() << std::endl;
#endif
  }

  virtual bool SetupPointsProjectionsFromCameras(const Isometry3T& camera1, const Isometry3T& camera2,
                                                 const Vector3TVector& points3D) {
    // Project points on the 2 camera planes
    return Project3DPointsInLocalCoordinates(camera1.inverse(), points3D, &m_expectedPoints1) &&
           Project3DPointsInLocalCoordinates(camera2.inverse(), points3D, &m_expectedPoints2);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief This test will place 2 cameras related by a varying translation (with random
  /// rotations). Varying the translation provides us a means to define a threshold for which the
  /// homography is deemed not to be a pure rotation only. The error in rotation is represented as
  /// the residual (Frobenius distance) of the off-diagonal coeficients of the scaled HT*H.
  /// @see GetResidualForHomographyAsARotationMatrix for more details on the calculation.
  ///-------------------------------------------------------------------------------------------------
  bool runTest(float min, float max, size_t numSamples) {
    // Generate 3D points - we want the points to be fixed for the entire test.
    const Vector3TVector points3D = GeneratePointsInCube(m_numPoints, m_minRange, m_maxRange);

    // Generate the translations for camera2 w.r.t camera1.
    const std::vector<float> translations = GenerateVectorInRange(min, max, numSamples);

    const Vector3T cameraInitialPosition = Vector3T::Random();

    // Create 2 cameras with random rotations at the initial position
    const Isometry3T expectedCamera1 = CreateIsometryTransform(Vector3T::Random(), cameraInitialPosition);
    Isometry3T expectedCamera2 = CreateIsometryTransform(Vector3T::Random(), cameraInitialPosition);

    // Find a direction orthogonal to the line passing through the camaer initial position and the
    // center of the 3D points sampling cube.
    const Vector3T directionCameraToCenter = (m_minRange + m_maxRange) / 2.0f - cameraInitialPosition;
    Vector3T transDirection = Vector3T(0.0f, 1.0f, 0.0).cross(directionCameraToCenter);
    {
      SCOPED_TRACE("Orthogonal direction could not be found");

      if (transDirection.norm() < 1.0e-5)  // Make sure that our vectors are not collinear
      {
        return false;
      }
    }
    transDirection.normalize();

    const float distanceCameraToCenter = directionCameraToCenter.norm();

    // Move the 2 cameras apart by a translation (as a ratio of the distance to the 3D points.
    for (auto translation : translations) {
      expectedCamera2.translation() = cameraInitialPosition + translation * distanceCameraToCenter * transDirection;
      {
        SCOPED_TRACE("Homography could not be set from cameras");

        if (!SetupPointsProjectionsFromCameras(expectedCamera1, expectedCamera2, points3D)) {
          return false;
        }
      }

      ComputeHomography computeHomography(m_expectedPoints1, m_expectedPoints2);
      ComputeFundamental computeFundamental(m_expectedPoints1, m_expectedPoints2);

      Matrix3T actualHomography;
      computeHomography.findHomography(actualHomography);

      Vector9T singularValues;

      if (!(ComputeFundamental::ReturnCode::Success == computeFundamental.getMatATASingularValues(singularValues) &&
            ComputeHomography::ReturnCode::Success == computeHomography.findHomography(actualHomography))) {
        return false;
      }

      float residual = GetResidualForHomographyAsARotationMatrix(actualHomography);
      logData(translation, residual, singularValues[6]);
    }

    return true;
  }

protected:
  std::stringstream ss;
};

TEST_F(HomographyRotationExperimentationTest, HomographyConsideredAsARotationMatrixThreshold) {
  // Reduce the sampling cube for the 3D points so that we do not have too much difference
  // between the actual and the expected center.
  m_minRange = Vector3T(-5.0f, -5.0f, -10.0f);
  m_maxRange = Vector3T(5.0f, 5.0f, -9.5f);

  ASSERT_TRUE(runTest(0.0f, 0.1f, 20));
  printOutData();
}

}  // namespace test::epipolar
