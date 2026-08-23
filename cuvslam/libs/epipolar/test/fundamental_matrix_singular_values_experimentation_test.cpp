
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

#include "epipolar/test/fundamental_matrix_utils_test.h"

// #define PRINT_OUT_DATA

namespace test::epipolar {

std::vector<std::pair<Vector2T, Vector2T>> GeneratePairs(const Vector2TVector& points1, const Vector2TVector& points2) {
  const size_t size = points1.size();
  std::vector<std::pair<Vector2T, Vector2T>> pairs(size);

  for (size_t i = 0; size > i; ++i) {
    pairs[i].first = points1[i];
    pairs[i].second = points2[i];
  }

  return pairs;
}

template <typename Type>
std::vector<Type> GenerateVectorInRange(Type min, Type max, size_t numSamples) {
  Type increment = std::abs(max - min) / static_cast<Type>(numSamples);
  Type current = min;
  std::vector<Type> samples(numSamples);
  std::generate(samples.begin(), samples.end(), [&increment, &current]() { return current += increment; });
  return samples;
}

/// @brief Class logging the singular values and the frobenius norm between expected and
/// reconstruction cameras.
class FundamentalMatrixValuesLogger : public FundamentalMatrixUtilsTest {
protected:
  FundamentalMatrixValuesLogger() : FundamentalMatrixUtilsTest() {
    ss << std::endl
       << "frobNorm,singularV1,eigenV2,eigenV3,eigenV4,eigenV5,eigenV6,"
       << "eigenV7,eigenV8,eigenV9" << std::endl;
  }

  void LogData(float frobeniusNorm, const Vector9T& vector) {
    ss << frobeniusNorm << ",";
    ss << vector(0) << ",";
    ss << vector(1) << ",";
    ss << vector(2) << ",";
    ss << vector(3) << ",";
    ss << vector(4) << ",";
    ss << vector(5) << ",";
    ss << vector(6) << ",";
    ss << vector(7) << ",";
    ss << vector(8) << std::endl;
  }

  void PrintOutData() {
#ifdef PRINT_OUT_DATA
    std::cout << ss.str() << std::endl;
#endif
  }

  template <typename Type>
  bool RunTest(float min, float max, size_t numSamples) {
    using LocalComputeFundamental = ComputeFundamentalT<Vector2TVector, Type>;

    std::vector<Type> samples = GenerateVectorInRange(static_cast<Type>(min), static_cast<Type>(max), numSamples);

    bool res(true);

    for (auto planeTickness : samples) {
      res &= CreateSetup(planeTickness);

      m_relativeTransform = (m_camera2.inverse()) * m_camera1;

      SetupPoints();

      LocalComputeFundamental computeFundamental(m_points2DLocal1, m_points2DLocal2);
      Matrix3T essential;
      res &= LocalComputeFundamental::ReturnCode::Success == computeFundamental.findFundamental(essential);

      // Decompose the essential matrix
      Isometry3T rotation1, rotation2;
      Vector3T translation;
      res &= ExtractRotationTranslationFromEssential(essential, rotation1, rotation2, translation);

      std::vector<cuvslam::EulerRotation3T> rotations = {rotation1.getEulerRotation(), rotation2.getEulerRotation()};
      std::vector<cuvslam::Translation3T> translations = {Translation3T(translation), Translation3T(-translation)};

      // Find the optimal combination of rotation/translation and reconstruct the transform from
      // camera1 to camera2.
      Isometry3T actualTransform;
      auto pairs = GeneratePairs(m_points2DLocal1, m_points2DLocal2);
      res &= FindOptimalCameraMatrixExhaustiveSearch(pairs.cbegin(), pairs.cend(), rotations, translations,
                                                     actualTransform);

      // Normalize translations
      m_relativeTransform.translation().normalize();
      actualTransform.translation().normalize();
      // Compute the Frobenius norm between the original and the reconstructed transforms
      float frobeniusNorm =
          CalculateDistanceFromExpectedRotationMatrix(m_relativeTransform.matrix(), actualTransform.matrix());

      Vector9T matATASingularValues;
      res &= CheckSuccess(computeFundamental.getMatATASingularValues(matATASingularValues));

      LogData(frobeniusNorm, matATASingularValues);
    }

    return res;
  }

  virtual bool CreateSetup(double variable) = 0;

private:
  std::stringstream ss;
};

class FundamentalMatrixValuesPlanarLogger : public FundamentalMatrixValuesLogger {
protected:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Create the setup for the planar homorgraphy evaluation. The setup chooses camera1 and
  /// camera2 at random, looking at the 3D points.
  ///
  /// @param sliceThickness The thickness of the slice in which 3D points are generated.
  ///-------------------------------------------------------------------------------------------------
  bool CreateSetup(double sliceThickness) override {
    bool res(true);
    Vector3T camera1Position = Vector3T::Random();
    Vector3T camera2Position = Vector3T::Random();
    m_points3DMinRange = Vector3T(-5.0f, -5.0f, -10.0f);
    m_points3DMaxRange = Vector3T(5.0f, 5.0f, -10.0f + sliceThickness);
    const Vector3T pointCloudCenter = (m_points3DMinRange + m_points3DMaxRange) / 2.0f;

    res &= CreateCameraMatrix(camera1Position, pointCloudCenter, Vector3T(0.0f, 1.0f, 1.0f), m_camera1);

    res &= CreateCameraMatrix(camera2Position, pointCloudCenter, Vector3T(0.0f, 1.0f, 1.0f), m_camera2);
    return res;
  }
};

///-------------------------------------------------------------------------------------------------
/// @brief We evaluate here the singular values for general camera motion with a planar scene.
/// The thickness of the scene plane ("slice") varies. Test for double precision.
///-------------------------------------------------------------------------------------------------
TEST_F(FundamentalMatrixValuesPlanarLogger, SingularValuesForDouble) {
  // This test runs with values that fail to decompose [R|t]. We cannot EXPECT_TRUE here.
  RunTest<double>(1.0e-7, 1.0e-3, 1000);
  PrintOutData();
}

///-------------------------------------------------------------------------------------------------
/// @brief We evaluate here the singular values for general camera motion with a planar scene.
/// The thickness of the scene plane ("slice") varies. Test for float precision.
///-------------------------------------------------------------------------------------------------
TEST_F(FundamentalMatrixValuesPlanarLogger, SingularValuesForFloat) {
  // This test runs with values that fail to decompose [R|t]. We cannot EXPECT_TRUE here.
  RunTest<float>(1.0e-3, 1.0e-0, 1000);
  PrintOutData();
}

class FundamentalMatrixValuesRotationLogger : public FundamentalMatrixValuesLogger {
protected:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Create the setup for the rotation evaluation. The setup chooses camera1 position at
  /// random, looking at the 3D points. Camera2 is defined as camera1 rotated in the x and y axes
  /// and with a translation as the evaluating variable.
  ///
  /// @param translation The base between the 2 cameras.
  ///-------------------------------------------------------------------------------------------------
  bool CreateSetup(double translation) override {
    bool res(true);

    Vector3T camera1Position = Vector3T::Random();
    const Vector3T pointCloudCenter = (m_points3DMinRange + m_points3DMaxRange) / 2.0f;

    res &= CreateCameraMatrix(camera1Position, pointCloudCenter, Vector3T(0.0f, 1.0f, 1.0f), m_camera1);

    m_camera2 = EulerRotation3T(Vector3T(30, 45, 0), AngleUnits::Degree).getQuaternion() * m_camera1;
    m_camera2.translation() = m_camera1.translation() + Vector3T(translation, 0.0f, 0.0f);
    return res;
  }
};

///-------------------------------------------------------------------------------------------------
/// @brief We evaluate here the singular values for the pure rotation case (with very small
/// translation). The range to generate the result is the same for double and float precision.
///-------------------------------------------------------------------------------------------------
TEST_F(FundamentalMatrixValuesRotationLogger, SingularValuesForDoubleAndFloat) {
  // This test runs with values that fail to decompose [R|t]. We cannot EXPECT_TRUE here.
  RunTest<double>(1.0e-2, 1.0e-1, 1000);
  PrintOutData();
}

}  // namespace test::epipolar
