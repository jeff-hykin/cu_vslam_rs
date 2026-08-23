
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

#include <algorithm>
#include <numeric>
#include <random>

#include "epipolar/homography_ransac.h"
#include "epipolar/test/homography_test.h"

namespace test::epipolar {

class HomographyRansacTest : public HomographyTest {
protected:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Find a homography which Frobenius distance is larger than a given threshold. The
  /// Frobenius distance is measured on the scaled and normalized homographies.
  /// @return fails if it reached the maximum number of trials without find a suitable homography.
  ///-------------------------------------------------------------------------------------------------
  bool FindDifferentHomography(const Matrix3T& referenceHomography, Matrix3T& differentHomography,
                               float threshold) const {
    const size_t MAX_NUM_TRIAL = 50;

    Matrix3T referenceHomographyNormalized = referenceHomography;
    referenceHomographyNormalized /= referenceHomographyNormalized(2, 2);
    referenceHomographyNormalized.normalize();

    for (size_t trialIndex = 0; MAX_NUM_TRIAL > trialIndex; ++trialIndex) {
      differentHomography = referenceHomographyNormalized + Matrix3T::Random();
      differentHomography /= differentHomography(2, 2);
      differentHomography.normalize();
      float distance =
          cuvslam::epipolar::CalculateMatricesFrobeniusDistance(referenceHomographyNormalized, differentHomography);

      if (distance > threshold) {
        return true;
      }
    }

    return false;
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Corrupts the data in the second image by setting the point in the second image to the
  /// value it would have had it been transformed by a different homography.
  ///-------------------------------------------------------------------------------------------------
  bool CorruptSecondImageData(const Matrix3T& homography, Vector2TPairVector& points2DPair, size_t numCorrupted) const {
    const size_t numPoints = points2DPair.size();
    const float corruptedThreshold = 0.5f;

    if (!(numCorrupted < numPoints)) {
      return false;
    }

    // Generate a selection of points that we are going to corrupt. We need to ensure here that a
    // point is selected at most once (no repeat). This ensures that exactly numCorruptedpoints are
    // corrupted.
    std::vector<size_t> randomSelection(numPoints);
    std::iota(randomSelection.begin(), randomSelection.end(), 0);
    std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
    std::shuffle(randomSelection.begin(), randomSelection.end(), gen);
    randomSelection.resize(numCorrupted);

    for (size_t corruptedIndex : randomSelection) {
      Matrix3T corruptedHomography;
      const size_t MAX_NUM_TRIAL = 50;
      const size_t SENTINEL_SUCCESS = 100;
      static_assert(SENTINEL_SUCCESS > MAX_NUM_TRIAL, "The sentinel needs to be greater than maximum number of trials");
      size_t trialIndex = 0;

      // Try to find some proper corrupted data
      for (; MAX_NUM_TRIAL > trialIndex; ++trialIndex) {
        // Find a homography that is different from the expected one.
        if (!FindDifferentHomography(homography, corruptedHomography, corruptedThreshold)) {
          continue;
        }

        Vector3T corruptedImage2Point = corruptedHomography * points2DPair[corruptedIndex].first.homogeneous();
        float corruptedZ = corruptedImage2Point.z();

        // Fail if the different homography projects the point on the camera plane.
        if (std::abs(corruptedZ) < 1.0e-7f) {
          continue;
        }

        points2DPair[corruptedIndex].second =
            Vector2T(corruptedImage2Point.x() / corruptedZ, corruptedImage2Point.y() / corruptedZ);

        // Ensure that the residual of the corrupted data is above the Ransac threshold,
        // by a good margin.
        if (cuvslam::epipolar::ComputeHomographyResidual(points2DPair[corruptedIndex].first,
                                                         points2DPair[corruptedIndex].second,
                                                         homography) > 10.0f * threshold_) {
          trialIndex = SENTINEL_SUCCESS;
        }
      }

      if (MAX_NUM_TRIAL == trialIndex) {
        return false;
      }
    }

    return true;
  }

protected:
  float threshold_ = 0.0005f;
};

TEST_F(HomographyRansacTest, HomographyRansacWithoutNoise) {
  ASSERT_TRUE(SetupGeneralHomographyCase());

  Vector2TPairVector points2DPairs;
  ConvertVectorOfPairsTo2Vectors(expectedPoints1_, expectedPoints2_, points2DPairs);

  cuvslam::epipolar::HomographyRansac homographyRansac;
  homographyRansac.setThreshold(threshold_);

  Matrix3T actualHomography;
  homographyRansac(actualHomography, points2DPairs.begin(), points2DPairs.end());

  const float epsDistance = 7.0e-3f;
  EXPECT_TRUE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
}

TEST_F(HomographyRansacTest, HomographyRansacWithNoise) {
  ASSERT_TRUE(SetupGeneralHomographyCase());

  Vector2TPairVector points2DPairs;
  ConvertVectorOfPairsTo2Vectors(expectedPoints1_, expectedPoints2_, points2DPairs);

  const float ratioCorrupted = 0.2f;
  const size_t numCorrupted = static_cast<size_t>(ratioCorrupted * points2DPairs.size());
  ASSERT_TRUE(CorruptSecondImageData(expectedHomography_, points2DPairs, numCorrupted));

  cuvslam::epipolar::HomographyRansac homographyRansac;
  homographyRansac.setThreshold(threshold_);

  Matrix3T actualHomography;
  EXPECT_GT(homographyRansac(actualHomography, points2DPairs.begin(), points2DPairs.end()), (size_t)0);

  const float epsDistance = 1.1e-2f;
  EXPECT_TRUE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
}

TEST_F(HomographyRansacTest, HomographyRansacWithNoiseCountInliners) {
  ASSERT_TRUE(SetupGeneralHomographyCase());

  Vector2TPairVector points2DPairs;
  ConvertVectorOfPairsTo2Vectors(expectedPoints1_, expectedPoints2_, points2DPairs);

  const float ratioCorrupted = 0.125f;
  const size_t numCorrupted = static_cast<size_t>(ratioCorrupted * points2DPairs.size());
  ASSERT_TRUE(CorruptSecondImageData(expectedHomography_, points2DPairs, numCorrupted));

  cuvslam::epipolar::HomographyRansac homographyRansac;
  homographyRansac.setThreshold(threshold_);

  Matrix3T actualHomography;
  homographyRansac(actualHomography, points2DPairs.begin(), points2DPairs.end());

  const float epsDistance = 8.0e-3f;
  ASSERT_TRUE(CompareHomography(expectedHomography_, actualHomography, epsDistance));
  const size_t numInliers =
      homographyRansac.countInliers(expectedHomography_, points2DPairs.begin(), points2DPairs.end());

  EXPECT_EQ(numInliers, points2DPairs.size() - numCorrupted);
}

}  // namespace test::epipolar
