
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

#include "math/ransac.h"

#include "epipolar/homography.h"

namespace cuvslam::epipolar {

namespace {
///-------------------------------------------------------------------------------------------------
/// @brief Convert from iteratorson a  vector of pairs {Data1, Data2} to 2 vectors of Data1 and Data2.
///-------------------------------------------------------------------------------------------------
template <typename Data1, typename Data2>
inline void ConvertVectorOfPairsTo2Vectors(
    typename std::vector<std::reference_wrapper<const std::pair<Data1, Data2>>>::const_iterator vectorOfPairsBegin,
    typename std::vector<std::reference_wrapper<const std::pair<Data1, Data2>>>::const_iterator vectorOfPairsEnd,
    std::vector<Data1>& vectorOfFirst, std::vector<Data2>& vectorOfSecond) {
  const size_t size = std::distance(vectorOfPairsBegin, vectorOfPairsEnd);
  vectorOfFirst.resize(size);
  vectorOfSecond.resize(size);
  auto vectorOfFirstIt = vectorOfFirst.begin();
  auto vectorOfSecondIt = vectorOfSecond.begin();

  for (auto vectorOfPairsIt = vectorOfPairsBegin; vectorOfPairsIt != vectorOfPairsEnd; ++vectorOfPairsIt) {
    (*vectorOfFirstIt++) = vectorOfPairsIt->get().first;
    (*vectorOfSecondIt++) = vectorOfPairsIt->get().second;
  }
}
}  // namespace

/// @brief Find the homography that best explains the 2D points in 2 images.
/// @see HypothesisBase and Ransac
class HomographyRansacHypothesis : public math::HypothesisBase<float, Vector2TPair, Matrix3T, 4> {
public:
  ///-------------------------------------------------------------------------------------------------
  /// @brief Sets the acceptance threshold. This threshold corresponds to the maximum Euclidian
  /// distance acceptable between a point in image 2 and its expected location from the
  /// transformation through the homography of the corresponding point in image 1.
  ///-------------------------------------------------------------------------------------------------
  void setThreshold(float threshold) { threshold_ = threshold; }

  ///-------------------------------------------------------------------------------------------------
  /// @brief The evaluation of the homography hypothesis from a vector of pair of corresponding
  /// points in image 1 and image 2.
  ///
  /// @tparam _ItType Type of the iterator type.
  /// @param [in,out] homographyMat The hypothetic homography matrix.
  /// @param beginIt                The begin iterator of the vector of pair of corresponding points.
  /// @param endIt                  The end iterator of the vector of pair of corresponding points.
  ///
  /// @return true if the homography hypothesis has been found, false otherwise.
  ///-------------------------------------------------------------------------------------------------
  template <typename _ItType>
  bool evaluate(Matrix3T& homographyMat, const _ItType beginIt, const _ItType endIt) const {
    Vector2TVector points1, points2;
    ConvertVectorOfPairsTo2Vectors(beginIt, endIt, points1, points2);

    ComputeHomography homography(points1, points2);
    return ComputeHomography::ReturnCode::Success == homography.findHomography(homographyMat);
  }

  ///-------------------------------------------------------------------------------------------------
  /// @brief Count the inliers for the hypothetic homography.
  ///
  /// @tparam _ItType Type of the iterator type.
  /// @param homographyMat The hypothetic homography matrix.
  /// @param beginIt       The begin iterator of the vector of pair of corresponding points.
  /// @param endIt         The end iterator of the vector of pair of corresponding points.
  ///
  /// @return The number of inliners for which the Euclidian distance, between points in image 1 and
  /// the corresponding points in image 2 transformed by the homography, falls below the set threshold.
  ///-------------------------------------------------------------------------------------------------
  template <typename _ItType>
  size_t countInliers(const Matrix3T& homographyMat, const _ItType beginIt, const _ItType endIt) const {
    return std::count_if(beginIt, endIt, [&](const typename std::iterator_traits<_ItType>::value_type& point2DPair) {
      return ComputeHomographyResidual(point2DPair.first, point2DPair.second, homographyMat) < threshold_;
    });
  }

private:
  float threshold_;
};

using HomographyRansac = math::Ransac<HomographyRansacHypothesis>;

}  // namespace cuvslam::epipolar
