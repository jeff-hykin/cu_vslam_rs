
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

#include <iterator>
#include <vector>

#include "math/ransac.h"

#include "epipolar/camera_selection.h"
#include "epipolar/fundamental_matrix_utils.h"

namespace cuvslam::epipolar {

class Fundamental
    : public math::HypothesisBase<float, Vector2TPair, Matrix3T, ComputeFundamental::MINIMUM_NUMBER_OF_POINTS> {
  using Base = math::HypothesisBase<float, Vector2TPair, Matrix3T, ComputeFundamental::MINIMUM_NUMBER_OF_POINTS>;
  using ThresholdWeightFunc = std::function<float(const Vector2T&)>;

public:
  enum Criteria { Epipolar, Cheirality };

  // riteria_ - criteria that we use to count inliers. Epipolar - if x'Fx < threshold, Cheirality - if triangulated
  // point in front of both frames cameras. threshold_ - threshold for epipolar (x'Fx) criteria evaluation, measured in
  // the units of track points motion (pixels, apertures) thresholdAuxiliary_ - threshold on average motion in 2 frames
  // of Fundamental calculation, for ~no motion case calculation of Fundamental is unreliable
  //                        also  measured in the units of track points motion (pixels, apertures)
  void setOptions(const Criteria c, const float t = 0, const float a = 0, const bool e = true) {
    criteria_ = c;
    threshold_ = t;
    thresholdAuxiliary_ = a;
    enforceEssential_ = e;
  }

  // set options at the ctor time
  Fundamental(const Criteria c = Epipolar, const float t = 0, const float a = 0, const bool e = true)
      : Base(), criteria_(c), threshold_(t), thresholdAuxiliary_(a), enforceEssential_(e) {
    thresholdWeight_ = [](const Vector2T&) { return 1.f; };
  }

  float getTheshold() { return threshold_; }

  void setThresholdWeight(const ThresholdWeightFunc& func) { thresholdWeight_ = func; }

private:
  Criteria criteria_;
  float threshold_;
  float thresholdAuxiliary_;
  bool enforceEssential_;
  ThresholdWeightFunc thresholdWeight_;

  template <typename _ItType>
  size_t numInFrontTriangulated(const Matrix3T& fundamentalMat, const _ItType beginIt, const _ItType endIt) const {
    Isometry3T relTransform;
    return (FindOptimalCameraMatrixFromEssential(beginIt, endIt, fundamentalMat, relTransform))
               ? CountPointsInFrontOfCameras(beginIt, endIt, relTransform.inverse())
               : 0;
  }

public:
  // Required method for Ransac, called by Ransac operator()
  template <typename _ItType>
  bool evaluate(Matrix3T& fundamentalMat, _ItType beginIt, _ItType endIt) const {
    float normMotion = 0;
    Vector2TVector points2dImage1;
    Vector2TVector points2dImage2;

    points2dImage1.reserve(std::distance(beginIt, endIt));
    points2dImage2.reserve(std::distance(beginIt, endIt));
    for_each(beginIt, endIt, [&](const Vector2TPair& i) {
      points2dImage1.push_back(i.first);
      points2dImage2.push_back(i.second);
      normMotion += (i.first - i.second).squaredNorm();
    });

    normMotion = std::sqrt(normMotion) / points2dImage1.size();

    if (normMotion < thresholdAuxiliary_) {
      return false;
    }

    ComputeFundamental fundamental(points2dImage1, points2dImage2);

    if (ComputeFundamental::ReturnCode::Success == fundamental.getStatus() || fundamental.isPotentialHomography()) {
      if (fundamental.findFundamental(fundamentalMat) != ComputeFundamental::ReturnCode::NonEssential ||
          !enforceEssential_) {
        return true;
      }
    }

    return false;
  }

  // Required method for Ransac, called by main Ransac method.
  template <typename _ItType>
  size_t countInliers(const Matrix3T& fundamentalMat, const _ItType beginIt, const _ItType endIt) const {
    switch (criteria_) {
      case Epipolar:
        return std::count_if(beginIt, endIt, [&](const Vector2TPair& i) { return isInlier(fundamentalMat, i); });

      case Cheirality:
        return numInFrontTriangulated(fundamentalMat, beginIt, endIt);
    }

    assert(false);
    return 0;
  }

  bool isInlier(const Matrix3T& fundamental, const Vector2TPair& i) const {
    assert(threshold_ > 0);
    auto weight = thresholdWeight_(i.first);
    assert(weight > 0);
    return ComputeQuadraticResidual(i.first, i.second, fundamental) < weight * threshold_;
  }
};

}  // namespace cuvslam::epipolar
