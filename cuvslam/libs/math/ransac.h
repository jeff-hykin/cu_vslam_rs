
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "common/log.h"
#include "common/types.h"

namespace cuvslam::math {

// DefaultRandomGenerator class
class DefaultRandomGenerator : public std::mt19937 {
public:
  // Fixed seed, matching the seed(0) the SLAM subsystem's reproduce_mode uses.
  // Seeding from std::random_device made CPU visual odometry nondeterministic
  // run-to-run: identical input produced trajectories differing by ~0.2 m rmse.
  DefaultRandomGenerator() : std::mt19937(0) {}
};

// HypothesisBase class template
template <typename _ScalarType,  // Basic scalar arithmetic type.
          typename _DataType,    // Type of input data to evaluate hypothesis.
          typename _ResultType,  // Type of the hypothesis parameters. For instance: Matrix3T for Fundamental matrix
                                 // evaluation.
          size_t _SampleSize,    // Size of sample set used during hypothesis evaluation.
          size_t _MinCombinations = 10>  // Default minimum number of permutations allowed.
class HypothesisBase {
public:
  using ScalarType = _ScalarType;
  using DataType = _DataType;
  using ResultType = _ResultType;
  using DataSetType = std::vector<DataType>;
  using DataItType = typename DataSetType::const_iterator;
  using RefDataSetType = std::vector<std::reference_wrapper<const DataType>>;
  using RefDataItType = typename RefDataSetType::const_iterator;

  // confidence and iterations limit should be set to sensible values in order to get good results out of RANSAC.
  // These are good initial values but each problem is unique so do some experimentations to find best values for your
  // problem.
  HypothesisBase(ScalarType conf = 0.995, size_t iter = 10000) : confidence_(conf), iterationsLimit_(iter) {}

  void setConfidence(const ScalarType confidence) {
    assert(confidence >= epsilon<ScalarType>());
    assert(confidence <= (1 - epsilon<ScalarType>()));
    confidence_ = confidence;
  }

  void setIterations(const size_t limit) {
    assert(limit > 0);
    iterationsLimit_ = limit;
  }

protected:
  static const size_t SampleSize = _SampleSize;
  static const size_t MinCombinations = _MinCombinations;

  ScalarType confidence_;
  size_t iterationsLimit_;

  // NOTE: This method should be fully declared in derived class and hide this default implementation.
  // Evaluate our hypothesis on the sample dataset.
  template <typename _ItType>
  bool evaluate(ResultType&, const _ItType, const _ItType) const;

  // NOTE: This method should be fully declared in derived class and hide this default implementation.
  // Calculate number of inlier, i.e. samples of data compliant with our hypothesis.
  template <typename _ItType>
  size_t countInliers(const ResultType&, const _ItType, const _ItType) const;

  // Default implementation but could be redefined in derived class for custom RefDataSetType type
  static size_t getSize(const RefDataSetType& d) { return d.size(); }

  // Calculate possible number of different samples we can get from our dataset up to a limit.
  // It will be lesser value of the limit and the binomial coefficient C(n, k) = n!/ (k! * (n - k)!).
  size_t getMaxIterations(const size_t k, size_t n) const {
    assert(k > 0 && iterationsLimit_ > 0);

    if (n <= k) {
      return (n == k) ? 1 : 0;
    }

    size_t res(n), i(1);
    const size_t iterations(std::min(k, n - k));

    while (i < iterations) {
      const size_t update = (res * --n) / ++i;

      // "update < res" is test for overflow
      if (update > iterationsLimit_ || update < res) {
        return iterationsLimit_;
      }

      res = update;
    }

    return res;
  }
};

// Ransac class template
template <typename _HypothesisType,                          // _HypothesisType should derive from HypothesisBase
          typename _RandomGenType = DefaultRandomGenerator>  // Could be replaced for testing
class Ransac : public _HypothesisType {
  using Base = _HypothesisType;
  using typename Base::DataItType;
  using typename Base::RefDataItType;
  using typename Base::RefDataSetType;
  using typename Base::ResultType;
  using typename Base::ScalarType;

  using Base::confidence_;
  using Base::evaluate;
  using Base::getMaxIterations;
  using Base::getSize;
  using Base::MinCombinations;
  using Base::SampleSize;

  mutable _RandomGenType gen_;

public:
  using Base::countInliers;

  template <typename... _Args>
  Ransac(const _Args&... args) : Base(args...) {}

  _RandomGenType& getRandomGenerator() const { return gen_; };

  // Body of RANSAC algorithm
  size_t operator()(ResultType& result, const DataItType beginDataIt, const DataItType endDataIt) const {
    static_assert(MinCombinations > 0, "MinCombinations should be > 0");
    static_assert(SampleSize > 0, "SampleSize should be > 0");

    RefDataSetType shuffled(beginDataIt, endDataIt);
    const size_t dataSize = getSize(shuffled);
    const size_t maxIterations = getMaxIterations(SampleSize, dataSize);

    if (confidence_ < epsilon<ScalarType>() || confidence_ > (1 - epsilon<ScalarType>())) {
      TraceError("Invalid Confidence threshold (%g)", confidence_);
      return 0;
    }

    if (maxIterations < MinCombinations) {
      TraceMessage("Too few points in RANSAC, can't get minimum number of sample variations.");
      return 0;
    }

    const ScalarType logConfidence = std::log(1 - confidence_);
    const ScalarType ratioEpsilon =
        std::max(epsilon<ScalarType>(), std::exp(std::log(1 - std::exp(logConfidence / maxIterations)) / SampleSize));

    const size_t samplesInData = dataSize / SampleSize;
    const size_t maxPasses = (maxIterations - 1) / samplesInData + 1;

    size_t iterationCount = maxIterations;
    ScalarType maxRatio = 0;

    for (size_t pass = 0, sampleCount = 1; pass < maxPasses; pass++) {
      std::shuffle(shuffled.begin(), shuffled.end(), gen_);

      RefDataItType itEnd = shuffled.cbegin();

      for (size_t i = 0; i < samplesInData; i++, sampleCount++) {
        RefDataItType itStart = itEnd;
        std::advance(itEnd, SampleSize);

        ResultType interim;

        if (evaluate(interim, itStart, itEnd)) {
          const size_t count = countInliers(interim, beginDataIt, endDataIt);
          assert(count <= dataSize);
          const ScalarType ratio = ScalarType(count) / ScalarType(dataSize);

          if (ratio > maxRatio) {
            result = interim;
            maxRatio = ratio;
            iterationCount =
                (1 - ratio < epsilon<ScalarType>()) ? 0
                : (ratio < ratioEpsilon)
                    ? iterationCount
                    : std::min(static_cast<size_t>(
                                   logConfidence / std::log(1 - std::pow(ratio, ScalarType(SampleSize))) + 0.5f),
                               iterationCount);
          }
        }

        if (sampleCount >= iterationCount) {
          return (iterationCount < maxIterations) ? sampleCount : 0;
        }
      }
    }

    assert(false);
    return 0;
  }
};

}  // namespace cuvslam::math
