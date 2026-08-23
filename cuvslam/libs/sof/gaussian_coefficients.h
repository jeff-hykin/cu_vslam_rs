
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

#include <cmath>

namespace cuvslam::sof {

enum class GaussianOpType { RAW, WEIGHTED, DERIVATIVE };

template <typename _Scalar, int _Size, GaussianOpType _OpType>
class GaussianCoefficients {
  static_assert(_Size % 2 == 1, "Kernel size should be always an odd number");

  static _Scalar Compute(const _Scalar x, const _Scalar sigmaSq) {
    return std::exp(-x * x / (2 * sigmaSq)) * (_OpType == GaussianOpType::DERIVATIVE ? -x / sigmaSq : 1);
  }

  GaussianCoefficients() {
    const _Scalar sigma = _Scalar(_Size + 1) / _Scalar(6);  // 3 Sigmas on each side
    const _Scalar sigmaSq = sigma * sigma;
    const int centerIndex = _Size / 2;

    weight_ = 0;

    for (int i = 0; i < _Size; i++) {
      const _Scalar c = Compute(_Scalar(i - centerIndex), sigmaSq);
      weight_ += coeffs_[i] = c;
    }

    weightCoeffs(_OpType == GaussianOpType::WEIGHTED);
  }

  void weightCoeffs(const bool isWeighted) {
    if (isWeighted) {
      for (size_t i = 0; i < _Size; i++) {
        coeffs_[i] /= weight_;
      }
    }
  }

  static const GaussianCoefficients& Instance() {
    static const GaussianCoefficients gc;
    return gc;
  }

public:
  typedef _Scalar CArrayType[_Size];

  static const CArrayType& CArray() { return Instance().coeffs_; }

  static _Scalar Weight() { return Instance().weight_; }

private:
  CArrayType coeffs_;
  _Scalar weight_;
};

template <int _Size>
using GaussianRawCoeffsT = GaussianCoefficients<float, _Size, GaussianOpType::RAW>;
template <int _Size>
using GaussianWeightedCoeffsT = GaussianCoefficients<float, _Size, GaussianOpType::WEIGHTED>;
template <int _Size>
using GaussianDerivativeCoeffsT = GaussianCoefficients<float, _Size, GaussianOpType::DERIVATIVE>;

constexpr float GaussianWeightedFeatureCoeffs[9] = {0.0135196f, 0.0476622f, 0.11723f,   0.201168f, 0.240841f,
                                                    0.201168f,  0.11723f,   0.0476622f, 0.0135196f};

constexpr float GaussianWeightedFeatureCoeffs7[] = {0.00598f,  0.060626f, 0.241843f, 0.383103f,
                                                    0.241843f, 0.060626f, 0.00598f};

constexpr float DSPDerivCoeffs[] = {112.0f / 8418.0f,   913.0f / 8418.0f,  3047.0f / 8418.0f, 0,
                                    -3047.0f / 8418.0f, -913.0f / 8418.0f, -112.0f / 8418.0f};

}  // namespace cuvslam::sof
