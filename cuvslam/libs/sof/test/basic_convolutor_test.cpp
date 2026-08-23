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
#include "common/types.h"
#include "sof/gaussian_coefficients.h"
#include "sof/kernel_operator.h"
#include "utils/image_loader.h"

namespace test::sof {
using namespace cuvslam;
using namespace cuvslam::sof;

template <int _Rows, int _Cols>
using ImageKernel2DT = ImageMatrixPatch<float, _Rows, _Cols>;

template <int _Cols>
using ImageKernel1DXT = RowVector<float, _Cols>;
template <int _Rows>
using ImageKernel1DYT = ColumnVector<float, _Rows>;

TEST(ImageUtilsTest, GaussianCoefficientsTest) {
  enum { kernelSize = 7 };

  ImageKernel1DXT<kernelSize> valuesRaw(GaussianRawCoeffsT<kernelSize>::CArray());
  ImageKernel1DXT<kernelSize> valuesWeighted(GaussianWeightedCoeffsT<kernelSize>::CArray());
  ImageKernel1DXT<kernelSize> valuesDerivative(GaussianDerivativeCoeffsT<kernelSize>::CArray());

  constexpr int wmid = kernelSize / 2;
  constexpr float sigma = static_cast<float>(kernelSize + 1) / 6;

  std::array<float, kernelSize> compRaw;
  std::array<float, kernelSize> compDerivative;
  float weight = 0;

  for (int i = -wmid; i <= wmid; i++) {
    const float v = exp(-(i * i) / (2 * sigma * sigma));
    compRaw[i + wmid] = v;
    compDerivative[i + wmid] = v * -i / (sigma * sigma);
    weight += v;
  }

  float sum = 0;

  for (size_t i = 0; i < compRaw.size(); i++) {
    const float vWeighted = compRaw[i] / weight;
    sum += vWeighted;

    EXPECT_NEAR(compRaw[i], valuesRaw[i], epsilon());
    EXPECT_NEAR(vWeighted, valuesWeighted[i], epsilon());
    EXPECT_NEAR(compDerivative[i], valuesDerivative[i], epsilon());
  }

  EXPECT_NEAR(1.f, sum, epsilon());

  const ImageKernel1DYT<kernelSize> values2(GaussianWeightedCoeffsT<kernelSize>::CArray());

  const ImageKernel2DT<kernelSize, kernelSize> kernel2D = values2 * valuesWeighted;
  sum = kernel2D.sum();

  EXPECT_NEAR(1.f, sum, 3 * epsilon());
}

}  // namespace test::sof
