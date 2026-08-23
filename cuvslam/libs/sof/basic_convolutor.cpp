
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

#include "sof/basic_convolutor.h"

#include "sof/gaussian_coefficients.h"

namespace {
using cuvslam::sof::GaussianWeightedFeatureCoeffs7;

float conv(float v0, float v1, float v2, float v3, float v4, float v5, float v6) noexcept {
  assert(GaussianWeightedFeatureCoeffs7[0] == GaussianWeightedFeatureCoeffs7[6]);
  assert(GaussianWeightedFeatureCoeffs7[1] == GaussianWeightedFeatureCoeffs7[5]);
  assert(GaussianWeightedFeatureCoeffs7[2] == GaussianWeightedFeatureCoeffs7[4]);

  return GaussianWeightedFeatureCoeffs7[0] * (v0 + v6) + GaussianWeightedFeatureCoeffs7[1] * (v1 + v5) +
         GaussianWeightedFeatureCoeffs7[2] * (v2 + v4) + GaussianWeightedFeatureCoeffs7[3] * v3;
}

void convW(const int w, const int h, const float* idata, float* odata) noexcept {
  for (int y = 0; y < h; ++y) {
    const float* s = idata + (w * y);
    float* t = odata + (w * y);

    *t = conv(s[2], s[1], s[0], s[0], s[1], s[2], s[3]);
    ++s;
    ++t;

    *t = conv(s[0], s[-1], s[-1], s[0], s[1], s[2], s[3]);
    ++s;
    ++t;

    *t = conv(s[-2], s[-2], s[-1], s[0], s[1], s[2], s[3]);
    ++s;
    ++t;

    for (int x = 3; x < w - 3; ++x) {
      *t = conv(s[-3], s[-2], s[-1], s[0], s[1], s[2], s[3]);
      ++s;
      ++t;
    }

    *t = conv(s[-3], s[-2], s[-1], s[0], s[1], s[2], s[2]);
    ++s;
    ++t;

    *t = conv(s[-3], s[-2], s[-1], s[0], s[1], s[1], s[0]);
    ++s;
    ++t;

    *t = conv(s[-3], s[-2], s[-1], s[0], s[0], s[-1], s[-2]);
    ++s;
    ++t;
  }
}

void convH(const int w, const int h, const float* idata, float* odata) noexcept {
  {
    const float* s = idata;
    float* t = odata;
    for (int x = 0; x < w; ++x) {
      *t = conv(s[2 * w], s[w], s[0], s[0], s[w], s[2 * w], s[3 * w]);
      ++s;
      ++t;
    }
  }

  {
    const float* s = idata + w;
    float* t = odata + w;
    for (int x = 0; x < w; ++x) {
      *t = conv(s[0], s[-w], s[-w], s[0], s[w], s[2 * w], s[3 * w]);
      ++s;
      ++t;
    }
  }

  {
    const float* s = idata + 2 * w;
    float* t = odata + 2 * w;
    for (int x = 0; x < w; ++x) {
      *t = conv(s[-2 * w], s[-2 * w], s[-w], s[0], s[w], s[2 * w], s[3 * w]);
      ++s;
      ++t;
    }
  }

  for (int y = 3; y < h - 3; ++y) {
    const float* s = idata + (w * y);
    float* t = odata + (w * y);
    for (int x = 0; x < w; ++x) {
      *t = conv(s[-3 * w], s[-2 * w], s[-w], s[0], s[w], s[2 * w], s[3 * w]);
      ++s;
      ++t;
    }
  }

  {
    const float* s = idata + (w * (h - 3));
    float* t = odata + (w * (h - 3));
    for (int x = 0; x < w; ++x) {
      *t = conv(s[-3 * w], s[-2 * w], s[-w], s[0], s[w], s[2 * w], s[2 * w]);
      ++s;
      ++t;
    }
  }

  {
    const float* s = idata + (w * (h - 2));
    float* t = odata + (w * (h - 2));
    for (int x = 0; x < w; ++x) {
      *t = conv(s[-3 * w], s[-2 * w], s[-w], s[0], s[w], s[w], s[0]);
      ++s;
      ++t;
    }
  }

  {
    const float* s = idata + (w * (h - 1));
    float* t = odata + (w * (h - 1));
    for (int x = 0; x < w; ++x) {
      *t = conv(s[-3 * w], s[-2 * w], s[-w], s[0], s[0], s[-w], s[-2 * w]);
      ++s;
      ++t;
    }
  }
}

}  // namespace

namespace cuvslam::sof {

void BasicConvolutor::convKernelGradDerivX(const ImageMatrixT& in, ImageMatrixT& out) {
  assert(in.rows() == out.rows() && in.cols() == out.cols());
  const Index h = in.rows();
  const Index w = in.cols();

  for (Index y = 0; y < h; y++) {
    for (Index x = 3; x < w - 3; x++) {
      float sum = 0;

      for (int i = 0; i < 7; ++i) {
        sum += DSPDerivCoeffs[i] * in(y, x - 3 + i);
      }

      out(y, x) = sum;
    }
  }

  out.block(0, 0, h, 3).setZero();
  out.block(0, w - 3, h, 3).setZero();
}

void BasicConvolutor::convKernelGradDerivY(const ImageMatrixT& in, ImageMatrixT& out) {
  assert(in.rows() == out.rows() && in.cols() == out.cols());
  const Index h = in.rows();
  const Index w = in.cols();

  for (Index y = 3; y < h - 3; y++) {
    for (Index x = 0; x < w; x++) {
      float sum = 0;

      for (int i = 0; i < 7; ++i) {
        sum += DSPDerivCoeffs[i] * in(y - 3 + i, x);
      }

      out(y, x) = sum;
    }
  }

  out.block(0, 0, 3, w).setZero();
  out.block(h - 3, 0, 3, w).setZero();
}

void BasicConvolutor::convKernelFeatX(const ImageMatrixT& in, ImageMatrixT& out) {
  assert(in.IsRowMajor);
  assert(out.IsRowMajor);
  assert(in.rows() == out.rows() && in.cols() == out.cols());
  convW(static_cast<int>(in.cols()), static_cast<int>(in.rows()), in.data(), out.data());
}

void BasicConvolutor::convKernelFeatY(const ImageMatrixT& in, ImageMatrixT& out) {
  assert(in.IsRowMajor);
  assert(out.IsRowMajor);
  assert(in.rows() == out.rows() && in.cols() == out.cols());
  convH(static_cast<int>(in.cols()), static_cast<int>(in.rows()), in.data(), out.data());
}

}  // namespace cuvslam::sof
