
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

#include "sof/gftt.h"

namespace {

float my_log1p(float x) noexcept {
  return 2 * x / (x + 2);  // very fast approximation of log(1+x)
}

float GFTTMeasure(const float gxx, const float gxy, const float gyy) noexcept {
  assert(gxx >= 0 && gyy >= 0);
  const float D = std::sqrt((gxx - gyy) * (gxx - gyy) + 4.f * gxy * gxy);
  const float T = gxx + gyy;
  const float eMin = (T - D) / 2.f;
  /* experimental
  const _Scalar eMax = (T + D) / 2.f;

  if (eMin < 0.02f) {
      return 0;
  }
  if (eMax / eMin > 100.f) {
      return 0;
  } */

  return my_log1p(eMin);
}

}  // namespace

namespace cuvslam::sof {

GFTT::GFTT() : convolutor_(CreateConvolutor()) {}

void GFTT::compute(const ImageMatrixT& gradX, const ImageMatrixT& gradY) {
  assert(gradX.rows() == gradY.rows() && gradX.cols() == gradY.cols());
  const Index n_rows = gradX.rows();
  const Index n_cols = gradX.cols();

  values_.resize(n_rows, n_cols);

  // Convolve XY, X2 and Y2 grads with Gaussian weighted kernel
  gradXY_ = gradX.cwiseProduct(gradY);  // XY gradient
  convolutor_->convKernelFeatX(gradXY_, values_);
  convolutor_->convKernelFeatY(values_, gradXY_);

  gradXX_ = gradX.cwiseProduct(gradX);  // X^2 gradient
  convolutor_->convKernelFeatX(gradXX_, values_);
  convolutor_->convKernelFeatY(values_, gradXX_);

  gradYY_ = gradY.cwiseProduct(gradY);  // Y^2 gradient
  convolutor_->convKernelFeatX(gradYY_, values_);
  convolutor_->convKernelFeatY(values_, gradYY_);

  const Index size = values_.size();
  assert(size == gradXX_.size());
  assert(size == gradXY_.size());
  assert(size == gradYY_.size());

  const float* xx_data = gradXX_.data();
  const float* xy_data = gradXY_.data();
  const float* yy_data = gradYY_.data();
  float* g_data = values_.data();

  for (Index i = 0; i < size; ++i) {
    g_data[i] = GFTTMeasure(xx_data[i], xy_data[i], yy_data[i]);
  }
}

const ImageMatrixT& GFTT::get() const { return values_; }

}  // namespace cuvslam::sof
