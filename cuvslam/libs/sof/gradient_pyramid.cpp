
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

#include "sof/gradient_pyramid.h"

#include "common/types.h"

#include "sof/convolutor.h"
#include "sof/gaussian_coefficients.h"
#include "sof/image_pyramid_float.h"

namespace {
using cuvslam::ImageMatrixT;
using cuvslam::Index;
using cuvslam::sof::DSPDerivCoeffs;

void CalcPixelGradients(const ImageMatrixT &in, Index x, Index y, ImageMatrixT &grad_x, ImageMatrixT &grad_y) noexcept {
  assert(3 <= x && x < in.cols() - 3);
  assert(3 <= y && y < in.rows() - 3);
  assert(in.size() == grad_x.size() && grad_x.size() == grad_y.size());

  float sum_x = 0;
  float sum_y = 0;

  for (int i = 0; i < 7; ++i) {
    const float dsp = DSPDerivCoeffs[i];
    sum_x += dsp * in(y, x - 3 + i);
    sum_y += dsp * in(y - 3 + i, x);
  }
  grad_x(y, x) = sum_x;
  grad_y(y, x) = sum_y;
}

void CalcPixelGradientsHorizontal(const ImageMatrixT &in, Index x, Index y, ImageMatrixT &grad_x) noexcept {
  assert(3 <= x && x < in.cols() - 3);
  assert(3 <= y && y < in.rows() - 3);
  assert(in.size() == grad_x.size());

  float sum_x = 0;

  for (int i = 0; i < 7; ++i) {
    const float dsp = DSPDerivCoeffs[i];
    sum_x += dsp * in(y, x - 3 + i);
  }
  grad_x(y, x) = sum_x;
}

void ZeroBorder(ImageMatrixT &image, Index border_width) noexcept {
  const Index n_cols = image.cols();
  const Index n_rows = image.rows();

  assert(border_width < n_cols && border_width < n_rows);

  image.block(0, 0, border_width, n_cols).setZero();
  image.block(n_rows - border_width, 0, border_width, n_cols).setZero();
  image.block(0, 0, n_rows, border_width).setZero();
  image.block(0, n_cols - border_width, n_rows, border_width).setZero();
}

}  // namespace

namespace cuvslam::sof {

static const int kLazyEvaluatedLevels = 1;

GradientPyramidT::GradientPyramidT() : n_lazy_evaluated_levels_(kLazyEvaluatedLevels), convolutor_(CreateConvolutor()) {
  evaluated_.resize(static_cast<size_t>(n_lazy_evaluated_levels_));
}

GradientPyramidT::GradientPyramidT(const GradientPyramidT &arg, const ImagePyramidT *image)
    : n_lazy_evaluated_levels_(arg.n_lazy_evaluated_levels_), convolutor_(CreateConvolutor()) {
  horizontal_ = arg.horizontal_;
  isSet_ = arg.isSet_;
  levelsCount_ = arg.levelsCount_;

  gradX_ = arg.gradX_;
  gradY_ = arg.gradY_;
  evaluated_ = arg.evaluated_;
  image_ = image;
}

int GradientPyramidT::getLevelsCount() const { return levelsCount_; }

void GradientPyramidT::setNumLevels(int numLevels) {
  gradX_.setLevelsCount(numLevels);
  gradY_.setLevelsCount(numLevels);
  levelsCount_ = numLevels;
  isSet_ = true;
}

bool GradientPyramidT::isGradientsLazyEvaluated(int level) const noexcept { return level < n_lazy_evaluated_levels_; }

ImageNoScalePyramidT &GradientPyramidT::gradX() { return gradX_; }
const ImageNoScalePyramidT &GradientPyramidT::gradX() const { return gradX_; }

ImageNoScalePyramidT &GradientPyramidT::gradY() { return gradY_; }
const ImageNoScalePyramidT &GradientPyramidT::gradY() const { return gradY_; }

bool GradientPyramidT::set(const ImagePyramidT &image, bool horizontal) {
  image_ = &image;
  horizontal_ = horizontal;

  isSet_ = false;
  const int levels = image.getLevelsCount();
  assert(levels > 0 && levels <= ImagePyramidT::MaxPyramidLevels);

  for (int level = 0; level < levels; level++) {
    // Validation of image will happen in Conv2D so no extra validation is needed
    // compute X and Y gradients
    const ImageMatrixT &in = image[level];
    const Index n_cols = in.cols();
    const Index n_rows = in.rows();
    ImageMatrixT &out_x = gradX_[level];
    ImageMatrixT &out_y = gradY_[level];
    out_x.resize(n_rows, n_cols);

    if (!horizontal_) {
      out_y.resize(n_rows, n_cols);
    }

    if (isGradientsLazyEvaluated(level)) {
      // this is lazy gradients evaluated level
      assert(level < static_cast<int>(evaluated_.size()));
      evaluated_[level].resize(n_rows, n_cols);
      evaluated_[level].setConstant(false);
      ZeroBorder(out_x, 3);
      if (!horizontal_) {
        ZeroBorder(out_y, 3);
      }
    } else {
      // calculate gradients for all pixels
      convolutor_->convKernelGradDerivX(in, out_x);
      if (!horizontal_) {
        convolutor_->convKernelGradDerivY(in, out_y);
      }
    }
  }

  gradX_.setLevelsCount(levels);
  gradY_.setLevelsCount(levels);
  levelsCount_ = levels;
  return (isSet_ = true);
}

void GradientPyramidT::forceNonLazyEvaluation(const int level) const {
  const ImageMatrixT &in = (*image_)[level];
  convolutor_->convKernelGradDerivX(in, gradX_[level]);
  convolutor_->convKernelGradDerivY(in, gradY_[level]);
  evaluated_[level].setConstant(true);
}

void GradientPyramidT::calcPatchGradients(const Vector2T &xy, int Dim, int level) const noexcept {
  if (!isGradientsLazyEvaluated(level)) {
    return;
  }
  assert(level < static_cast<int>(evaluated_.size()));

  const ImageMatrixT &in = (*image_)[level];
  ImageMatrixT &grad_x = gradX_[level];
  ImageMatrixT &grad_y = gradY_[level];

  Vector2<int32_t> tlxy, brxy;
  compute_patch_bbox(xy, Dim, tlxy, brxy);

  int32_t b = 3;  // zero border width
  const auto w = static_cast<int32_t>(in.cols());
  const auto h = static_cast<int32_t>(in.rows());

  // crop border 3
  const Index x_start = std::max(b, tlxy.x());
  const Index x_end = std::min(brxy.x(), w - 1 - b);

  const Index y_start = std::max(b, tlxy.y());
  const Index y_end = std::min(brxy.y(), h - 1 - b);

  for (Index y = y_start; y <= y_end; ++y) {
    for (Index x = x_start; x <= x_end; ++x) {
      bool &evaluated = evaluated_[level](y, x);
      if (!evaluated) {
        if (!horizontal_) {
          CalcPixelGradients(in, x, y, grad_x, grad_y);
        } else {
          CalcPixelGradientsHorizontal(in, x, y, grad_x);
        }
        evaluated = true;
      }
    }
  }
}

}  // namespace cuvslam::sof
