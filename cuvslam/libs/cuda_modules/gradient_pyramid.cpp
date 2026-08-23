
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

#include "cuda_modules/gradient_pyramid.h"

#include <utility>

#include "cuda_modules/cuda_convolutor.h"
#include "cuda_modules/image_pyramid.h"

namespace cuvslam::cuda {

GPUGradientPyramid::GPUGradientPyramid(size_t width, size_t height) { prepare_levels(width, height); }

void GPUGradientPyramid::prepare_levels(size_t width, size_t height) {
  CUDA_CHECK(init_conv_kernels());
  gradX_ = std::make_unique<GaussianGPUImagePyramid>(width, height);
  gradY_ = std::make_unique<GaussianGPUImagePyramid>(width, height);

  levelsCount_ = gradX_->getLevelsCount();
}

int GPUGradientPyramid::getLevelsCount() const { return levelsCount_; }

void GPUGradientPyramid::init(size_t width, size_t height) {
  if (levelsCount_ == 0) {
    prepare_levels(width, height);
  }
}

const GaussianGPUImagePyramid& GPUGradientPyramid::gradX() const {
  assert(gradX_ != nullptr);
  return *gradX_;
}
const GaussianGPUImagePyramid& GPUGradientPyramid::gradY() const {
  assert(gradY_ != nullptr);
  return *gradY_;
}

GaussianGPUImagePyramid& GPUGradientPyramid::gradX() {
  assert(gradY_ != nullptr);
  return *gradX_;
}
GaussianGPUImagePyramid& GPUGradientPyramid::gradY() {
  assert(gradY_ != nullptr);
  return *gradY_;
}

GPUGradientPyramid& GPUGradientPyramid::operator=(const GPUGradientPyramid& grad) {
  if (this == &grad) {
    return *this;
  }
  assert(levelsCount_ == grad.levelsCount_);
  assert(levelsCount_ > 0);
  *gradX_ = *grad.gradX_;
  *gradY_ = *grad.gradY_;
  return *this;
}

GPUGradientPyramid& GPUGradientPyramid::operator=(GPUGradientPyramid&& other) noexcept {
  horizontal_ = std::exchange(other.horizontal_, false);
  isSet_ = std::exchange(other.isSet_, false);
  levelsCount_ = std::exchange(other.levelsCount_, 0);
  gradX_ = std::move(other.gradX_);
  gradY_ = std::move(other.gradY_);
  return *this;
}

bool GPUGradientPyramid::set(const GaussianGPUImagePyramid& image, cudaStream_t& stream, bool horizontal) {
  if (levelsCount_ == 0) {  // TODO: get rid of prepare_levels here
    prepare_levels(image[0].cols(), image[0].rows());
  }
  horizontal_ = horizontal;
  isSet_ = false;

  const int levels = image.getLevelsCount();
  assert(gradX_->base().rows() == image[0].rows());
  assert(gradX_->base().cols() == image[0].cols());
  assert(gradY_->base().cols() == image[0].cols());
  assert(gradY_->base().cols() == image[0].cols());

  unsigned int h;
  unsigned int w;

  for (int level = 0; level < levels; level++) {
    const GPUImageT& in = image[level];
    // calculate gradients for all pixels
    GPUImageT& grad_x = gradX_->operator[](level);

    h = in.rows();
    w = in.cols();

    CUDA_CHECK(conv_grad_x(in.get_texture_filter_point(), {w, h}, grad_x.ptr(), grad_x.pitch(), stream));

    if (!horizontal_) {
      GPUImageT& grad_y = gradY_->operator[](level);

      CUDA_CHECK(conv_grad_y(in.get_texture_filter_point(), {w, h}, grad_y.ptr(), grad_y.pitch(), stream));
    }
  }

  isSet_ = true;
  return true;
}

}  // namespace cuvslam::cuda
