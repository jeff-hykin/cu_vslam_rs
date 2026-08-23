
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

#include <array>
#include <utility>

#include "common/types.h"
#include "common/vector_2t.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "cuda_modules/box_prefilter.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

size_t ScaleDownDim(size_t dim);

class ImageGaussianScaler {
public:
  ImageGaussianScaler() = default;
  bool operator()(const GPUImageT& inputImage, GPUImageT& outputImage, cudaStream_t& stream);

private:
  std::unique_ptr<Stream> stream_;
  bool initialized = false;
};

class ImageDummyScaler {
public:
  bool operator()(const GPUImageT& inputImage, GPUImageT& outputImage, cudaStream_t& stream) {
    (void)inputImage;
    (void)outputImage;
    (void)stream;
    return false;
  }
};

template <typename GPUImageType, typename ScalerType, size_t _MaxPyramidLevels>
class GPUImagePyramid {
  static_assert(_MaxPyramidLevels > 1, "Compile time _MaxPyramidLevels validation error");

public:
  using ArrayOfImages = std::array<std::unique_ptr<GPUImageType>, _MaxPyramidLevels>;

  enum { MaxPyramidLevels = _MaxPyramidLevels };

  GPUImagePyramid() = default;

  GPUImagePyramid& operator=(GPUImagePyramid&& other) noexcept {
    levelCount_ = std::exchange(other.levelCount_, 0);
    imageLevels_ = std::move(other.imageLevels_);
    return *this;
  }

  explicit GPUImagePyramid(size_t width, size_t height) { prepare_levels(width, height); }

  void init(size_t width, size_t height) {
    if (levelCount_ == 0) {
      prepare_levels(width, height);
    }
  }

  void build(const GPUImageType& image, bool blur_filter, cudaStream_t& stream) {
    if (levelCount_ == 0) {
      prepare_levels(image.cols(), image.rows());
    }
    if (blur_filter) {
      if (!prefilter_) {
        prefilter_ = std::make_unique<CudaBoxPrefilter>();
      }
      prefilter_->prefilter(image, *imageLevels_[0], stream);
    } else {
      CUDA_CHECK(cudaMemcpy2DAsync(imageLevels_[0]->ptr(), imageLevels_[0]->pitch(), image.ptr(), image.pitch(),
                                   image.cols() * sizeof(float), image.rows(), cudaMemcpyDeviceToDevice, stream));
    }

    for (size_t level = 0; level < levelCount_ - 1; level++) {
      scaler(*imageLevels_[level], *imageLevels_[level + 1], stream);
    }
  }

  GPUImagePyramid& operator=(const GPUImagePyramid& pyramid) {
    if (this == &pyramid) {
      return *this;
    }
    assert(pyramid.levelCount_ == levelCount_);
    assert(pyramid.levelCount_ > 0);

    for (size_t i = 0; i < levelCount_; i++) {
      GPUImageT& this_image = *imageLevels_[i];
      const GPUImageT& pyramid_image = *pyramid.imageLevels_[i];
      assert(this_image.cols() == pyramid_image.cols());
      assert(this_image.rows() == pyramid_image.rows());

      this_image = pyramid_image;
    }
    return *this;
  }

  size_t getLevelsCount() const { return levelCount_; }

  const GPUImageType& operator[](int i) const {
    assert(i < static_cast<int>(levelCount_));
    return *imageLevels_[i];
  }
  GPUImageType& operator[](int i) {
    assert(i < MaxPyramidLevels);
    return *imageLevels_[i];
  }
  const GPUImageType& base() const { return *imageLevels_[0]; }

private:
  void prepare_levels(size_t width, size_t height) {
    TRACE_EVENT ev = profiler_domain_.trace_event("GPUImagePyramid::prepare_levels()", profiler_color_);
    size_t height_ = height;
    size_t width_ = width;

    while (levelCount_ < _MaxPyramidLevels) {
      imageLevels_[levelCount_] = std::make_unique<GPUImageType>(width_, height_);
      levelCount_++;

      height_ = ScaleDownDim(height_);
      width_ = ScaleDownDim(width_);

      if (height_ < MinImageDimSize || width_ < MinImageDimSize) {
        break;
      }
    }
  }

  size_t levelCount_ = 0;
  ArrayOfImages imageLevels_;
  const size_t MinImageDimSize = 15;
  ScalerType scaler;
  std::unique_ptr<CudaBoxPrefilter> prefilter_ = nullptr;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0x00FFd0;
};

const int MAX_PYRAMID_LEVELS = 10;
using GaussianGPUImagePyramid = GPUImagePyramid<GPUImageT, ImageGaussianScaler, MAX_PYRAMID_LEVELS>;
using DummyGPUImagePyramid = GPUImagePyramid<GPUImageT, ImageDummyScaler, MAX_PYRAMID_LEVELS>;

}  // namespace cuvslam::cuda
