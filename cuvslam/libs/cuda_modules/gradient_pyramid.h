
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

#include <memory>
#include <vector>

#include "cuda_modules/cuda_convolutor.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/image_pyramid.h"

namespace cuvslam::cuda {

class GPUGradientPyramid {
public:
  GPUGradientPyramid() = default;
  GPUGradientPyramid(size_t width, size_t height);

  int getLevelsCount() const;
  void init(size_t width, size_t height);

  const GaussianGPUImagePyramid& gradX() const;
  const GaussianGPUImagePyramid& gradY() const;

  GaussianGPUImagePyramid& gradX();
  GaussianGPUImagePyramid& gradY();

  GPUGradientPyramid& operator=(const GPUGradientPyramid& grad);
  GPUGradientPyramid& operator=(GPUGradientPyramid&& other) noexcept;
  bool set(const GaussianGPUImagePyramid& image, cudaStream_t& stream, bool horizontal = false);

private:
  void prepare_levels(size_t width, size_t height);
  // number of level with lazy evaluated gradients
  bool horizontal_ = false;

  bool isSet_ = false;
  int levelsCount_ = 0;

  mutable std::unique_ptr<GaussianGPUImagePyramid> gradX_ = nullptr;
  mutable std::unique_ptr<GaussianGPUImagePyramid> gradY_ = nullptr;
};

}  // namespace cuvslam::cuda
