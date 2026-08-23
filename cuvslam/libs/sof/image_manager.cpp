
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

#include "sof/image_manager.h"

namespace cuvslam::sof {

void ImageManager::init(const ImageShape &shape, size_t num_images_no_depth, bool use_gpu, size_t num_imgs_with_depth) {
  images_data.resize(num_images_no_depth, nullptr);
  for (size_t i = 0; i < num_images_no_depth; i++) {
    images_data[i] = std::make_shared<ImageContext>(shape, use_gpu, false);
  }

  depth_data.resize(num_imgs_with_depth, nullptr);
  for (size_t i = 0; i < num_imgs_with_depth; i++) {
    depth_data[i] = std::make_shared<ImageContext>(shape, use_gpu, true);
  }

  is_initialized_ = true;
}

std::shared_ptr<ImageContext> ImageManager::acquire() {
  std::lock_guard<std::mutex> lock(m_);
  for (std::shared_ptr<ImageContext> &x : images_data) {
    if (x.use_count() == 1) {
      x->reset();  // clear all internal states
      return x;
    }
  }
  return nullptr;
}

std::shared_ptr<ImageContext> ImageManager::acquire_with_depth() {
  std::lock_guard<std::mutex> lock(m_);
  for (std::shared_ptr<ImageContext> &x : depth_data) {
    if (x.use_count() == 1) {
      x->reset();  // clear all internal states
      return x;
    }
  }
  return nullptr;
}

bool ImageManager::is_initialized() const { return is_initialized_; }

}  // namespace cuvslam::sof
