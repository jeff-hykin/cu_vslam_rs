
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

#include <mutex>
#include <vector>

#include "sof/image_context.h"

namespace cuvslam::sof {

class ImageManager {
public:
  void init(const ImageShape& shape, size_t num_images_no_depth, bool use_gpu, size_t num_imgs_with_depth = 0);

  ImageContextPtr acquire();
  ImageContextPtr acquire_with_depth();

  bool is_initialized() const;

private:
  bool is_initialized_ = false;
  std::vector<ImageContextPtr> images_data;  // images without available depth
  std::vector<ImageContextPtr> depth_data;   // images with properly aligned depth
  std::mutex m_;
};

}  // namespace cuvslam::sof
