
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

#include "common/image.h"

#include "sof/basic_image_downscaler.h"

namespace cuvslam::sof {

class ImagePyramidU8 {
  using ImageSourceAndShape = std::pair<ImageSource, ImageShape>;

public:
  void build(const ImageSource &image, const ImageShape &shape, bool blur_filter);

  int num_levels() const;
  const ImageSourceAndShape &operator[](int i) const;

private:
  static constexpr int MaxLevels = 10;
  int num_levels_ = 0;
  std::array<ImageSourceAndShape, MaxLevels> levels_{};

  std::array<std::vector<char>, MaxLevels> pool_;
  BasicImageDownscaler scaler_;

  void clear();
  void add_level(int width, int height);
};

}  // namespace cuvslam::sof
