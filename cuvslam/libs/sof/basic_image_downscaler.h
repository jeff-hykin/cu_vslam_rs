
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

#include <vector>

#include "common/image.h"

namespace cuvslam::sof {

// Non-optimized (basic) reference downsampler implementation.
// Downsamples image by a factor of two.
// Input is convolved with a 5x5 smoothing filter before resampling.
class BasicImageDownscaler {
public:
  void compute_new_size(int& w, int& h);

  // Input image must have U8 pixel type.
  void scale(const ImageSource& input, const ImageShape& ishape, ImageSource& output, const ImageShape& oshape);

private:
  std::vector<uint16_t> buffer_;
};

}  // namespace cuvslam::sof
