
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

#include "sof/box_blur.h"

#include <cstdint>

#include "common/types.h"

namespace cuvslam::sof {

void BoxBlur3Cpu(const cuvslam::ImageMatrix<uint8_t>& input, uint8_t* output) {
  const size_t w = input.cols();
  const size_t h = input.rows();

  cuvslam::Matrix3T kernel;
  kernel << 1.f / 16, 1.f / 8, 1.f / 16, 1.f / 8, 1.f / 4, 1.f / 8, 1.f / 16, 1.f / 8, 1.f / 16;

  cuvslam::Matrix3T window;

  size_t x, y;
  for (y = 1; y < h - 1; ++y) {
    for (x = 1; x < w - 1; ++x) {
      window = input.block<3, 3>(y - 1, x - 1).cast<float>();
      float sum = window.cwiseProduct(kernel).sum();
      output[x + y * w] = static_cast<uint8_t>(sum);
    }
  }
  for (x = 0; x < w; ++x) {
    y = 0;
    output[x + y * w] = input(y, x);
    y = h - 1;
    output[x + y * w] = input(y, x);
  }
  for (y = 1; y < h - 1; ++y) {
    x = 0;
    output[x + y * w] = input(y, x);
    x = w - 1;
    output[x + y * w] = input(y, x);
  }
}

void BoxBlur3(const ImageMatrix<uint8_t>& input, uint8_t* output) {
  assert(output != nullptr);
  BoxBlur3Cpu(input, output);
}

}  // namespace cuvslam::sof
