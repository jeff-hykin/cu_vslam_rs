
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

#include <chrono>
#include <thread>

namespace cuvslam {

struct ScopedThrottler {
  using clock = std::chrono::steady_clock;

  ScopedThrottler(double fps) : fps_(fps), start_(clock::now()) {}

  ~ScopedThrottler() {
    if (fps_ == 0.) {
      return;
    }
    const auto min_duration = std::chrono::duration<double>(1.0 / fps_);
    const auto elapsed = clock::now() - start_;
    if (elapsed < min_duration) {
      std::this_thread::sleep_for(min_duration - elapsed);
    }
  }

  double fps_;
  clock::time_point start_;
};

}  // namespace cuvslam
