
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

#include <cstdint>

namespace cuvslam::pnp {

struct ICPSettings {
  float lambda = 1e-2;
  float huber_vis = 1e-2;

  float huber_depth = 5e-2;

  int32_t max_iteration = 20;
  bool verbose = false;
  float cost_thresh = 0.6;

  int32_t min_scale_level = 0;
  int32_t max_scale_level = 4;
  int32_t num_iters_per_scale = 20;

  float blending_alpha = 0.8f;
};

}  // namespace cuvslam::pnp
