
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

#include "gflags/gflags.h"

#include "odometry/svo_config.h"

DEFINE_double(survivor_from_last, 41.0,
              "The current frame becomes keyframe if "
              "the percent of survivor tracks from the last keyframe is less than this value");
DEFINE_int64(max_timedelta_between_kfs_s, 60, "max timedelta between consecutive keyframes in seconds");

namespace cuvslam::odom {

void ParseSettings(KeyFrameSettings& settings) {
  settings.survivor_from_last = static_cast<float>(FLAGS_survivor_from_last);
  settings.max_timedelta_between_kfs_s = FLAGS_max_timedelta_between_kfs_s;
}

}  // namespace cuvslam::odom
