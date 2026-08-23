
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

#include "gflags/gflags.h"

#include "sof/sof_config.h"

DEFINE_int32(num_desired_tracks, 450, "Number of tracks for left selection");

DEFINE_int32(border_top, 0, "Top border to ignore in pixels");
DEFINE_int32(border_bottom, 0, "Bottom border to ignore in pixels");
DEFINE_int32(border_left, 0, "Left border to ignore in pixels");
DEFINE_int32(border_right, 0, "Right border to ignore in pixels");

DEFINE_bool(box3_prefilter, false, "Preprocess input images with box filter");

DEFINE_bool(ransac_filter, false, "Preprocess input images with ransac filter");

DEFINE_string(tracker, "lk", "Feature tracker");

DEFINE_string(lr_tracker, "lk", "L-R feature tracker");

DEFINE_string(multicam_mode, "moderate", "Multicamera mode");

namespace cuvslam::sof {

void ParseSettings(Settings& settings) {
  settings.num_desired_tracks = FLAGS_num_desired_tracks;
  settings.border_top = FLAGS_border_top;
  settings.border_bottom = FLAGS_border_bottom;
  settings.border_left = FLAGS_border_left;
  settings.border_right = FLAGS_border_right;
  settings.box3_prefilter = FLAGS_box3_prefilter;
  settings.ransac_filter = FLAGS_ransac_filter;
  settings.tracker = ParseTrackerType(FLAGS_tracker);
  settings.lr_tracker = ParseTrackerType(FLAGS_lr_tracker);
  settings.multicam_mode = camera::ParseMulticameraMode(FLAGS_multicam_mode);
}

}  // namespace cuvslam::sof
