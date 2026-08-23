
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

#include "launcher/visual_inertial_launcher.h"

#include "gflags/gflags.h"

DEFINE_bool(debug_imu_mode, false, "Enable IMU debug mode (pure IMU integration, no visual tracking)");

namespace cuvslam::launcher {
VisualInertialLauncher::VisualInertialLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings)
    : MultiCameraBaseLauncher(cameraRig, svo_settings) {
  TraceMessage("Inertial launcher is selected");
}

void VisualInertialLauncher::SetupTracker(const odom::Settings& svo_settings, bool use_gpu) {
  tracker = std::make_unique<odom::StereoInertialOdometry>(rig, fig, svo_settings, use_gpu, FLAGS_debug_imu_mode);
  cameraRig_.registerIMUCallback(
      [&](const imu::ImuMeasurement& measurement) { tracker->add_imu_measurement(measurement); });

  tracker->enable_stat(true);
}

bool VisualInertialLauncher::launch_vo(Isometry3T& delta, Matrix6T& pose_info,
                                       const odom::TrackPerFrameSettings& per_frame) {
  return tracker->track(curr_sources, {}, curr_image_ptrs, prev_image_ptrs, masks_sources, delta, pose_info, per_frame);
}

const odom::IVisualOdometry::VOFrameStat& VisualInertialLauncher::last_vo_stat() { return *tracker->get_last_stat(); }
}  // namespace cuvslam::launcher
