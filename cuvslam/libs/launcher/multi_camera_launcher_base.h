
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

#include "camera/frustum_intersection_graph.h"

#include "launcher/base_launcher.h"

namespace cuvslam::launcher {
class MultiCameraBaseLauncher : public BaseLauncher {
public:
  // `auto_allow_stereo_track_for_depth` lets a subclass force the FIG to keep secondary edges
  // to depth-aligned cameras even when the global -allow_stereo_track_for_depth flag is off.
  // Used by MultisensorCameraLauncher (multisensor mode always wants those cross-camera tracks).
  MultiCameraBaseLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings,
                          bool auto_allow_stereo_track_for_depth = false);

  // Predicate used by BaseLauncher::launch() to size the image pool and to decide which cameras
  // get acquire_with_depth(). Membership in the resolved depth-id list (FIG-owned global flag).
  bool isDepthCamera(CameraId id) const override;

protected:
  camera::FrustumIntersectionGraph fig;
  // Resolved at ctor from -fig_depth_camera_ids (owned by multi_camera_launcher_base.cpp);
  // consumed by both the FIG and the BaseLauncher::launch() ingestion path so the two can never
  // drift.
  std::vector<CameraId> depth_ids_;
};
}  // namespace cuvslam::launcher
