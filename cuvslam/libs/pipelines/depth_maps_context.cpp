
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

#include "pipelines/depth_maps_context.h"

#include <algorithm>

#include "common/vector_2t.h"

namespace cuvslam::pipelines {

DepthMapsContext::DepthMapsContext(const camera::Rig& rig) : rig_(rig), triangulator_(rig), track_lifter_(rig) {}

void DepthMapsContext::reset() {
  triangulator_.reset();
  plane_map_.clear();
  depth_point_map_.clear();
}

bool DepthMapsContext::update_post_solve(const pnp::RGBDInfos& depth_infos, const Isometry3T& world_from_rig) {
  if (std::none_of(depth_infos.begin(), depth_infos.end(), [](const pnp::RGBDInfo* info) { return info != nullptr; })) {
    return false;
  }
  auto depth_cameras = build_depth_camera_infos(depth_infos, world_from_rig);
  depth_point_map_.update(depth_cameras);
  return depth_point_map_.added_last_update();
}

std::vector<Landmark> DepthMapsContext::build_keyframe_landmarks(const Isometry3T& world_from_rig,
                                                                 const std::vector<camera::Observation>& observations,
                                                                 const pnp::RGBDInfos& depth_infos) {
  std::vector<Landmark> landmarks = triangulator_.triangulate(world_from_rig, observations);

  for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
    const pnp::RGBDInfo* depth_info = depth_infos[cam_id];
    if (depth_info == nullptr) {
      continue;
    }
    std::vector<camera::Observation> camera_observations;
    camera_observations.reserve(observations.size());
    for (const auto& obs : observations) {
      if (obs.cam_id == cam_id) {
        camera_observations.push_back(obs);
      }
    }
    std::vector<Landmark> mono_landmarks;
    track_lifter_.lift_tracks(*depth_info, world_from_rig, camera_observations, mono_landmarks);
    std::move(mono_landmarks.begin(), mono_landmarks.end(), std::back_inserter(landmarks));
  }
  return landmarks;
}

void DepthMapsContext::update_at_keyframe(const pnp::RGBDInfos& depth_infos, const Isometry3T& world_from_rig) {
  if (std::none_of(depth_infos.begin(), depth_infos.end(), [](const pnp::RGBDInfo* info) { return info != nullptr; })) {
    return;
  }
  auto depth_cameras = build_depth_camera_infos(depth_infos, world_from_rig);
  plane_map_.update_at_keyframe(depth_cameras);
}

std::vector<map::DepthCameraInfo> DepthMapsContext::build_depth_camera_infos(const pnp::RGBDInfos& depth_infos,
                                                                             const Isometry3T& world_from_rig) const {
  std::vector<map::DepthCameraInfo> depth_cameras;
  depth_cameras.reserve(
      std::count_if(depth_infos.begin(), depth_infos.end(), [](const pnp::RGBDInfo* info) { return info != nullptr; }));
  for (CameraId cam_id = 0; cam_id < depth_infos.size(); ++cam_id) {
    const pnp::RGBDInfo* depth_info = depth_infos[cam_id];
    if (depth_info == nullptr) {
      continue;
    }
    const auto& intrinsics = *rig_.intrinsics[cam_id];
    Vector2T focal = intrinsics.getFocal();
    Vector2T principal = intrinsics.getPrincipal();
    depth_cameras.push_back(
        {depth_info->curr_depth[0].get_texture_filter_point(),
         {focal.x(), focal.y()},
         {principal.x(), principal.y()},
         {static_cast<int>(depth_info->curr_depth[0].cols()), static_cast<int>(depth_info->curr_depth[0].rows())},
         // world_from_cam = world_from_rig * (camera_from_rig)^{-1}
         world_from_rig * rig_.camera_from_rig[cam_id].inverse()});
  }
  return depth_cameras;
}

}  // namespace cuvslam::pipelines
