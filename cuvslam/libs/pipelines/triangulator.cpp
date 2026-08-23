
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

#include "pipelines/triangulator.h"

#include <numeric>

#include "epipolar/camera_selection.h"

namespace cuvslam::pipelines {

using obs_ref = std::reference_wrapper<const camera::Observation>;

MulticamTriangulator::MulticamTriangulator(const camera::Rig& rig) : rig_(rig) {}

std::vector<Landmark> MulticamTriangulator::triangulate(const Isometry3T& world_from_rig,
                                                        const std::vector<camera::Observation>& observations) {
  TRACE_EVENT ev = profiler_domain_.trace_event("triangulate");

  std::vector<Landmark> out;
  out.reserve(observations.size());

  std::vector<Isometry3T> world_from_cam(rig_.num_cameras);
  for (CameraId cam_id = 0; cam_id < static_cast<CameraId>(rig_.num_cameras); cam_id++) {
    world_from_cam[cam_id] = world_from_rig * rig_.camera_from_rig[cam_id].inverse();
  }

  std::unordered_map<TrackId, std::vector<obs_ref>> obs_map;
  obs_map.reserve(observations.size());

  for (const auto& obs : observations) {
    obs_map[obs.id].push_back(std::cref(obs));
  }

  visible_tracks_.clear();
  for (const auto& [track_id, obs_refs] : obs_map) {
    visible_tracks_.insert(track_id);

    if (triangulated_tracks_.find(track_id) == triangulated_tracks_.end()) {
      if (obs_refs.size() < 2) {
        continue;
      }

      const camera::Observation& o1 = obs_refs[0].get();
      const camera::Observation& o2 = obs_refs[1].get();

      assert(o1.cam_id != o2.cam_id);

      const Isometry3T& world_from_cam1 = world_from_cam[o1.cam_id];

      Isometry3T cam1_from_cam2 = rig_.camera_from_rig[o1.cam_id] * rig_.camera_from_rig[o2.cam_id].inverse();

      Vector3T xyz;
      epipolar::TriangulationState ts;
      // measures how orthogonal the rays are (sine of the angle between them)
      float pm;
      if (epipolar::OptimalTriangulation(cam1_from_cam2, o1.xy, o2.xy, xyz, pm, ts)) {
        out.push_back({track_id, world_from_cam1 * xyz});
        triangulated_tracks_.insert(track_id);
      }
    }
  }

  visible_traingulated_tracks_.clear();
  std::set_intersection(visible_tracks_.begin(), visible_tracks_.end(), triangulated_tracks_.begin(),
                        triangulated_tracks_.end(),
                        std::inserter(visible_traingulated_tracks_, visible_traingulated_tracks_.begin()));

  triangulated_tracks_ = visible_traingulated_tracks_;

  return out;
}

void MulticamTriangulator::reset() {
  visible_tracks_.clear();
  visible_traingulated_tracks_.clear();
  triangulated_tracks_.clear();
}
}  // namespace cuvslam::pipelines
