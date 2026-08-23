
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

#include "pipelines/inertial_pnp.h"

#include <unordered_map>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/log.h"
#include "common/vector_3t.h"
#include "imu/soft_inertial_pnp.h"

#include "pipelines/track.h"

namespace cuvslam::pipelines {

bool InertialPnP::Solve(const imu::ImuCalibration& calib, const std::unordered_map<TrackId, Track>& tracks3d,
                        const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                        const Vector3T& gravity_w, const InertialPnPSettings& settings,
                        sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                        sba_imu::Pose& curr_pose) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("Solve");

  sba_imu::StereoPnPInput input;
  input.gravity = gravity_w;
  input.prior_acc = 0.0f;
  input.prior_gyro = 0.0f;
  input.freeze_bias = true;
  input.rig = rig;

  if (settings.max_iteration < 0 || settings.min_observations < 0) {
    TraceError("InertialPnPSettings count fields must be non-negative");
    return false;
  }

  input.robustifier_scale = settings.robustifier_scale;
  input.imu_penalty = 1e-3f;
  input.robustifier_scale_pose = -1.0f;
  input.max_iterations = settings.max_iteration;
  input.outlier_thresh = {10.f, 7.f};
  input.translation_constraint = 0;
  input.robustifier_scale_tr = 0.02f;

  // Scale observation info from the default σ=3px to σ=1px (9x stronger visual weight).
  constexpr float visual_noise_px = 1.0f;
  constexpr float info_scale = (3.0f / visual_noise_px) * (3.0f / visual_noise_px);

  std::unordered_map<int, int> track_to_point;
  for (const auto& track : observations) {
    const TrackId track_id = track.id;
    const auto it = tracks3d.find(track_id);
    if (it == tracks3d.end()) {
      continue;
    }
    const Track& track3d = it->second;

    if (track3d.hasLocation()) {
      const Vector3T p_w = track3d.getLocation3D();
      int point_id = static_cast<int>(input.points.size());
      auto it = track_to_point.find(track_id);
      if (it == track_to_point.end()) {
        track_to_point[track_id] = point_id;
        input.points.push_back(p_w);
      } else {
        point_id = it->second;
      }
      input.point_ids.push_back(point_id);
      input.camera_ids.push_back(track.cam_id);
      input.observation_xys.push_back(track.xy);
      input.observation_infos.push_back(track.xy_info * info_scale);
    }
  }

  TraceMessage("InertialPnP: obs=%d points=%d", (int)input.observation_xys.size(), (int)input.points.size());

  if (input.observation_xys.size() < static_cast<size_t>(settings.min_observations)) {
    TraceMessage("InertialPnP: REJECTED obs < %d", settings.min_observations);
    return false;
  }

  TRACE_EVENT ev1 = profiler_domain_.trace_event("SoftInertialPnP");

  bool status = SoftInertialPnP(calib, input, prev_pose, curr_pose, input.imu_penalty);
  // bool status = SoftInertialPnPWithOutliers(calib, input, prev_pose, curr_pose,   input.imu_penalty);

  TraceMessage("InertialPnP: status=%d", status);
  return status;
}

}  // namespace cuvslam::pipelines
