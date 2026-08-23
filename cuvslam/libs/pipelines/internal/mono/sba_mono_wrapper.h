
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

#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "camera/observation.h"
#include "common/frame_id.h"
#include "common/isometry.h"
#include "common/track_id.h"
#include "common/vector_3t.h"

#include "pipelines/track.h"

namespace cuvslam::pipelines {

struct TrackingState {
  // int64_t time_ns;
  Isometry3T world_from_rig = Isometry3T::Identity();
  Vector3T velocity = Vector3T::Zero();
  Vector3T gyro_bias = Vector3T::Zero();
  Vector3T acc_bias = Vector3T::Zero();
};

struct TrackingMap {
  static const int N_MAXIMUM_CAMERAS{2};
  int n_cameras = 2;

  // Observations per camera per key frame formerly known as sofTracks_
  std::array<std::map<FrameId, std::vector<camera::Observation>>, N_MAXIMUM_CAMERAS> observations;

  // aka points aka landmarks
  std::map<TrackId, Track> tracks;

  // key frame poses
  std::map<FrameId, TrackingState> tracking_states;
};

void sparseBA(TrackingMap& map, int max_iter, bool do_winsorization);
void DoWinsorization(TrackingMap& map, size_t start_frame, size_t n_constained_frames,
                     const std::vector<FrameId>& in_keyframes);

}  // namespace cuvslam::pipelines
