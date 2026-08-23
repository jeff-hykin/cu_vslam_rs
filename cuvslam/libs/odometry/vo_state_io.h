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

#include <memory>

#include "common/state_serial.h"
#include "map/map.h"
#include "sof/sof_multicamera_interface.h"

#include "odometry/ivisual_odometry.h"
#include "odometry/pose_prediction.h"

namespace cuvslam::odom {

// Shared serialization of the visual-odometry state common to the Multicamera and RGBD trackers
// (pose prediction history, previous pose, keyframe/landmark map, SOF tracker state, last frame
// stats). The solver-specific state is serialized by the caller before these run.

inline void SaveVoCommonState(serial::Writer& w, const PosePredictionModel& prediction_model,
                              const Isometry3T& prev_world_from_rig, const map::UnifiedMap& map,
                              const sof::IMultiSOF& feature_tracker,
                              const std::unique_ptr<IVisualOdometry::VOFrameStat>& last_frame_stat) {
  prediction_model.save_state(w);
  w.write_isometry(prev_world_from_rig);
  map.save_state(w);
  feature_tracker.save_state(w);

  const bool has_stat = last_frame_stat != nullptr;
  w.write_bool(has_stat);
  if (has_stat) {
    w.write_bool(last_frame_stat->keyframe);
    w.write_bool(last_frame_stat->heating);
    w.write_size(last_frame_stat->tracks2d.size());
    for (const Track2D& t : last_frame_stat->tracks2d) {
      w.write_pod(t.cam_id);
      w.write_pod(t.track_id);
      w.write_eigen(t.uv);
    }
    w.write_size(last_frame_stat->tracks3d.size());
    for (const auto& [track_id, point] : last_frame_stat->tracks3d) {
      w.write_pod(track_id);
      w.write_eigen(point);
    }
  }
}

inline void LoadVoCommonState(serial::Reader& r, PosePredictionModel& prediction_model, Isometry3T& prev_world_from_rig,
                              map::UnifiedMap& map, sof::IMultiSOF& feature_tracker,
                              std::unique_ptr<IVisualOdometry::VOFrameStat>& last_frame_stat) {
  prediction_model.load_state(r);
  r.read_isometry(prev_world_from_rig);
  map.load_state(r);
  feature_tracker.load_state(r);

  const bool has_stat = r.read_bool();
  last_frame_stat = has_stat ? std::make_unique<IVisualOdometry::VOFrameStat>() : nullptr;
  if (has_stat) {
    last_frame_stat->keyframe = r.read_bool();
    last_frame_stat->heating = r.read_bool();
    const size_t num_tracks2d = r.read_size();
    last_frame_stat->tracks2d.reserve(num_tracks2d);
    for (size_t i = 0; i < num_tracks2d; ++i) {
      Track2D t;
      t.cam_id = r.read_pod<CameraId>();
      t.track_id = r.read_pod<TrackId>();
      r.read_eigen(t.uv);
      last_frame_stat->tracks2d.push_back(t);
    }
    const size_t num_tracks3d = r.read_size();
    for (size_t i = 0; i < num_tracks3d; ++i) {
      const TrackId track_id = r.read_pod<TrackId>();
      Vector3T point;
      r.read_eigen(point);
      last_frame_stat->tracks3d.emplace(track_id, point);
    }
  }
}

}  // namespace cuvslam::odom
