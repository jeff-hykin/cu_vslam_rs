
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
#include <deque>

#include "common/state_serial.h"
#include "odometry/ipredictor.h"

namespace cuvslam::odom {

/*

This class predicts motion trajectory given a few sample poses
from the past.

*/
class PosePredictionModel : public IPredictor {
public:
  // Predict relative motion from the last pose:
  // pose(timestamp) = update * latest_pose
  //
  // The method can optionally return latest pose.
  bool predict_left_update(Isometry3T& update, int64_t timestamp_ns, Isometry3T* latest_pose = nullptr) const;

  // Timestamp must be in nanoseconds
  void add_known_pose(const Isometry3T& pose, int64_t timestamp_ns);

  // Should be called whenever we reset coordinate system for the poses.
  void reset();

  int64_t last_timestamp_ns() const;

  bool predict(int64_t prev_timestamp, int64_t current_timestamp, Isometry3T& delta) const override final;

  void save_state(serial::Writer& w) const {
    w.write_tag(0x50505244);  // "PPRD"
    w.write_size(poses_.size());
    for (const Isometry3T& pose : poses_) {
      w.write_isometry(pose);
    }
    w.write_size(timestamps_ns_.size());
    for (const int64_t t : timestamps_ns_) {
      w.write_pod(t);
    }
  }

  void load_state(serial::Reader& r) {
    r.expect_tag(0x50505244, "PosePredictionModel");
    poses_.clear();
    const size_t num_poses = r.read_size();
    for (size_t i = 0; i < num_poses; ++i) {
      Isometry3T pose;
      r.read_isometry(pose);
      poses_.push_back(pose);
    }
    timestamps_ns_.clear();
    const size_t num_ts = r.read_size();
    for (size_t i = 0; i < num_ts; ++i) {
      timestamps_ns_.push_back(r.read_pod<int64_t>());
    }
  }

private:
  std::deque<Isometry3T> poses_;
  std::deque<int64_t> timestamps_ns_;
};

}  // namespace cuvslam::odom
