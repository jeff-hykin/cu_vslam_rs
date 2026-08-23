
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
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <utility>

#include "common/isometry.h"

#include "slam/slam/slam.h"

namespace cuvslam::slam {

// Nodes that were sent to the slam stream but from which notifications
// about addition to the pos graph were not still received.
// Current pose schema: PG_1->...->PG_N->tail_1->...->tail_tip->+odometry_delta
//                                   |                   |             |
//                               slam tread          main thread   GetSlamPose
// All posses in world space
class Tail {
public:
  explicit Tail(uint64_t retention_time_ns = std::numeric_limits<uint64_t>::max());
  void Clear();

  // called from both threads
  std::optional<std::pair<int64_t, Isometry3T>> GetTip() const;
  [[nodiscard]] bool IsTimestampRetained(int64_t timestamp_ns) const;

  // timestamp_ns can't be in past
  [[nodiscard]] bool UpdateTimeByOdometry(int64_t timestamp_ns, const Isometry3T& pose);
  // called from slam thread when LC or LoadMap. timestamp_ns can be in past
  [[nodiscard]] bool UpdatePoseBySLAM(int64_t timestamp_ns, const Isometry3T& pose);

private:
  [[nodiscard]] int64_t OldestAllowedTimestamp() const;
  [[nodiscard]] bool TimestampIsRetained(int64_t timestamp_ns) const;
  void PruneOldTail();

  // first - timestamp ns
  // second - pose in world space
  std::list<std::pair<int64_t, Isometry3T>> tail_;  // updates from main and slam thread see @tail_guard_
  uint64_t retention_time_ns_;
  mutable std::mutex tail_guard_;
};

}  // namespace cuvslam::slam
