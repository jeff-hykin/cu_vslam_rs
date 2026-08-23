
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

#include <list>
#include <memory>
#include <mutex>
#include <optional>

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

namespace cuvslam::map {

class Landmark {
public:
  Landmark(TrackId id);
  Landmark(TrackId id, const Vector3T& pose_in_w);
  void set_pose(const Vector3T& pose_in_w);
  void reset();
  std::optional<Vector3T> get_pose() const;
  TrackId id() const;

private:
  const TrackId id_;
  std::optional<Vector3T> pose_in_w_ = std::nullopt;
  mutable std::mutex data_mutex_;
};

struct State {
  Isometry3T rig_from_w = Isometry3T::Identity();

  Vector3T velocity = Vector3T::Zero();
  Vector3T acc_bias = Vector3T::Zero();
  Vector3T gyro_bias = Vector3T::Zero();
};

class KeyFrame {
public:
  KeyFrame(const State& state, int64_t time_ns);
  KeyFrame(int64_t time_ns);

  State get_state() const;
  void set_state(const State& state);

  Isometry3T get_pose() const;
  void set_pose(const Isometry3T& rig_from_w);

  Vector3T get_velocity() const;
  void set_velocity(const Vector3T& velocity);

  Vector3T get_acc_bias() const;
  void set_acc_bias(const Vector3T& acc_bias);

  Vector3T get_gyro_bias() const;
  void set_gyro_bias(const Vector3T& gyro_bias);

  int64_t time_ns() const;

private:
  State state_;
  mutable std::mutex state_mutex_;
  const int64_t time_ns_;
};

}  // namespace cuvslam::map
