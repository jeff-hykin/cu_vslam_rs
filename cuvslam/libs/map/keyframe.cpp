
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

#include "map/keyframe.h"

#include "common/log.h"

namespace cuvslam::map {

KeyFrame::KeyFrame(const State& state, int64_t time_ns) : state_(state), time_ns_(time_ns) {}

KeyFrame::KeyFrame(int64_t time_ns) : time_ns_(time_ns) {}

State KeyFrame::get_state() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

void KeyFrame::set_state(const State& state) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_ = state;
}

Isometry3T KeyFrame::get_pose() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.rig_from_w;
}

void KeyFrame::set_pose(const Isometry3T& pose) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.rig_from_w = pose;
}

Vector3T KeyFrame::get_velocity() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.velocity;
}

void KeyFrame::set_velocity(const Vector3T& velocity) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.velocity = velocity;
}

Vector3T KeyFrame::get_acc_bias() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.acc_bias;
}

void KeyFrame::set_acc_bias(const Vector3T& acc_bias) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.acc_bias = acc_bias;
}

Vector3T KeyFrame::get_gyro_bias() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_.gyro_bias;
}

void KeyFrame::set_gyro_bias(const Vector3T& gyro_bias) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_.gyro_bias = gyro_bias;
}

int64_t KeyFrame::time_ns() const { return time_ns_; }

}  // namespace cuvslam::map
