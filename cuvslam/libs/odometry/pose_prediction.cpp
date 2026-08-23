
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

#include "odometry/pose_prediction.h"

#include "common/types.h"
#include "math/twist.h"

namespace {

// We are going to use just two latest poses.
constexpr size_t kMaxPoses = 2;

void Exp(cuvslam::Isometry3T& pose, const cuvslam::Vector6T& vw) {
  using namespace cuvslam;
  Isometry3T p;
  math::Exp(p, vw);
  pose = p;
}

void Log(cuvslam::Vector6T& vw, const cuvslam::Isometry3T& pose) {
  using namespace cuvslam;
  Isometry3T p = pose;
  math::Log(vw, p);
}

}  // namespace

namespace cuvslam::odom {

// Estimates velocity using two latest poses and use it to predict the next pose
bool PosePredictionModel::predict_left_update(Isometry3T& update, int64_t timestamp_ns, Isometry3T* latest_pose) const {
  assert(poses_.size() == timestamps_ns_.size());

  if (poses_.size() < 2) {
    return false;
  }

  assert(poses_.size() == 2);

  // No interpolation: we only want to predict.
  if (timestamp_ns < timestamps_ns_.back()) {
    assert(false);
    return false;
  }

  const int64_t previous_dt_ns = timestamps_ns_.back() - timestamps_ns_.front();
  const int64_t dt_ns = timestamp_ns - timestamps_ns_.back();
  const float factor = static_cast<float>(dt_ns) / static_cast<float>(previous_dt_ns);

  const Isometry3T previous_update = poses_.back() * poses_.front().inverse();
  Vector6T diff;
  ::Log(diff, previous_update);
  ::Exp(update, diff * factor);

  if (latest_pose) {
    *latest_pose = poses_.back();
  }

  return true;
}

void PosePredictionModel::add_known_pose(const Isometry3T& pose, int64_t timestamp_ns) {
  assert(poses_.size() == timestamps_ns_.size());
  if (poses_.size() + 1 > kMaxPoses) {
    poses_.pop_front();
    timestamps_ns_.pop_front();
  }

  if (!timestamps_ns_.empty() && timestamps_ns_.back() == timestamp_ns) {
    assert(false);
    return;
  }

  poses_.push_back(pose);
  timestamps_ns_.push_back(timestamp_ns);
}

void PosePredictionModel::reset() {
  poses_.clear();
  timestamps_ns_.clear();
}

int64_t PosePredictionModel::last_timestamp_ns() const {
  if (timestamps_ns_.empty()) {
    return 0;
  }
  return timestamps_ns_.back();
}

bool PosePredictionModel::predict(int64_t prev_timestamp, int64_t current_timestamp, Isometry3T& delta) const {
  if (poses_.size() < 2) {
    return false;
  }

  const int64_t previous_dt_ns = timestamps_ns_.back() - timestamps_ns_.front();
  const Isometry3T previous_update = poses_.back() * poses_.front().inverse();

  const int64_t request_dt_ns = current_timestamp - prev_timestamp;
  const float factor = static_cast<float>(request_dt_ns) / static_cast<float>(previous_dt_ns);

  Vector6T diff;
  Isometry3T update;
  ::Log(diff, previous_update);
  ::Exp(update, diff * factor);

  delta = update;
  return true;
}

}  // namespace cuvslam::odom
