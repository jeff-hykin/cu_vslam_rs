
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

namespace cuvslam::map {

Landmark::Landmark(TrackId id) : id_(id) {}

Landmark::Landmark(TrackId id, const Vector3T& pose_in_w) : id_(id), pose_in_w_(pose_in_w) {}

void Landmark::set_pose(const Vector3T& pose_in_w) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  pose_in_w_ = pose_in_w;
}

std::optional<Vector3T> Landmark::get_pose() const {
  std::lock_guard<std::mutex> lock(data_mutex_);
  return pose_in_w_;
}

TrackId Landmark::id() const { return id_; }

void Landmark::reset() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  pose_in_w_ = std::nullopt;
}

}  // namespace cuvslam::map
