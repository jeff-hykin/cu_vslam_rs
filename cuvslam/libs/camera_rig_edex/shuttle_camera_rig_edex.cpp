
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

#include "camera_rig_edex/shuttle_camera_rig_edex.h"

#include "common/camera_id.h"

#include "camera_rig_edex/camera_rig_edex.h"

namespace cuvslam::camera_rig_edex {

bool ShuttleCameraRigEdex::check_end_of_sequence() {
  if (base_->isFinished()) {
    if (!base_->getReplayForward()) {
      current_loop_++;
    }
    base_->setReplayForward(!base_->getReplayForward());
  }
  if (current_loop_ >= num_loops_) {
    return false;
  }
  return true;
}

int64_t ShuttleCameraRigEdex::adjust_timestamp(int64_t timestamp) {
  int64_t duration = std::abs(base_->getLastTimestamp() - base_->getFirstTimestamp());
  int64_t max_ts = std::max(base_->getLastTimestamp(), base_->getFirstTimestamp());
  // TODO: use edex fps instead of hardcode to adjust time shift by 1 frame
  const int64_t delta = 33'333'333;

  // single loop is twice the duration, first forwards, then backwards
  return (base_->getReplayForward() ? timestamp : (2 * max_ts + delta - timestamp)) +
         2 * current_loop_ * (duration + delta);
}

ErrorCode ShuttleCameraRigEdex::getFrame(Sources& sources, Metas& metas, Sources& masks_sources,
                                         DepthSources& depth_sources) {
  if (!check_end_of_sequence()) {
    return ErrorCode::E_Bounds;
  }

  auto res = base_->getFrame(sources, metas, masks_sources, depth_sources);
  if (!res) {
    return res;
  }

  for (CameraId cam_id = 0; cam_id < metas.size(); ++cam_id) {
    if (sources[cam_id].data != nullptr) {
      metas[cam_id].timestamp = adjust_timestamp(metas[cam_id].timestamp);
    }
  }
  return ErrorCode::S_True;
}

void ShuttleCameraRigEdex::registerIMUCallback(const std::function<void(const imu::ImuMeasurement&)>&) {
  // Do nothing, IMU is not supported in shuttle replay
}

}  // namespace cuvslam::camera_rig_edex
