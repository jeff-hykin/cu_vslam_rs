
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

#include "camera/camera.h"
#include "common/interfaces.h"
#include "edex/edex.h"
#include "edex/timeline.h"

#include "camera_rig_edex/camera_rig_edex.h"
namespace cuvslam::camera_rig_edex {

class ShuttleCameraRigEdex : public ICameraRigReplay {
public:
  ShuttleCameraRigEdex(std::unique_ptr<CameraRigEdex>&& base, size_t num_loops)
      : base_(std::move(base)), num_loops_(num_loops){};

  ErrorCode getFrame(Sources& sources, Metas& metas, Sources& masks_sources, DepthSources& depth_sources) override;
  void registerIMUCallback(const std::function<void(const imu::ImuMeasurement&)>& func) override;

  uint32_t getCamerasNum() const override { return base_->getCamerasNum(); };
  std::vector<CameraId> getCamerasWithDepth() const override { return base_->getCamerasWithDepth(); };
  const camera::ICameraModel& getIntrinsic(uint32_t index) const override { return base_->getIntrinsic(index); }
  const Isometry3T& getExtrinsic(uint32_t index) const override { return base_->getExtrinsic(index); }
  ErrorCode start() override { return base_->start(); }
  ErrorCode stop() override {
    current_loop_ = 0;
    return base_->stop();
  }

  const std::vector<CameraId>& getCameraIds() const override { return base_->getCameraIds(); }
  bool setCurrentFrame(int frame) override { return base_->setCurrentFrame(frame); }

private:
  std::unique_ptr<CameraRigEdex> base_;
  size_t num_loops_;
  size_t current_loop_ = 0;

  bool check_end_of_sequence();
  int64_t adjust_timestamp(int64_t timestamp);
};

}  // namespace cuvslam::camera_rig_edex
