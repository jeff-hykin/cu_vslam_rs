
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

#include "camera/camera.h"

#include "common/camera_id.h"
#include "common/error.h"
#include "common/image.h"
#include "common/imu_measurement.h"
#include "common/isometry.h"

namespace cuvslam {

class ICameraRig {
public:
  virtual ~ICameraRig() {}

  virtual ErrorCode start() = 0;

  virtual ErrorCode stop() = 0;

  // Images are loaded as-is. Clients are responsible for transforming them
  // into log space, if needed.
  virtual ErrorCode getFrame(Sources& sources, Metas& metas, Sources& masks_sources, DepthSources& depth_sources) = 0;

  virtual uint32_t getCamerasNum() const = 0;
  virtual std::vector<CameraId> getCamerasWithDepth() const = 0;
  virtual const camera::ICameraModel& getIntrinsic(uint32_t index) const = 0;
  virtual const Isometry3T& getExtrinsic(uint32_t index) const = 0;

  virtual void registerIMUCallback(const std::function<void(const imu::ImuMeasurement&)>& func) = 0;
};
}  // namespace cuvslam
