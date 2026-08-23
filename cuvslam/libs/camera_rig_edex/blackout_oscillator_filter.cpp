
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

#include "camera_rig_edex/blackout_oscillator_filter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "common/imu_measurement.h"
#include "common/log.h"

namespace cuvslam::camera_rig_edex {

BlackoutOscillatorFilter::BlackoutOscillatorFilter(std::unique_ptr<ICameraRig>&& base, int period, int duration)
    : base_(std::move(base)), period_(period), duration_(duration) {
  assert(period > duration);
}

ErrorCode BlackoutOscillatorFilter::start() { return base_->start(); }

ErrorCode BlackoutOscillatorFilter::stop() { return base_->stop(); }

const camera::ICameraModel& BlackoutOscillatorFilter::getIntrinsic(uint32_t index) const {
  return base_->getIntrinsic(index);
}

const Isometry3T& BlackoutOscillatorFilter::getExtrinsic(uint32_t index) const { return base_->getExtrinsic(index); }

ErrorCode BlackoutOscillatorFilter::getFrame(Sources& sources, Metas& metas, Sources& masks_sources,
                                             DepthSources& depth_sources) {
  ErrorCode status = base_->getFrame(sources, metas, masks_sources, depth_sources);

  assert(sources.size() == metas.size());

  if (status == ErrorCode::S_True && !metas.empty()) {
    auto first_source =
        std::find_if(sources.begin(), sources.end(), [](const ImageSource& source) { return source.data != nullptr; });
    if (first_source == sources.end()) {
      return status;
    }
    const CameraId first_cam = static_cast<CameraId>(std::distance(sources.begin(), first_source));
    int frame_id = static_cast<int>(metas[first_cam].frame_id);
    int index = frame_id % period_;
    if (index < duration_ && frame_id >= period_) {
      const auto image_size_bytes = [](const ImageSource& source, const ImageShape& shape) {
        switch (source.type) {
          case ImageSource::U8:
            return static_cast<size_t>(shape.width) * shape.height *
                   (source.image_encoding == ImageEncoding::RGB8 ? 3 : 1);
          case ImageSource::F32:
            return static_cast<size_t>(shape.width) * shape.height * sizeof(float);
          case ImageSource::U16:
            return static_cast<size_t>(shape.width) * shape.height * sizeof(uint16_t);
        }
        return size_t{0};
      };

      // do blackout
      for (CameraId cam_id = 0; cam_id < sources.size(); ++cam_id) {
        auto& source = sources[cam_id];
        if (source.data != nullptr) {
          std::memset(source.data, 0, image_size_bytes(source, metas[cam_id].shape));
        }
      }
      for (CameraId cam_id = 0; cam_id < depth_sources.size(); ++cam_id) {
        auto& source = depth_sources[cam_id];
        if (source.data != nullptr) {
          std::memset(source.data, 0, image_size_bytes(source, metas[cam_id].shape));
        }
      }
      log::Value<LogRoot>("blackout_oscilator", true);
    } else {
      log::Value<LogRoot>("blackout_oscilator", false);
    }
  }

  return status;
}

void BlackoutOscillatorFilter::registerIMUCallback(
    const std::function<void(const imu::ImuMeasurement& integrator)>& func) {
  base_->registerIMUCallback(func);
}

}  // namespace cuvslam::camera_rig_edex
