
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

#include "common/camera_id.h"
#include "common/image.h"
#include "common/isometry.h"

#include "sof/image_context.h"
#include "sof/sof.h"
#include "sof/sof_config.h"

namespace cuvslam::odom {
struct KeyFrameSettings;
}

namespace cuvslam::sof {

struct ImageAndSource {
  ImageAndSource(const ImageSource& source_, ImageContextPtr image_) : source(source_), image(image_) {}

  const ImageSource& source;
  ImageContextPtr image;
};

struct MonoSOFFrameSettings {
  const Settings& sof;
  const odom::KeyFrameSettings& kf;
  // Multicamera uses MonoSOF for per-camera feature tracking; its keyframe decision is global.
  bool is_mono_mode = true;
};

enum FrameState { None, Key };

class IMonoSOF {
public:
  virtual ~IMonoSOF() = default;

  virtual void track(const ImageAndSource& curr_image, const ImageContextPtr& prev_image,
                     const Isometry3T& predicted_world_from_rig, const MonoSOFFrameSettings& frame_settings,
                     const ImageSource* mask_src = nullptr) = 0;

  virtual const TracksVector& finish(FrameState& state, const MonoSOFFrameSettings& frame_settings) = 0;

  virtual void reset() = 0;

  virtual CameraId camera_id() const = 0;

  // Checkpoint support: serialize/restore the persistent cross-frame tracking state
  // (alive tracks, last-keyframe tracks). Per-frame scratch is not serialized.
  virtual void save_state(serial::Writer& w) const = 0;
  virtual void load_state(serial::Reader& r) = 0;
};

}  // namespace cuvslam::sof
