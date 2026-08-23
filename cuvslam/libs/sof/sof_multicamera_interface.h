
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
#include <vector>

#include "camera/observation.h"
#include "common/image.h"
#include "common/isometry.h"

#include "odometry/svo_config.h"
#include "sof/feature_prediction_interface.h"
#include "sof/image_manager.h"
#include "sof/kf_selector.h"
#include "sof/sof_config.h"
#include "sof/sof_mono_interface.h"

namespace cuvslam::sof {

// Camera-indexed vector sized to rig.num_cameras; an empty inner vector means no observations for that camera.
using MulticamObservations = std::vector<std::vector<camera::Observation>>;

class IMultiSOF {
public:
  virtual bool trackNextFrame(const Sources& curr_sources, Images& curr_images, const Images& prev_images,
                              const Sources& masks_sources, const Isometry3T& predicted_world_from_rig,
                              MulticamObservations& observations, FrameState& state,
                              const odom::TrackPerFrameSettings& per_frame) = 0;

  virtual void reset() = 0;

  virtual void reset_keyframe_selector() = 0;

  // Checkpoint support: serialize/restore persistent cross-frame tracking state (keyframe
  // selector, last-keyframe tracks, per-camera mono SOF state). Per-frame scratch is skipped.
  virtual void save_state(serial::Writer& w) const = 0;
  virtual void load_state(serial::Reader& r) = 0;

  // Rebuild the image/gradient pyramids of a restored previous-frame image context so the next
  // trackNextFrame() call sees the same GPU/CPU pyramid state as before the checkpoint.
  virtual void rebuild_prev_context(CameraId cam_id, const ImageSource& source, const ImageContextPtr& ctx) = 0;

  virtual ~IMultiSOF() = default;
};

}  // namespace cuvslam::sof
