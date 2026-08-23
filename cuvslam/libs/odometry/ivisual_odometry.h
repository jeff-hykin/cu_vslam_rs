
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

#include "common/isometry.h"
#include "common/state_serial.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/tracker_state_machine.h"
#include "sof/image_context.h"

#include "odometry/ipredictor.h"
#include "odometry/svo_config.h"

namespace cuvslam::odom {

class IVisualOdometry {
public:
  virtual ~IVisualOdometry() = default;

  // OUT: delta           - delta between frames in the world coordinate system
  //      static_info_exp - information matrix for the static pose in exponential mapping form
  //                        in the world coordinate system
  // per_frame_setting: per-frame settings bundle; use TrackPerFrameSettings{} for defaults.
  //                    Add new per-frame setting categories to TrackPerFrameSettings rather than extending this
  //                    signature.
  virtual bool track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
                     const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta,
                     Matrix6T& static_info_exp, const TrackPerFrameSettings& per_frame_setting) = 0;

  struct VOFrameStat {
    bool keyframe;
    bool heating;
    std::vector<Track2D> tracks2d;  // in pixels coordinates
    Tracks3DMap tracks3d;           // coordinates in tha camera spaces
  };

  virtual void enable_stat(bool enable) = 0;
  virtual const std::unique_ptr<VOFrameStat>& get_last_stat() const = 0;

  // Checkpoint support. save_state() quiesces asynchronous work (SBA) before serializing, hence
  // non-const. Default implementations throw for odometry modes without checkpoint support yet.
  virtual void save_state(serial::Writer& /*w*/) {
    throw std::runtime_error("cuVSLAM: state serialization is not implemented for this odometry mode");
  }
  virtual void load_state(serial::Reader& /*r*/) {
    throw std::runtime_error("cuVSLAM: state deserialization is not implemented for this odometry mode");
  }

  // Rebuild the pyramids of a restored previous-frame image context so the next track() call sees
  // the same state as before the checkpoint. Optional depth source is used by RGBD odometry.
  virtual void rebuild_prev_context(CameraId /*cam_id*/, const ImageSource& /*source*/,
                                    const ImageSource* /*depth_source*/, const ImageSource* /*mask_source*/,
                                    const sof::ImageContextPtr& /*ctx*/) {
    throw std::runtime_error("cuVSLAM: context rebuild is not implemented for this odometry mode");
  }
};

}  // namespace cuvslam::odom
