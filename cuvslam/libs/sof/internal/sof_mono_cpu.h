
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
#include "sof/internal/sof_mono_base.h"

#include <array>
#include <vector>

#include "camera/camera.h"
#include "camera/observation.h"
#include "common/image.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "sof/feature_prediction_interface.h"
#include "sof/feature_tracker.h"
#include "sof/image_pyramid_float.h"
#include "sof/image_pyramid_u8.h"
#include "sof/kf_selector.h"
#include "sof/selection.h"
#include "sof/selector_interface.h"
#include "sof/sof_config.h"

namespace cuvslam::sof {

class MonoSOFCPU : public MonoSOFBase {
public:
  MonoSOFCPU(CameraId cam_id, const camera::ICameraModel& intrinsics, std::unique_ptr<ISelector> selector,
             FeaturePredictorPtr feature_predictor, const Settings& sof_settings);

  void track(const ImageAndSource& curr_image, const ImageContextPtr& prev_image,
             const Isometry3T& predicted_world_from_rig, const MonoSOFFrameSettings& frame_settings,
             const ImageSource* mask_src = nullptr) final;

  const TracksVector& finish(FrameState& state, [[maybe_unused]] const MonoSOFFrameSettings& frame_settings) final;

  void reset() final;

  void save_state(serial::Writer& w) const final {
    save_base_state(w);
    last_keyframe_tracks_.save_state(w);
    w.write_pod<uint8_t>(static_cast<uint8_t>(last_frame_state_));
  }
  void load_state(serial::Reader& r) final {
    load_base_state(r);
    last_keyframe_tracks_.load_state(r);
    last_frame_state_ = static_cast<FrameState>(r.read_pod<uint8_t>());
  }

private:
  const camera::ICameraModel& intrinsics_;

  // internal state
  TracksVector last_keyframe_tracks_;  // last keyframe tracks
  FrameState last_frame_state_;

  std::unique_ptr<IFeatureTracker> feature_tracker_;
  GoodFeaturesToTrackDetector detector_;

  ImageMatrix<size_t> tracksMap_;  // for collapse
  std::vector<Vector2T> new_tracks_;

  void addFeatures(const ImageContextPtr& image, TracksVector& existing_tracks, std::vector<Vector2T>& new_tracks,
                   const Settings& sof_settings);
  void trackFeatures(const ImageContextPtr& curr_image, const ImageContextPtr& prev_image,
                     const Isometry3T& worldFromRig);
  void collapseTracks(const ImageContextPtr& image, TracksVector& tracks);
  static void ransacFilter(const camera::ICameraModel& intrinsics, const TracksVector& last_keyframe_tracks,
                           TracksVector& tracks);

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::sof
