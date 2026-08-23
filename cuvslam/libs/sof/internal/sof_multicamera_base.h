
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

#include "sof/sof_multicamera_interface.h"

#include <functional>
#include <list>
#include <vector>

#include "common/camera_id.h"
#include "odometry/svo_config.h"
#include "profiler/profiler.h"

#include "sof/kf_selector.h"
#include "sof/sof.h"

namespace cuvslam::sof {

class MultiSOFBase : public IMultiSOF {
protected:
public:
  MultiSOFBase(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid, const Settings& sof_settings,
               const odom::KeyFrameSettings& keyframe_settings);

  bool trackNextFrame(const Sources& curr_sources, Images& curr_images, const Images& prev_images,
                      const Sources& masks_sources, const Isometry3T& predicted_world_from_rig,
                      MulticamObservations& observations, FrameState& state,
                      const odom::TrackPerFrameSettings& per_frame) final;

  void reset_keyframe_selector() override;

  void save_state(serial::Writer& w) const override;
  void load_state(serial::Reader& r) override;

protected:
  virtual void LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id, const Sources& curr_sources,
                                                Images& curr_images,
                                                const std::vector<camera::Observation>& primary_obs,
                                                std::vector<camera::Observation>* secondary_obs = nullptr) = 0;
  virtual void GetTrackingResults(MulticamObservations& observations) = 0;
  virtual void StartKeyframe() = 0;

  struct TracksVectorAndCam {
    CameraId cam_id;
    std::reference_wrapper<const TracksVector> ref;
  };
  using MulticamTracksVector = std::vector<TracksVectorAndCam>;

  bool is_primary_cam(CameraId cam_id) const;

  bool is_keyframe(const MulticamTracksVector& tracks, const int64_t current_timestamp_ns,
                   const odom::KeyFrameSettings& kf_settings);

  camera::Rig rig_;
  camera::FrustumIntersectionGraph fid_;
  bool box_prefilter_ = false;
  std::list<std::unique_ptr<IMonoSOF>> mono_sof_;
  KFSelector kf_selector_;
  std::unordered_map<CameraId, TracksVector> last_kf_tracks_;
  int64_t last_kf_timestamp_ = 0;

  // keep allocated memory for is_keyframe
  TracksVector all_tracks_vec_;
  TracksVector last_kf_tracks_vec_;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("MultiSOF");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::sof
