
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

#include "pipelines/track_online_mono.h"
#include "sof/kf_selector.h"
#include "sof/sof_mono_interface.h"

#include "odometry/ivisual_odometry.h"
#include "odometry/pose_prediction.h"
#include "odometry/svo_config.h"

namespace cuvslam::odom {

class MonoVisualOdometry : public IVisualOdometry {
public:
  MonoVisualOdometry(const camera::Rig& rig, const Settings& svo_settings, bool use_gpu);
  ~MonoVisualOdometry() override = default;

  bool track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
             const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta, Matrix6T& static_info_exp,
             const TrackPerFrameSettings& per_frame_setting) override;

  void enable_stat(bool enable) override;
  const std::unique_ptr<VOFrameStat>& get_last_stat() const override;

private:
  const camera::ICameraModel& intrinsics_;
  PosePredictionModel prediction_model_;
  Isometry3T prev_world_from_rig_ = Isometry3T::Identity();
  Settings settings_;
  std::unique_ptr<VOFrameStat> last_frame_stat_;
  std::unique_ptr<sof::IMonoSOF> feature_tracker_;
  std::unique_ptr<pipelines::SolverSfMMono> solver_;
  std::vector<camera::Observation> observations_;
  Tracks3DMap tracks3d_;
  bool do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction);
};

}  // namespace cuvslam::odom
