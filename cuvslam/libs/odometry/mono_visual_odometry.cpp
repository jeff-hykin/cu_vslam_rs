
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

#include "odometry/mono_visual_odometry.h"

#include <algorithm>
#include <memory>

#include "pipelines/track_online_mono.h"
#include "sof/selector_mono.h"
#include "sof/sof_create.h"

#ifdef USE_CUDA
#include "odometry/svo_config.h"
#endif

namespace cuvslam::odom {

MonoVisualOdometry::MonoVisualOdometry(const camera::Rig& rig, const Settings& settings, bool use_gpu)
    : intrinsics_(*(rig.intrinsics[0])), settings_(settings), solver_(std::make_unique<pipelines::SolverSfMMono>(rig)) {
  settings_.sof_settings.ransac_filter = true;

  sof::Implementation implementation;
#ifdef USE_CUDA
  if (use_gpu) {
    implementation = sof::Implementation::kGPU;
  } else {
    implementation = sof::Implementation::kCPU;
  }
#else
  if (use_gpu) {
    TraceError("To use GPU SOF one must use USE_CUDA=true cmake option");
  }
  implementation = sof::Implementation::kCPU;
#endif

  auto selector = std::make_unique<sof::SelectorMono>(sof::SelectorMonoSettings(), true);
  feature_tracker_ =
      sof::CreateMonoSOF(implementation, 0, intrinsics_, std::move(selector), nullptr, settings_.sof_settings);
}

bool MonoVisualOdometry::track(const Sources& curr_sources, [[maybe_unused]] const DepthSources& depth_sources,
                               sof::Images& curr_images, const sof::Images& prev_images, const Sources& masks_sources,
                               Isometry3T& delta, Matrix6T& static_info_exp,
                               const TrackPerFrameSettings& per_frame_setting) {
  const sof::Settings& effective_sof = per_frame_setting.sof;
  assert(std::none_of(depth_sources.begin(), depth_sources.end(),
                      [](const ImageSource& source) { return source.data != nullptr; }));
  const CameraId camera_id = 0;
  const size_t camera_idx = static_cast<size_t>(camera_id);
  if (camera_idx >= curr_sources.size() || camera_idx >= curr_images.size() ||
      curr_sources[camera_idx].data == nullptr || curr_images[camera_idx] == nullptr) {
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track, mono image is not available");
    return false;
  }
  const ImageSource& left_curr_source = curr_sources[camera_idx];
  sof::ImageContextPtr left_curr_image = curr_images[camera_idx];
  sof::ImageContextPtr left_prev_image = nullptr;
  if (camera_idx < prev_images.size()) {
    left_prev_image = prev_images[camera_idx];
  }

  sof::FrameState frame_type;

  Isometry3T predicted_world_from_rig = prev_world_from_rig_;
  const int64_t timestamp = left_curr_image->get_image_meta().timestamp;  // current frame timestamp
  if (settings_.use_prediction) {
    do_predict(&prediction_model_, timestamp, predicted_world_from_rig);
  }
  const ImageSource* mask_src = (camera_idx < masks_sources.size() && masks_sources[camera_idx].data != nullptr)
                                    ? &masks_sources[camera_idx]
                                    : nullptr;

  const sof::MonoSOFFrameSettings sof_frame_settings{effective_sof, per_frame_setting.kf};
  feature_tracker_->track(sof::ImageAndSource(left_curr_source, left_curr_image), left_prev_image,
                          predicted_world_from_rig, sof_frame_settings, mask_src);
  const sof::TracksVector& tracks_vector = feature_tracker_->finish(frame_type, sof_frame_settings);
  tracks_vector.export_to_observations_vector(intrinsics_, observations_);

  IVisualOdometry::VOFrameStat* stat = last_frame_stat_.get();
  std::vector<Track2D>* tracks2d = stat ? &(stat->tracks2d) : nullptr;

  storage::Isometry3<float> world_from_rig;
  const ErrorCode err = solver_->monoSolveNextFrame(observations_, left_curr_image->get_image_meta().frame_id,
                                                    frame_type == sof::FrameState::Key, nullptr, world_from_rig,
                                                    tracks2d, tracks3d_, static_info_exp);
  if (stat) {
    stat->keyframe = frame_type == sof::FrameState::Key;
    stat->heating = !solver_->resectioningStarted();
    stat->tracks3d = tracks3d_;
  }

  if (err != ErrorCode::S_True) {
    return false;
  }
  prediction_model_.add_known_pose(world_from_rig, timestamp);
  delta = prev_world_from_rig_.inverse() * world_from_rig;
  prev_world_from_rig_ = world_from_rig;

  return true;
}

void MonoVisualOdometry::enable_stat(bool enable) {
  const bool current_state_is_enable = last_frame_stat_ != nullptr;
  if (current_state_is_enable == enable) {
    return;  // if nothing is changed do nothing
  }
  last_frame_stat_ = enable ? std::make_unique<VOFrameStat>() : nullptr;
}

const std::unique_ptr<IVisualOdometry::VOFrameStat>& MonoVisualOdometry::get_last_stat() const {
  return last_frame_stat_;
}

bool MonoVisualOdometry::do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) {
  Isometry3T update;
  const int64_t prev_abs_timestamp = prediction_model_.last_timestamp_ns();
  if (predictor->predict(prev_abs_timestamp, timestamp, update)) {
    sof_prediction = update * sof_prediction;
    return true;
  }
  return false;
}

}  // namespace cuvslam::odom
