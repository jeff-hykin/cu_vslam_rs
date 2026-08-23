
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

#include "odometry/rgbd_odometry.h"

#include "odometry/vo_state_io.h"

#include <algorithm>

#include "math/twist.h"
#include "pipelines/feature_predictor.h"
#include "sof/sof_create.h"

namespace cuvslam::odom {

RGBDOdometry::RGBDOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                           const Settings& settings, bool use_gpu)

    : rig_(rig),
      fig_(fig),
      settings_(settings),
      map_(20),
      feature_predictor_(std::make_shared<pipelines::FeaturePredictor>(map_, rig)),
      solver_(map_, rig, settings.sba_settings) {
  observations_.resize(rig.num_cameras);

  sof::Implementation implementation = sof::Implementation::kCPU;
  if (use_gpu) {
#ifdef USE_CUDA
    implementation = sof::Implementation::kGPU;
#else
    TraceError("To use GPU SOF one must use USE_CUDA=true cmake option");
#endif
  }
  feature_tracker_ =
      sof::CreateMultiSOF(implementation, rig, fig_, feature_predictor_, settings.sof_settings, settings.kf_settings);
}

void RGBDOdometry::reset() {
  prediction_model_.reset();  // don't do any prediction until two frames tracked successfully
  feature_tracker_->reset();
  solver_.reset();
  map_.clear();
}

bool RGBDOdometry::track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
                         const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta,
                         Matrix6T& static_info_exp, const TrackPerFrameSettings& per_frame_setting) {
  const auto first_image = std::find_if(curr_images.begin(), curr_images.end(),
                                        [](const sof::ImageContextPtr& image) { return image != nullptr; });
  if (first_image == curr_images.end()) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track, images are not available");
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("RGBDOdometry::track()", profiler_color_);
  const int64_t timestamp = (*first_image)->get_image_meta().timestamp;  // current frame timestamp
  Isometry3T predicted_world_from_rig = prev_world_from_rig_;

  if (settings_.use_prediction) {
    do_predict(&prediction_model_, timestamp, predicted_world_from_rig);
  }

  sof::FrameState frame_type;
  for (auto& cam_observations : observations_) {
    cam_observations.clear();
  }

  const bool track_result =
      feature_tracker_->trackNextFrame(curr_sources, curr_images, prev_images, masks_sources, predicted_world_from_rig,
                                       observations_, frame_type, per_frame_setting);
  if (!track_result) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the 2D tracking stage");
    return false;
  }

  IVisualOdometry::VOFrameStat* stat = last_frame_stat_.get();
  std::vector<Track2D>* tracks2d = stat ? &(stat->tracks2d) : nullptr;
  Tracks3DMap* tracks3d = stat ? &(stat->tracks3d) : nullptr;
  Isometry3T world_from_rig;

  bool depth_icp = false;
  CameraId camera_id_icp;

  for (CameraId cam_id = 0; cam_id < curr_images.size(); ++cam_id) {
    const auto& image = curr_images[cam_id];
    if (image == nullptr) {
      continue;
    }
    if (image->support_depth()) {
      camera_id_icp = cam_id;

      const auto& depthptr = depth_sources[cam_id];

      const ImageSource* mask_source = nullptr;
      if (masks_sources[cam_id].data != nullptr) {
        mask_source = &masks_sources[cam_id];
      }

      auto depth_pyramids = image->build_gpu_depth_pyramid(depthptr, stream.get_stream(), mask_source);

      if (depth_pyramids) {
        depth_icp = true;
      }
    }
  }

  cudaStreamSynchronize(stream.get_stream());

  pipelines::SFMInputs inputs{observations_, nullptr};

  bool have_pose;
  if (depth_icp) {
    pnp::IcpInfo depth_info{
        camera_id_icp,
        curr_images[camera_id_icp]->gpu_image_pyramid(),
        curr_images[camera_id_icp]->gpu_gradient_pyramid(),
        curr_images[camera_id_icp]->gpu_depth_pyramid()->get(),
    };

    inputs.depth_info = &depth_info;

    have_pose = solver_.solveNextFrame(
        timestamp, frame_type, inputs, world_from_rig, static_info_exp,
        {per_frame_setting.sba, per_frame_setting.sm, per_frame_setting.vo_pnp, per_frame_setting.inertial_stereo_pnp,
         per_frame_setting.imu_pnp, per_frame_setting.icp},
        tracks2d, tracks3d);
  } else {
    have_pose = solver_.solveNextFrame(
        timestamp, frame_type, inputs, world_from_rig, static_info_exp,
        {per_frame_setting.sba, per_frame_setting.sm, per_frame_setting.vo_pnp, per_frame_setting.inertial_stereo_pnp,
         per_frame_setting.imu_pnp, per_frame_setting.icp},
        tracks2d, tracks3d);
  }

  if (stat) {
    stat->keyframe = frame_type == sof::FrameState::Key;
    stat->heating = false;
  }

  if (!have_pose) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the PnP stage");
    return false;
  }

  prediction_model_.add_known_pose(world_from_rig, timestamp);
  delta = prev_world_from_rig_.inverse() * world_from_rig;
  prev_world_from_rig_ = world_from_rig;

  return true;
}

void RGBDOdometry::enable_stat(bool enable) {
  const bool current_state_is_enable = last_frame_stat_ != nullptr;
  if (current_state_is_enable == enable) {
    return;  // if nothing is changed do nothing
  }
  last_frame_stat_ = enable ? std::make_unique<IVisualOdometry::VOFrameStat>() : nullptr;
}

const std::unique_ptr<IVisualOdometry::VOFrameStat>& RGBDOdometry::get_last_stat() const { return last_frame_stat_; }

bool RGBDOdometry::do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) {
  Isometry3T update;
  const int64_t prev_abs_timestamp = prediction_model_.last_timestamp_ns();
  if (predictor->predict(prev_abs_timestamp, timestamp, update)) {
    sof_prediction = update * sof_prediction;
    return true;
  }
  return false;
}

void RGBDOdometry::save_state(serial::Writer& w) {
  w.write_tag(0x52474244);  // "RGBD"
  solver_.save_state(w);    // quiesces async SBA before the map is serialized
  SaveVoCommonState(w, prediction_model_, prev_world_from_rig_, map_, *feature_tracker_, last_frame_stat_);
}

void RGBDOdometry::load_state(serial::Reader& r) {
  r.expect_tag(0x52474244, "RGBDOdometry");
  solver_.load_state(r);
  LoadVoCommonState(r, prediction_model_, prev_world_from_rig_, map_, *feature_tracker_, last_frame_stat_);
}

void RGBDOdometry::rebuild_prev_context(CameraId cam_id, const ImageSource& source, const ImageSource* depth_source,
                                        const ImageSource* mask_source, const sof::ImageContextPtr& ctx) {
  feature_tracker_->rebuild_prev_context(cam_id, source, ctx);
  if (depth_source != nullptr && ctx->support_depth()) {
    // Same call track() makes for the depth-providing camera, including the optional mask.
    ctx->build_gpu_depth_pyramid(*depth_source, stream.get_stream(), mask_source);
    cudaStreamSynchronize(stream.get_stream());
  }
}

}  // namespace cuvslam::odom
