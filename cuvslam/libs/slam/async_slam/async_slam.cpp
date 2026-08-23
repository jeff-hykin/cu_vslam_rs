
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

#include "slam/async_slam/async_slam.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>

#include "common/coordinate_system.h"
#include "common/log_types.h"
#include "common/rerun.h"
#include "common/thread_safe_queue.h"
#include "common/unaligned_types.h"
#include "cuvslam/internal.h"
#include "pipelines/visualizer.h"
#include "profiler/profiler.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/slam/slam.h"
#include "slam/view/map_to_view.h"
#include "slam/view/view_landmarks.h"
#include "sof/image_manager.h"
#include "visualizer/visualizer.hpp"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define CALLBACK_AND_RETURN_IF(condition, callback, type, message) \
  if (condition) {                                                 \
    callback(Result<type>::Error(message));                        \
    return;                                                        \
  }

namespace {
using namespace cuvslam;

#ifdef USE_RERUN
void LogSlamPose(const Isometry3T& slam_pose) {
  const Vector3T& p = slam_pose.translation();
  thread_local std::vector<rerun::Position3D> poses(1);
  poses.clear();
  poses.emplace_back(p.x(), p.y(), p.z());
  visualizer::RerunVisualizer::getInstance().getRecordingStream().log(
      "world/slam_pose", rerun::Points3D(poses).with_colors(Color(0, 255, 0)).with_radii(5.f));
}
#endif

}  // namespace

namespace cuvslam::slam {

AsyncSlam::AsyncSlam(const camera::Rig& rig, const std::vector<CameraId>& cameras, const AsyncSlamOptions& options)
    : rig_(rig),
      cameras_(cameras),
      slam_(std::make_unique<LocalizerAndMapper>(rig, FeatureDescriptorType::kShiTomasi6, options.use_gpu)),
      tail_(options.retention_time_ms * 1'000'000ULL) {
  reproduce_mode_ = options.reproduce_mode;

  // no second thread here - safe access to all data
  options_ = options;
  slam_->SetReproduceMode(reproduce_mode_);
  slam_->SetLandmarksSpatialIndex(options.spatial_index_options);
  const bool randomized = !reproduce_mode_;
  loop_closure_solver_.reset(
      CreateLoopClosureSolver(options.loop_closure_solver_type, RansacType::kPnP, randomized, rig_));
  slam_->SetPoseGraphOptimizerOptions(options.pgo_options);
  slam_->SetActiveCameras(cameras_);
  throttling_time_ns_ = options.throttling_time_ms * 1'000'000;

  if (options.max_pose_graph_nodes) {
    slam_->SetKeyframesLimit(options.max_pose_graph_nodes);
  }
  if (options.pose_for_frame_required) {
    slam_->SetKeepTrackPoses(true);
  }

  tail_.Clear();

  if (!options.map_cache_path.empty()) {
    if (!slam_->AttachToNewDatabase(options.map_cache_path)) {
      throw std::runtime_error("Failed to connect SLAM to map at " + options.map_cache_path);
    }
  }

  if (!reproduce_mode_) {
    SlamStdout("Starting background thread");
    thread_ = std::thread{&AsyncSlam::Run, this};
  } else {
    SlamStdout("Do not start background thread because reproduce_mode is set.");
  }
}
AsyncSlam::~AsyncSlam() {
  Shutdown();
  SlamStdout("Destroyed AsyncSlam instance. ");
}

void AsyncSlam::TrackResult(FrameId frameId, int64_t timestamp_ns, const odom::IVisualOdometry::VOFrameStat& stat,
                            const sof::Images& images, const Isometry3T& delta) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AsyncSlam::TrackResult()", profiler_color_);

  assert(track_data_.from_keyframe.matrix().allFinite());

  // extract telemetry
  {
    std::shared_ptr<AsyncSlamLCTelemetry> async_slam_telemetry;
    while (telemetry_queue_.TryPop(async_slam_telemetry)) {
      if (async_slam_telemetry) {
        last_telemetry_ = *async_slam_telemetry;
      }
    }
  }

  bool is_keyframe = stat.keyframe;

  // first frame
  if (is_first_frame_) {
    VO_ResetFrameData(frameId, timestamp_ns, track_data_);
    tail_.Clear();
    is_first_frame_ = false;
  }

  VO_IncrementFrameData(frameId, timestamp_ns, delta, track_data_);

  const bool has_images = std::any_of(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (is_keyframe && has_images) {
    const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
    const Isometry3T current_pose = GetSlamPose();

    vo_keyframe->vo_pose_at_this_frame = current_pose;
    vo_keyframe->track_data = track_data_;
    vo_keyframe->frame_data.frame_id = frameId;
    vo_keyframe->frame_data.timestamp_ns = timestamp_ns;
    vo_keyframe->frame_data.frame_information = FrameInformationString(images);

    {
      std::lock_guard image_guard(processing_images_mutex_);
      processing_images_ = images;
    }
    {
      vo_keyframe->frame_data.tracks2d_norm.reserve(stat.tracks2d.size());

      std::unordered_set<TrackId> added_tracks;
      for (const auto& x : stat.tracks2d) {
        const Vector2T& uv = x.uv;
        Vector2T uv_norm;

        const TrackId& track_id = x.track_id;
        if (stat.tracks3d.find(track_id) == stat.tracks3d.end()) {
          continue;  // remove landmarks without 3d
        }
        const Vector3T& v = stat.tracks3d.at(track_id);
        if (v.norm() > options_.max_landmarks_distance) {
          continue;  // remove landmarks outside the max distance
        }

        if (x.cam_id >= images.size() || images[x.cam_id] == nullptr) {
          continue;
        }
        if (x.cam_id >= rig_.num_cameras) {
          SlamStdout("Found track with invalid camera id");
          continue;  // skip invalid camera ID
        }
        const auto& intrinsics = rig_.intrinsics[x.cam_id];
        intrinsics->normalizePoint(uv, uv_norm);
        vo_keyframe->frame_data.tracks2d_norm.emplace_back(VOFrameData::Track2DXY{x.cam_id, x.track_id, uv_norm});
        added_tracks.insert(x.track_id);
      }
      // xyz to camera space
      for (const auto& [track_id, xyz_rel] : stat.tracks3d) {
        if (added_tracks.find(track_id) == added_tracks.end()) {
          continue;
        }
        vo_keyframe->frame_data.tracks3d_rel[track_id] = xyz_rel;
      }
    }
    if (tail_.UpdateTimeByOdometry(timestamp_ns, current_pose)) {
      input_queue_.Push(vo_keyframe);
    }
    VO_ResetFrameData(frameId, timestamp_ns, track_data_);
  }
  if (options_.pose_for_frame_required) {
    trajectory_[frameId] = track_data_;
  }

#ifdef USE_RERUN
  {
    const Isometry3T current_pose = GetSlamPose();
    LogSlamPose(current_pose);
    RERUN(pipelines::logTrajectory, current_pose.inverse(), "world/trajectories/slam", Color(255, 255, 255),
          TrajectoryType::GT);
  }
#endif
}

Isometry3T AsyncSlam::GetSlamPose() const {
  std::lock_guard slam_guard(slam_mutex_);
  const auto may_be_tip = tail_.GetTip();
  if (!may_be_tip) {
    return Isometry3T::Identity();
  }
  const Isometry3T& tail_tip = may_be_tip->second;
  Isometry3T slam_pose = tail_tip * track_data_.from_keyframe;

  if (options_.planar_constraints) {
    slam_pose.translation().y() = 0;
  }
  return slam_pose;
}

void AsyncSlam::ProcessInputSynchronously() {
  if (reproduce_mode_) {
    ProcessInput();
  }
}

// stop thread
void AsyncSlam::Stop() {
  Shutdown();
  reproduce_mode_ = true;
}

bool AsyncSlam::GetPoseForFrame(FrameId frameId, Isometry3T& pose) const {
  pose.setIdentity();

  if (!options_.pose_for_frame_required) {
    return false;
  }

  auto it = trajectory_.find(frameId);
  if (it == trajectory_.end()) {
    SlamStderr("Pose not found for frame %zd.\n", static_cast<uint64_t>(frameId));
    return false;
  }
  auto& track_data = it->second;
  assert(track_data.from_keyframe.matrix().allFinite());

  Isometry3T m;
  std::lock_guard slam_guard(slam_mutex_);
  if (slam_->CalcFramePose(track_data.end_frame_id, m)) {
    m = m * track_data.from_keyframe;
    pose = m;
    return true;
  }

  return false;
}

bool AsyncSlam::GetPosesForAllFrames(std::map<uint64_t, storage::Isometry3<float>>& frames) const {
  if (!options_.pose_for_frame_required) {
    return false;
  }

  std::lock_guard slam_guard(slam_mutex_);
  for (auto it : trajectory_) {
    auto& track_data = it.second;
    uint64_t timestamp_ns = it.second.timestamp_ns;
    assert(track_data.from_keyframe.matrix().allFinite());

    Isometry3T m;
    if (slam_->CalcFramePose(track_data.end_frame_id, m)) {
      m = m * track_data.from_keyframe;
      Isometry3T pose = m;
      frames[timestamp_ns] = pose;
    }
  }
  return true;
}

bool AsyncSlam::GetLastTelemetry(AsyncSlamLCTelemetry& telemetry) const {
  telemetry = last_telemetry_;
  return true;
}

const std::list<LoopClosureStamped>& AsyncSlam::GetLastLoopClosuresStamped() { return last_loop_closures_stamped_; }

void AsyncSlam::CopyToDatabase(const std::string& path, const std::function<void(bool)>& callback) {
  const auto copy_to_database_cmd = std::make_shared<CopyToDatabaseCmd>(path);
  assert(!copy_to_database_callback_);
  copy_to_database_callback_ = callback;
  TraceDebug("AttachToNewDatabaseSaveMapAndDetach reproduce_mode_=%d", reproduce_mode_ ? 1 : 0);
  if (reproduce_mode_) {
    // sync
    constexpr FrameId end_frame_id = ~0UL;
    const Isometry3T vo_pose_at_that_frame = Isometry3T::Identity();
    copy_to_database_cmd->Execute(*this, end_frame_id, vo_pose_at_that_frame);
  } else {
    // async
    const auto vo_keyframe = std::make_shared<VOKeyframeInfo>(VOKeyframeInfo());
    vo_keyframe->command = copy_to_database_cmd;
    input_queue_.Push(vo_keyframe);
  }
}

bool AsyncSlam::CopyToDatabase_internal(const std::string& path) {
  const bool status = slam_->AttachToNewDatabaseSaveMapAndDetach(path);
  if (copy_to_database_callback_) {
    copy_to_database_callback_(status);
  }
  copy_to_database_callback_ = nullptr;
  return status;
}

// Set landmarks view
void AsyncSlam::SetLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view) { this->landmarks_view_ = view; }

// Set loop closure view
void AsyncSlam::SetLoopClosureView(std::shared_ptr<ViewManager<ViewLandmarks>> view) { this->loop_close_view_ = view; }

// Set pose graph view
void AsyncSlam::SetPoseGraphView(std::shared_ptr<ViewManager<ViewPoseGraph>> view) { this->pose_graph_view_ = view; }

void AsyncSlam::Run() {
#ifdef USE_CUDA
  if (options_.use_gpu) {
    cudaSetDevice(0);
  }
#endif
  for (;;) {
    // wait for news in input_queue
    if (!input_queue_.Wait()) {
      // aborted
      return;
    }
    ProcessInput();
  }
}

void AsyncSlam::ProcessInput() {
  FrameId end_frame_id = InvalidFrameId;
  Isometry3T vo_pose_at_that_frame = Isometry3T::Identity();

  Isometry3T world_from_rig_guess;
  FrameId frame_id = InvalidFrameId;
  uint64_t timestamp_ns = 0;
  bool has_input = false;

  Images current_images;

  for (;;) {
    TRACE_EVENT ev = profiler_domain_.trace_event("process vo data", profiler_color_);

    // extract all from input_queue
    std::shared_ptr<VOKeyframeInfo> vo_kf;
    {
      TRACE_EVENT ev_input = profiler_domain_.trace_event("input", profiler_color_);
      if (!input_queue_.TryPop(vo_kf)) {
        break;  // input_queue_ is empty
      }
    }

    const auto& frame_data = vo_kf->frame_data;
    RERUN(visualizer::RerunVisualizer::getInstance().setupTimeline, frame_data.frame_id, frame_data.timestamp_ns);

    // execute command from input_queue_
    if (vo_kf->command) {
      std::shared_ptr<ICommand>& cmd = vo_kf->command;
      cmd->Execute(*this, end_frame_id, vo_pose_at_that_frame);
      continue;
    }

    {
      std::lock_guard image_guard(processing_images_mutex_);
      current_images = processing_images_;
    }
    const auto current_image =
        std::find_if(current_images.begin(), current_images.end(), [](const auto& image) { return image != nullptr; });
    const bool is_valid_image =
        current_image != current_images.end() && ((*current_image)->get_image_meta().frame_id == frame_data.frame_id);
    Isometry3T pose_estimate_slam;
    {
      const VOTrackData& track_data = vo_kf->track_data;

      Isometry3T last_keyframe_pose;
      int64_t last_keyframe_ts;
      if (slam_->GetLastKeyframePoseAndTimestamp(last_keyframe_pose, last_keyframe_ts) && last_keyframe_ts > 0 &&
          track_data.timestamp_ns < static_cast<uint64_t>(last_keyframe_ts)) {
        continue;  // no need to add keyframe in past if slam in a future
      }
      std::lock_guard slam_guard(slam_mutex_);
      slam_->AddKeyframe(track_data.from_keyframe, frame_data, is_valid_image ? current_images : Images());
      pose_estimate_slam = slam_->GetCurrentPose();
    }
    SlamStdout("'");

    {
      TRACE_EVENT ev_cd = profiler_domain_.trace_event("copy data", profiler_color_);

      frame_id = frame_data.frame_id;
      timestamp_ns = frame_data.timestamp_ns;
      end_frame_id = vo_kf->track_data.end_frame_id;
      vo_pose_at_that_frame = vo_kf->vo_pose_at_this_frame;
      world_from_rig_guess = pose_estimate_slam;
    }

    vo_kf.reset();
    has_input = true;
  }
  {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->ReduceKeyframes();
  }
  if (!has_input) {
    return;
  }
  {
    std::lock_guard image_guard(processing_images_mutex_);
    current_images = processing_images_;
  }
  const auto current_image =
      std::find_if(current_images.begin(), current_images.end(), [](const auto& image) { return image != nullptr; });
  bool is_valid_image =
      current_image != current_images.end() && ((*current_image)->get_image_meta().frame_id == frame_id);

  if (is_valid_image) {
    // init last_step_telemetry_
    AsyncSlamLCTelemetry last_step_telemetry;
    last_step_telemetry.timestamp_ns = timestamp_ns;

    TRACE_EVENT ev = profiler_domain_.trace_event("LC & optimization", profiler_color_);

    bool skip_loop_closure = false;
    if (!last_loop_closures_stamped_.empty()) {
      // no need for LC if previous LC was recent
      if ((timestamp_ns - last_loop_closures_stamped_.back().timestamp_ns) < throttling_time_ns_) {
        skip_loop_closure = true;  // loop closure not needed
      }
    }

    //--- Loop Closure detection ---
    if (loop_closure_solver_ != nullptr && !skip_loop_closure) {
      LocalizerAndMapper::LoopClosureStatus lc_status;
      bool lc_found = false;
      {
        std::lock_guard slam_guard(slam_mutex_);
        slam_->DetectLoopClosure(*loop_closure_solver_, current_images, world_from_rig_guess, lc_status);
        lc_found = lc_status.success;
      }

      // view loop closure
      std::shared_ptr<ViewLandmarks> lc_view = loop_close_view_ ? loop_close_view_->acquire_earliest() : nullptr;
      if (lc_view) {
        std::lock_guard slam_guard(slam_mutex_);
        PublishLoopClosureToView(slam_->GetMap(), lc_status.landmarks, *lc_view);
        lc_view->timestamp_ns = timestamp_ns;
        lc_view.reset();
      }

      // Update Landmark Statistic in spatial index
      {
        std::lock_guard slam_guard(slam_mutex_);
        slam_->UpdateLandmarkProbeStatistics(lc_status.discarded_landmarks);
      }
      // Add LC edge and Add Landmark Relation to spatial index and pose graph
      if (lc_found) {
        std::lock_guard slam_guard(slam_mutex_);
        if (slam_->ApplyLoopClosureResult(lc_status.result_pose, lc_status.result_pose_covariance,
                                          lc_status.landmarks)) {
          const LoopClosureStamped lc_pose_stamped = {frame_id, timestamp_ns, lc_status.result_pose};
          last_loop_closures_stamped_.push_back(lc_pose_stamped);
          while (last_loop_closures_stamped_.size() > max_num_last_lcs_) {
            last_loop_closures_stamped_.pop_front();
          }
        } else {
          SlamStdout("Can't apply loop closure result");
        }
      }

      last_step_telemetry.lc_status = lc_found;
      last_step_telemetry.lc_selected_landmarks_count = lc_status.selected_landmarks_count;
      last_step_telemetry.lc_tracked_landmarks_count = lc_status.tracked_landmarks_count;
      last_step_telemetry.lc_pnp_landmarks_count = lc_status.pnp_landmarks_count;
      last_step_telemetry.lc_good_landmarks_count = lc_status.good_landmarks_count;
      if (lc_found) {
        SlamStdout("S");  // Successful LC
      }
      // Slam Pose Graph Optimization
      bool optimization_happens = false;
      if (lc_found || options_.planar_constraints) {
        // TODO: ? optimize_options.keyframes_in_sight = loop_closure_status.keyframes_in_sight;
        std::lock_guard slam_guard(slam_mutex_);

        optimization_happens = slam_->OptimizePoseGraph(options_.planar_constraints);
      }
      if (optimization_happens) {
        const Isometry3T pose_estimate_slam = slam_->GetCurrentPose();
        TRACE_EVENT ev1 = profiler_domain_.trace_event("post optimization", profiler_color_);
        log::Value<LogFrames>("pose_slam", pose_estimate_slam);
        last_step_telemetry.pgo_status = true;

        int64_t last_keyframe_ts;
        Isometry3T last_keyframe_pose;
        if (slam_->GetLastKeyframePoseAndTimestamp(last_keyframe_pose, last_keyframe_ts)) {
          if (!tail_.UpdatePoseBySLAM(last_keyframe_ts, last_keyframe_pose)) {
            TraceWarning(
                "Failed to update SLAM tail after pose graph optimization: keyframe timestamp is outside retention.");
          }
        }

        SlamStdout(":");
      }
      telemetry_queue_.Push(std::make_shared<AsyncSlamLCTelemetry>(last_step_telemetry));
    }
  }

  // view landmarks
  std::shared_ptr<ViewLandmarks> landmarks_view = landmarks_view_ ? landmarks_view_->acquire_earliest() : nullptr;
  if (landmarks_view) {
    std::lock_guard slam_guard(slam_mutex_);
    landmarks_view->landmarks.clear();
    PublishAllLandmarksToView(slam_->GetMap(), timestamp_ns, *landmarks_view);
    landmarks_view.reset();
  }

  // view pose graph
  std::shared_ptr<ViewPoseGraph> pose_graph_view = pose_graph_view_ ? pose_graph_view_->acquire_earliest() : nullptr;
  if (pose_graph_view) {
    std::lock_guard slam_guard(slam_mutex_);
    PublishPoseGraphToView(slam_->GetMap(), timestamp_ns, *pose_graph_view);
    pose_graph_view.reset();
  }

  if (input_queue_.IsEmpty()) {
    std::lock_guard slam_guard(slam_mutex_);
    slam_->FlushActiveDatabase();
  }
}

void AsyncSlam::Shutdown() {
  input_queue_.Abort();
  telemetry_queue_.Abort();

  if (thread_.joinable()) {
    thread_.join();
  }
  // Copy to database
  std::lock_guard slam_guard(slam_mutex_);
  slam_->DetachDatabase();
}

std::string AsyncSlam::FrameInformationString(const sof::Images& images) {
#ifdef CUVSLAM_LOG_ENABLE
  Json::Value json;

  const auto first_image =
      std::find_if(images.begin(), images.end(), [](const auto& image) { return image != nullptr; });
  if (first_image != images.end()) {
    auto& meta_0 = (*first_image)->get_image_meta();
    json["frame_id"] = static_cast<Json::UInt64>(meta_0.frame_id);
    json["timestamp"] = static_cast<Json::UInt64>(meta_0.timestamp);
    json["frame_number"] = meta_0.frame_number;
    for (CameraId cam_id = 0; cam_id < images.size(); ++cam_id) {
      const auto& img = images[cam_id];
      if (img != nullptr) {
        json["image_file" + std::to_string(cam_id)] = img->get_image_meta().filename;
      }
    }
  }

  std::string jsonStr = writeString(Json::StreamWriterBuilder(), json);
  return jsonStr;
#else
  return "";
#endif
}

// reset VOFrameData (if data was post to ProcessVOFrameData)
void AsyncSlam::VO_ResetFrameData(FrameId frame_id, uint64_t timestamp_ns, VOTrackData& data) {
  data.end_frame_id = frame_id;
  data.timestamp_ns = timestamp_ns;
  data.from_keyframe = Isometry3T::Identity();
}

void AsyncSlam::VO_IncrementFrameData(FrameId frame_id, uint64_t timestamp_ns, const Isometry3T& pose_estimate_rel,
                                      VOTrackData& data) {
  data.end_frame_id = frame_id;
  data.timestamp_ns = timestamp_ns;
  data.from_keyframe = data.from_keyframe * pose_estimate_rel;
}

}  // namespace cuvslam::slam
