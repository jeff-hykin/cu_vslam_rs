
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

#include <chrono>
#include <vector>

#include "common/error.h"
#include "common/frame_id.h"
#include "common/interfaces.h"
#include "common/isometry_utils.h"
#include "common/stopwatch.h"
#include "common/track_id.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "edex/edex_types.h"
#include "odometry/multi_visual_odometry_base.h"
#include "odometry/svo_config.h"
#include "slam/async_slam/async_slam.h"
#include "sof/image_context.h"
#include "sof/image_manager.h"

namespace cuvslam::launcher {

class BaseLauncher {
public:
  BaseLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings);
  virtual ~BaseLauncher() = default;

  void SetupSlam();
  ErrorCode launch();

  size_t nFrames() const;

  const CameraMap& cameraMap() const;

  const CameraMap& LoopClosureMap() const;

  void calcTotalOnlineResidual(float& totalOnlineResidual, size_t& totalTriangulatedTracks) const;

  // [OUT] averageOnlineResidual, maxFrameResidual, maxFrameResidualFrame - runtime statistics,
  // which calculates when new frame arrives.
  // Use calcBatchResidual method to get final (AKA batch) residual,
  // which is smaller, because we refine 3d points each frame.
  void calcOnlineResidual(float& averageOnlineResidual, float& maxFrameResidual, FrameId& maxFrameResidualFrame) const;

  float calcBatchResidual() const;

  // return average and max elapsed time for tracker.track() procedure
  void getTimers(float& average_track_sec, float& max_track_sec, FrameId& max_track_frame) const;

  void calcVisible2DTracksStats(float& averageVisible2DTracks, size_t& minVisible2DTracks,
                                FrameId& minVisible2DTracksFrame) const;

  // Save all data to precomputed edex file. (sequence field will be empty)
  // if filter2dTracks then keep only 2dtracks visible >=2 keyframe.
  bool saveResultToEdex(const std::string& outEdexName, const camera::ICameraModel& intrinsics, bool filter2dTracks,
                        edex::RotationStyle rs = edex::RotationStyle::RotationMatrix) const;
  [[maybe_unused]] bool saveResultToComposedRTJson(const std::string& outComposedRTJsonFileName) const;

  void registerTrackingLostCB(const std::function<void()>& cb);

  void updateGTPoses(const Isometry3TVector& poses);

  // Predicate that controls runtime depth ingestion: pool sizing and acquire_with_depth() path
  // in BaseLauncher::launch(). Default: false (no camera supplies depth). Subclasses that
  // care about depth (RGBD, Multisensor, multi-RGBD) override this to test membership in the
  // launcher's resolved depth-id list, so the runtime ingestion never drifts from the FIG.
  virtual bool isDepthCamera(CameraId /*id*/) const { return false; }

protected:
  virtual void SetupTracker(const odom::Settings& svo_settings, bool use_gpu) = 0;

  virtual bool launch_vo(Isometry3T& delta, Matrix6T& pose_info, const odom::TrackPerFrameSettings& per_frame) = 0;
  virtual const odom::IVisualOdometry::VOFrameStat& last_vo_stat() = 0;

  void SetPerFrameSettings(const odom::TrackPerFrameSettings& s) { per_frame_settings_ = s; }

  static void convertTracksContainer(const Tracks2DFrameMap& mapContainer, Tracks2DVectorsMap& vecPairContainer);

  // Keep only 2dtracks visible >=visibleInNKeyFrames keyframe.
  static void filter2DTracksByKeyFrameVisibility(const std::unordered_map<FrameId, std::vector<Track2D>>& tracks2D,
                                                 const FrameSet& keyFrames, size_t visibleInNKeyFrames,
                                                 Tracks2DVectorsMap& track2DFiltered);

  bool getTrackTriangulationInFrame(const FrameId& frameId, const TrackId& trackId, Vector3T& track3dPosition) const;

  // async slam
  std::unique_ptr<slam::AsyncSlam> async_slam_;

  struct FrameStat {
    float totalFrameResidual;  // using only triangulated tracker
    size_t nTriangulatedTracks;
  };
  ICameraRig& cameraRig_;
  odom::Settings svo_settings_;
  odom::TrackPerFrameSettings per_frame_settings_;
  std::chrono::time_point<std::chrono::steady_clock> last_frame_start_;

  camera::Rig rig;
  std::vector<CameraId> slam_cameras_;

  Sources curr_sources;
  DepthSources depth_sources;
  Sources masks_sources;
  Metas curr_meta;
  sof::Images curr_image_ptrs;
  sof::Images prev_image_ptrs;
  sof::Images slam_images;

  Stopwatch sw_track_proc_;
  double max_track_proc_duration_ = 0;
  FrameId max_track_proc_frame_ = 0;

  CameraMap cameraSolved_;
  CameraMap loop_closure_map_;
  std::unordered_map<FrameId, std::vector<Track2D>> tracks2D_;  // 2d track in normalized space
  std::map<FrameId, FrameStat> statistic_;
  Tracks3DMap finalTracks3d_;  // final 3d positions of all tracks in the world frame
  FrameSet finalKeyframes_;    // all keyframes if
  FrameSet failedFrames_;
  std::function<void()> tracking_lost_cb_;

  sof::ImageManager image_manager_;
  // set of triangulated 3d tracks per frame
  std::map<FrameId, std::vector<TrackId>> triangulatedTracksPerFrame_;

  // compute tracking quality stats
  void computeStatistics(const Isometry3T& cameraExtrinsics, const FrameId& frameId, const Tracks3DMap& tracks3d);

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0xFF00FF;

  Isometry3TVector gt_poses_;
};

}  // namespace cuvslam::launcher
