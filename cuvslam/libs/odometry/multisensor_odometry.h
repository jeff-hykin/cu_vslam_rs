
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

#include <optional>
#include <unordered_map>
#include <vector>

#include "common/imu_measurement.h"
#include "common/isometry.h"
#include "common/vector_3t.h"
#include "pipelines/track_online_multisensor.h"

#include "odometry/ipredictor.h"
#include "odometry/ivisual_odometry.h"
#include "odometry/pose_prediction.h"
#include "odometry/svo_config.h"

namespace cuvslam::odom {

// Multisensor odometry using the cuNLS-based MultisensorPoseEstimator solver.
// The available sensor set (IMU presence, depth presence) is selected at construction
// time via Settings::multisensor_settings. When `with_imu=true` the inertial path is
// enabled (gravity estimation, inertial SBA, IMU-seeded pose prediction) — identical
// flow to SolverSfMInertial but with MultisensorPoseEstimator in place of InertialPnP.
class MultisensorOdometry : public IVisualOdometry {
public:
  MultisensorOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig, const Settings& settings,
                      bool use_gpu);

  bool has_imu() const { return settings_.multisensor_settings.with_imu; }

  bool track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
             const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta, Matrix6T& static_info_exp,
             const TrackPerFrameSettings& per_frame_setting) override;

  void enable_stat(bool enable) override;
  const std::unique_ptr<IVisualOdometry::VOFrameStat>& get_last_stat() const override;

  // IMU integration (no-op when constructed with with_imu=false).
  void add_imu_measurement(const cuvslam::imu::ImuMeasurement& m);
  std::optional<Vector3T> get_gravity() const;
  std::optional<pipelines::SolverSfMMultisensor::ImuState> GetImuState() const;

protected:
  bool do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction);
  void reset();

  camera::FrustumIntersectionGraph fig_;
  Settings settings_;
  PosePredictionModel prediction_model_;
  map::UnifiedMap map_;
  sof::FeaturePredictorPtr feature_predictor_;

private:
  pipelines::SolverSfMMultisensor solver_;
  Isometry3T prev_world_from_rig_ = Isometry3T::Identity();

  std::unique_ptr<IVisualOdometry::VOFrameStat> last_frame_stat_;
  std::unique_ptr<sof::IMultiSOF> feature_tracker_;
  pipelines::MulticamObservations observations_;
  std::vector<CameraId> depth_cam_ids_;
  std::vector<pnp::RGBDInfo> depth_info_storage_;
  std::vector<const pnp::RGBDInfo*> depth_infos_;
  cuda::Stream stream;

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("Multisensor VO");
  const uint32_t profiler_color_ = 0xFFFF00;
};

}  // namespace cuvslam::odom
