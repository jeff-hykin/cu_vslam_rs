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
#include "pnp/multicam_pnp.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"

namespace cuvslam::slam {

class LoopClosureSolverSimple : public ILoopClosureSolver {
public:
  LoopClosureSolverSimple(const camera::Rig& rig, RansacType ransac_type, bool randomized = true,
                          LSIGrid::FetchStrategy fetch_strategy = LSIGrid::FetchStrategy::Volume);
  ~LoopClosureSolverSimple() override;

  bool Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
             const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& pose, Matrix6T& pose_covariance,
             std::vector<LandmarkInSolver>* landmarks, DiscardLandmarkCB* discard_landmark_cb,
             KeyframeInSightCB* keyframe_in_sight_cb) const override;

public:
  // can be called after Solve()
  float GetRansacPerTracked() const;

protected:
  struct LandmarkInfo {
    LandmarkId id;
    Vector3T xyz;
    CameraId cam_id;  // for the obs
    Vector2T uv_norm;
    Vector2T uv_guess_norm;
    float ncc = 0;
    Matrix2T info;  // information matrix
  };

  virtual void SelectLandmarksCandidates(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                                         const IFeatureDescriptorOps* feature_descriptor_ops,
                                         std::vector<LandmarkInfo>& landmark_candidates,
                                         DiscardLandmarkCB* discard_landmark_cb,
                                         KeyframeInSightCB* keyframe_in_sight_cb) const;

  int PnpRansacFilter(const Isometry3T& rig_from_world, const std::vector<LandmarkInfo>& landmark_candidates,
                      std::vector<bool>& landmark_candidates_ok) const;
  int FundamentalRansacFilter(const std::vector<LandmarkInfo>& landmark_candidates,
                              std::vector<bool>& landmark_candidates_ok) const;

  // how LSI works
  LSIGrid::FetchStrategy fetch_strategy_ = LSIGrid::FetchStrategy::Volume;

  RansacType ransac_type_ = RansacType::kNone;
  bool randomized_ = true;

  // statistic
  mutable float ransac_per_tracked_ = 0;
  mutable pnp::PNPSolver pnp_;

  // profiler
  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0xFFFF00;

  mutable bool enable_dump_ = true;

  camera::Rig rig_;
};

}  // namespace cuvslam::slam
