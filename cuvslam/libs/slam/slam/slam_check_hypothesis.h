
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

#include <unordered_set>

#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"

#include "slam/common/slam_common.h"
#include "slam/map/pose_graph/pose_graph.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"

namespace cuvslam::slam {

struct SlamCheckTask {
  // input
  LoopClosureTask loop_closure_task;

  // output
  bool succesed = false;
  Isometry3T result_pose;
  Matrix6T result_pose_covariance;
  std::vector<LandmarkInSolver> landmarks;
  double reprojection_error;

  std::array<int, LP_MAX> probes_types;
  std::vector<std::pair<LandmarkId, LandmarkProbe>> discarded_landmarks;  // tODO: redo

  std::unordered_set<KeyFrameId> keyframes_in_sight;

public:
  void Reset();
};

// prepared to async
void SlamCheckHypothesis(const LSIGrid& landmarks_spatial_index, const ILoopClosureSolver* loop_closure_solver,
                         const camera::Rig& rig, const IFeatureDescriptorOps* feature_descriptor_ops,
                         SlamCheckTask& task);

}  // namespace cuvslam::slam
