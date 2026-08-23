
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

#include "slam/slam/slam_check_hypothesis.h"

#include "common/log_types.h"

#include "slam/common/slam_common.h"
#include "slam/common/slam_log_types.h"
#include "slam/map/spatial_index/lsi_grid.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"

namespace cuvslam::slam {

void SlamCheckTask::Reset() {
  this->succesed = false;
  landmarks.clear();
  this->probes_types.fill(0);
  discarded_landmarks.clear();
  keyframes_in_sight.clear();
}

// prepared to async
void SlamCheckHypothesis(const LSIGrid& landmarks_spatial_index, const ILoopClosureSolver* loop_closure_solver,
                         const camera::Rig& rig, const IFeatureDescriptorOps* feature_descriptor_ops,
                         SlamCheckTask& task) {
  task.Reset();

  ILoopClosureSolver::DiscardLandmarkCB discard_landmark_cb = [&](LandmarkId landmark_id, LandmarkProbe reason) {
    task.probes_types[reason]++;
    // TODO: it is test
    task.discarded_landmarks.emplace_back(std::pair<LandmarkId, LandmarkProbe>(landmark_id, reason));
  };

  ILoopClosureSolver::KeyframeInSightCB keyframe_in_sight_cb = [&](KeyFrameId keyframe_id) {
    task.keyframes_in_sight.insert(keyframe_id);
  };

  Isometry3T pose;
  Matrix6T pose_covariance;
  task.succesed = loop_closure_solver->Solve(task.loop_closure_task, landmarks_spatial_index, feature_descriptor_ops,
                                             task.result_pose, task.result_pose_covariance, &task.landmarks,
                                             &discard_landmark_cb, &keyframe_in_sight_cb);

  log::Value<LogSlamHypothesis>("landmarks_discarded_count", task.discarded_landmarks.size());

  task.reprojection_error = 0;
  if (task.succesed && !task.landmarks.empty()) {
    // LP_SOLVER_OK
    task.probes_types[LP_SOLVER_OK] = static_cast<int>(task.landmarks.size());

    // calc reprojection error
    Isometry3T left_camera_from_world = rig.camera_from_rig[0] * task.result_pose.inverse();
    double sum_s_reprojection_errors = 0;
    for (auto& landmark : task.landmarks) {
      const Vector3T xyz =
          landmarks_spatial_index.GetLandmarkOrStagedCoords(landmark.id, *task.loop_closure_task.pose_graph_hypothesis);
      auto local_coord = left_camera_from_world * xyz;
      const float z = AvoidZero(local_coord.z());
      Vector2T projected_point = local_coord.head(2) / z;
      double reprojection_error = (projected_point - landmark.uv_norm).squaredNorm();
      sum_s_reprojection_errors += reprojection_error;
    }
    task.reprojection_error = sqrt(sum_s_reprojection_errors / task.landmarks.size());
  }
  if (!task.succesed) {
    // LP_PNP_FAILED
    task.probes_types[LP_PNP_FAILED] = static_cast<int>(task.landmarks.size());
  }
}

}  // namespace cuvslam::slam
