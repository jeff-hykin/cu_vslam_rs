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

#include "slam/slam/loop_closure_solver/lcs_two_steps_easy.h"

#include "common/isometry.h"

#include "slam/common/slam_common.h"
#include "slam/common/slam_log_types.h"

namespace cuvslam::slam {

template <class T>
float lerp_inv(T zero_value, T one_value, T value) {
  float res = static_cast<float>(value - zero_value) / static_cast<float>(one_value - zero_value);
  res = std::max(0.f, res);
  res = std::min(1.f, res);
  return res;
}

template <class T>
T lerp(T zero_value, T one_value, float value) {
  T res = (1 - value) * zero_value + value * one_value;
  return res;
}

ILoopClosureSolverPtr CreateLoopClosureSolverTwoStepsEasy(const camera::Rig& rig, RansacType ransac_type,
                                                          bool randomized) {
  return new LoopClosureSolverTwoStepsEasy(rig, ransac_type, randomized);
}

LoopClosureSolverTwoStepsEasy::LoopClosureSolverTwoStepsEasy(const camera::Rig& rig, RansacType ransac_type,
                                                             bool randomized)
    : lc_simple_(rig, ransac_type, randomized), rig_(rig) {}
LoopClosureSolverTwoStepsEasy::~LoopClosureSolverTwoStepsEasy() {}

bool LoopClosureSolverTwoStepsEasy::Solve(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                                          const IFeatureDescriptorOps* feature_descriptor_ops, Isometry3T& pose,
                                          Matrix6T& pose_covariance, std::vector<LandmarkInSolver>* landmarks,
                                          DiscardLandmarkCB* discard_landmark_cb,
                                          KeyframeInSightCB* keyframe_in_sight_cb) const {
  Isometry3T first_lc_pose;
  Matrix6T first_lc_pose_covariance;
  bool first_lc_succesed = lc_simple_.Solve(task, landmarks_spatial_index, feature_descriptor_ops, first_lc_pose,
                                            first_lc_pose_covariance, nullptr, nullptr, nullptr);

  if (!first_lc_succesed) {
    return false;
  }

  LoopClosureTask second_task = task;
  second_task.guess_world_from_rig = first_lc_pose;
  Isometry3T second_lc_pose;
  Matrix6T second_lc_pose_covariance;

  bool second_lc_succesed =
      lc_simple_.Solve(second_task, landmarks_spatial_index, feature_descriptor_ops, second_lc_pose,
                       second_lc_pose_covariance, landmarks, discard_landmark_cb, keyframe_in_sight_cb);
  size_t landmarks_count = landmarks ? (*landmarks).size() : 0;
  float second_lc_ransac_per_tracked = lc_simple_.GetRansacPerTracked();

  if (!second_lc_succesed) {
    return false;
  }

  // compare
  {
    Isometry3T cur_scnd_diff = task.guess_world_from_rig.inverse() * second_lc_pose;
    Isometry3T first_scnd_diff = first_lc_pose.inverse() * second_lc_pose;
    float cur_scnd_tr = cur_scnd_diff.translation().norm();
    float first_scnd_tr = first_scnd_diff.translation().norm();

    float cur_scnd_rot = cur_scnd_diff.linear().norm();
    float first_scnd_rot = first_scnd_diff.linear().norm();

    // condition: first->second must be better than cur->second
    if (first_scnd_tr > cur_scnd_tr && first_scnd_rot > cur_scnd_rot) {
      return false;
    }
  }

  // landmarks count threshold
  const size_t landmarks_count_threshold = 40;
  if (landmarks && (*landmarks).size() < landmarks_count_threshold) {
    return false;
  }

  // 200 -> 0
  // 40 -> 1
  float ransac_per_tracked_factor = lerp_inv(40, 200, static_cast<int>(landmarks_count));
  float ransac_per_tracked_threshold = lerp(0.5f, 0.3f, ransac_per_tracked_factor);
  ransac_per_tracked_threshold = std::max(ransac_per_tracked_threshold, 0.3f);

  if (second_lc_ransac_per_tracked < ransac_per_tracked_threshold) {
    return false;
  }

  // covariance factor
  {
    float cov_factor = 1;

    // weight depends on ransac_per_tracked
    if (true) {
      float factor = second_lc_ransac_per_tracked * 1 + (1 - second_lc_ransac_per_tracked) * 100;
      cov_factor *= factor;
    }

    // update pose_covariance
    pose_covariance = pose_covariance * cov_factor;
  }
  pose = second_lc_pose;
  pose_covariance = second_lc_pose_covariance;
  return true;
}

}  // namespace cuvslam::slam
