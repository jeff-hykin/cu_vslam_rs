
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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ceres/ceres.h"

#include "cuvslam/cuvslam2.h"

namespace cuvslam::refinement {

struct BundleAdjustmentProblem {
  // The camera Rig which contains the camera intrinsics and extrinsics.
  Rig rig;

  // A mapping from landmark id to 3D point in world frame
  std::unordered_map<uint64_t, Landmark> points_in_world;

  // A mapping from frame id to rig pose in world frame
  std::unordered_map<uint64_t, Pose> world_from_rigs;

  // A mapping from frame id to a vector of observations
  std::unordered_map<uint64_t, std::vector<Observation>> observations;

  // A set containing the ids of the fixed points
  std::unordered_set<uint64_t> fixed_points;

  // A set containing the ids of the fixed frames
  std::unordered_set<uint64_t> fixed_frames;
};

struct BundleAdjustmentProblemOptions {
  // Solver options
  bool verbose = false;

  // Whether to export the iteration state
  bool export_iteration_state = false;

  bool estimate_intrinsics = false;
  bool symmetric_focal_length = false;  // When true, fx will be optimized and fy will be set equal to fx

  bool use_loss_function = false;
  float max_reprojection_error = 10.0f;
  float cauchy_loss_scale = 3.0f;

  ceres::Solver::Options ceres_options;
};

struct BundleAdjustmentProblemSummary {
  ceres::Solver::Summary ceres_summary;

  // Keep track of all states at each iteration. This is populated if the
  // `export_iteration_state` flag is set in the `BundleAdjustmentProblemOptions`.
  std::vector<std::unordered_map<uint64_t, Pose>> iteration_rigs_from_world;
  std::vector<std::unordered_map<uint64_t, Landmark>> iteration_points_in_world;
  std::vector<Rig> iteration_cameras;
};

}  // namespace cuvslam::refinement
