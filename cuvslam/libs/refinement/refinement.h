
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

#include "ceres/ceres.h"
#include "ceres/loss_function.h"
#include "ceres/rotation.h"
#include "refinement/bundle_adjustment_problem.h"

namespace cuvslam::refinement {

static constexpr double kEPS = 1e-4;

/**
 * @brief Inverts a pose transformation.
 *
 * This function computes the inverse of a pose, which represents the transformation
 * from the target frame to the source frame when the input pose represents the
 * transformation from the source frame to the target frame.
 *
 * For a pose with rotation R and translation t, the inverse pose has:
 * - Rotation: R^-1 (inverse of the quaternion)
 * - Translation: -R^-1 * t (negative of the inverse rotation applied to the translation)
 *
 * @param pose The input pose to invert.
 * @return cuvslam::Pose The inverted pose.
 */
cuvslam::Pose invertPose(const cuvslam::Pose& pose_const);

/**
 * @brief Converts a pose to a 4x4 transformation matrix.
 *
 * This function constructs a 4x4 homogeneous transformation matrix from a pose,
 * which contains a rotation (as a quaternion) and a translation vector.
 *
 * For a pose with rotation R and translation t, the resulting matrix is:
 * [ R | t ]
 * [ 0 | 1 ]
 *
 * @param pose The input pose to convert.
 * @return Eigen::Matrix4f The 4x4 transformation matrix.
 */
Eigen::Matrix4f poseToMatrix(const cuvslam::Pose& pose_const);

/**
 * @brief Refines the bundle adjustment problem.
 *
 * @param problem The bundle adjustment problem to refine.
 * @param options The options for the refinement.
 * @param refined_problem The refined bundle adjustment problem.
 * @return The summary of the refinement.
 */
BundleAdjustmentProblemSummary refine(const BundleAdjustmentProblem& problem,
                                      const BundleAdjustmentProblemOptions& options,
                                      BundleAdjustmentProblem& refined_problem);

}  // namespace cuvslam::refinement
