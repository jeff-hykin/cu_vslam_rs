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

namespace cuvslam::refinement::pinhole {

/**
 * @brief Calculates the predicted observation of a point in a camera.
 *
 * @param ptr_angleaxis_rig_from_world The angle-axis representation of the
 * camera pose.
 * @param ptr_translation_rig_from_world The translation of the camera pose.
 * @param ptr_V_worldpoint The point in world coordinates.
 * @param ptr_fx_fy_cx_cy The focal length in the x-direction and the y-direction, the u-coordinate of the principal
 * point, and the v-coordinate of the principal point.
 * @param predicted_u The predicted u-coordinate of the observation.
 * @param predicted_v The predicted v-coordinate of the observation.
 * @param symmetric_focal_length Whether to use fx for both focal lengths.
 */
template <typename T>
bool calculatePredictedObservation(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_V_worldpoint,
                                   const T* const ptr_fx_fy_cx_cy, const Eigen::Matrix<T, 4, 4>& camera_from_rig,
                                   T* predicted_u, T* predicted_v, bool symmetric_focal_length);

/**
 * @brief Calculates the reprojection error between the predicted and observed
 * observation of a point in a camera.
 */
struct ReprojectionError {
  ReprojectionError(Eigen::Matrix<double, 4, 4>* camera_from_rig, double observed_x, double observed_y,
                    bool symmetric_focal_length = false)
      : camera_from_rig(camera_from_rig),
        observed_x(observed_x),
        observed_y(observed_y),
        symmetric_focal_length(symmetric_focal_length) {
    if (camera_from_rig == nullptr) {
      throw std::runtime_error("camera_from_rig is nullptr");
    }
  }

  /**
   * @brief Calculates the reprojection error between the predicted and observed
   * observation of a point in a camera.
   *
   * @param ptr_angleaxis_rig_from_world The angle-axis representation of the camera pose.
   * @param ptr_translation_rig_from_world The translation of the camera pose.
   * @param ptr_point The point in world coordinates.
   * @param ptr_fx_fy_cx_cy The focal length in the x-direction and the y-direction, the u-coordinate of the principal
   * point, and the v-coordinate of the principal point.
   * @param residuals The reprojection error.
   */
  template <typename T>
  bool operator()(const T* const ptr_angleaxis_rig_from_world, const T* const ptr_translation_rig_from_world,
                  const T* const ptr_point, const T* const ptr_fx_fy_cx_cy, T* residuals) const;

  Eigen::Matrix<double, 4, 4>* camera_from_rig;
  double observed_x;
  double observed_y;
  bool symmetric_focal_length;
};

}  // namespace cuvslam::refinement::pinhole
