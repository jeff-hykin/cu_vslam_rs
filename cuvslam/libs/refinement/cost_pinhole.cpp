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

#include "refinement/cost_pinhole.h"
#include "refinement/refinement.h"

#include "ceres/ceres.h"

namespace cuvslam::refinement::pinhole {

/**
 * @brief Calculates the predicted observation of a point in a camera.
 *
 * @param ptr_angleaxis_rig_from_world The angle-axis representation of the
 * camera pose.
 * @param ptr_translation_rig_from_world The translation of the camera pose.
 * @param ptr_V_worldpoint The point in world coordinates.
 * @param ptr_fx The focal length in the x-direction.
 * @param ptr_fy The focal length in the y-direction.
 * @param ptr_cx The u-coordinate of the principal point.
 * @param ptr_cy The v-coordinate of the principal point.
 * @param predicted_u The predicted u-coordinate of the observation.
 * @param predicted_v The predicted v-coordinate of the observation.
 */
template <typename T>
bool calculatePredictedObservation(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_V_worldpoint,
                                   const T* const ptr_fx_fy_cx_cy, const Eigen::Matrix<T, 4, 4>& camera_from_rig,
                                   T* predicted_u, T* predicted_v, bool symmetric_focal_length) {
  Eigen::Vector<T, 3> V_world_point{ptr_V_worldpoint[0], ptr_V_worldpoint[1], ptr_V_worldpoint[2]}, V_camera_point;

  ceres::AngleAxisRotatePoint(ptr_angleaxis_rig_from_world, V_world_point.data(), V_camera_point.data());

  // Transform the point from world to camera coordinate
  V_camera_point = V_camera_point + Eigen::Map<const Eigen::Vector<T, 3>>(ptr_translation_rig_from_world);

  V_camera_point =
      camera_from_rig.template block<3, 3>(0, 0) * V_camera_point + camera_from_rig.template block<3, 1>(0, 3);

  if (V_camera_point[2] < T(kEPS)) {
    // Prevent divide-by-zero and points behind the camera.
    return false;
  }

  // Compute final projected point position.
  const T& fx = ptr_fx_fy_cx_cy[0];
  const T& fy = ptr_fx_fy_cx_cy[1];
  const T& cx = ptr_fx_fy_cx_cy[2];
  const T& cy = ptr_fx_fy_cx_cy[3];
  const T& focal_length_x = fx;
  const T& focal_length_y = symmetric_focal_length ? fx : fy;  // Use fx for both if symmetric

  *predicted_u = focal_length_x * (V_camera_point[0] / V_camera_point[2]) + cx;
  *predicted_v = focal_length_y * (V_camera_point[1] / V_camera_point[2]) + cy;
  return true;
}

/**
 * @brief Calculates the reprojection error between the predicted and observed
 * observation of a point in a camera.
 */
template <typename T>
bool ReprojectionError::operator()(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_point,
                                   const T* const ptr_fx_fy_cx_cy, T* residuals) const {
  T predicted_x;
  T predicted_y;

  const Eigen::Matrix<T, 4, 4> camera_from_rig_cast = camera_from_rig->template cast<T>();

  if (!calculatePredictedObservation(ptr_angleaxis_rig_from_world, ptr_translation_rig_from_world, ptr_point,
                                     ptr_fx_fy_cx_cy, camera_from_rig_cast, &predicted_x, &predicted_y,
                                     symmetric_focal_length)) {
    // Prevent divide-by-zero and points behind the camera.
    residuals[0] = T(0);
    residuals[1] = T(0);
    return true;
  }

  // The error is the difference between the predicted and observed position.
  residuals[0] = predicted_x - T(observed_x);
  residuals[1] = predicted_y - T(observed_y);

  return true;
}

// Explicit template instantiations
template bool calculatePredictedObservation<double>(const double* const, const double* const, const double* const,
                                                    const double* const, const Eigen::Matrix<double, 4, 4>&, double*,
                                                    double*, bool);
template bool calculatePredictedObservation<ceres::Jet<double, 13>>(
    const ceres::Jet<double, 13>* const, const ceres::Jet<double, 13>* const, const ceres::Jet<double, 13>* const,
    const ceres::Jet<double, 13>* const, const Eigen::Matrix<ceres::Jet<double, 13>, 4, 4>&, ceres::Jet<double, 13>*,
    ceres::Jet<double, 13>*, bool);

template bool ReprojectionError::operator()(double const*, double const*, double const*, double const*, double*) const;
template bool ReprojectionError::operator()(ceres::Jet<double, 13> const*, ceres::Jet<double, 13> const*,
                                            ceres::Jet<double, 13> const*, ceres::Jet<double, 13> const*,
                                            ceres::Jet<double, 13>*) const;

}  // namespace cuvslam::refinement::pinhole
