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
#include "refinement/refinement.h"

namespace cuvslam::refinement::rational_polynomial {

/**
 * @brief Calculates the predicted observation of a point in a camera with radial-tangential distortion.
 * This follows OpenCV's distortion model exactly:
 * x'' = x'*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + 2*p1*x'*y' + p2*(r^2 + 2*x'^2)
 * y'' = y'*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + p1*(r^2 + 2*y'^2) + 2*p2*x'*y'
 * where r^2 = x'^2 + y'^2
 *
 * @param ptr_angleaxis_rig_from_world The angle-axis representation of the camera pose.
 * @param ptr_translation_rig_from_world The translation of the camera pose.
 * @param ptr_V_worldpoint The point in world coordinates.
 * @param ptr_fx The focal length in the x-direction.
 * @param ptr_fy The focal length in the y-direction.
 * @param ptr_cx The u-coordinate of the principal point.
 * @param ptr_cy The v-coordinate of the principal point.
 * @param ptr_k1_k2_k3_k4_k5_k6 The radial distortion coefficients (k1 through k6).
 * @param ptr_p1_p2 The tangential distortion coefficients (p1 and p2).
 * @param predicted_u The predicted u-coordinate of the observation.
 * @param predicted_v The predicted v-coordinate of the observation.
 * @param symmetric_focal_length If true, uses fx for both fx and fy.
 */
template <typename T>
bool calculatePredictedObservation(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_V_worldpoint,
                                   const T* const ptr_fx_fy_cx_cy, const T* const ptr_k1_k2_p1_p2_k3_k4_k5_k6,
                                   const Eigen::Matrix<T, 4, 4>& camera_from_rig, T* predicted_u, T* predicted_v,
                                   bool symmetric_focal_length = false);

/**
 * @brief Calculates the reprojection error between the predicted and observed
 * observation of a point in a camera with radial-tangential distortion.
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

  template <typename T>
  bool operator()(const T* const ptr_angleaxis_rig_from_world, const T* const ptr_translation_rig_from_world,
                  const T* const ptr_point, const T* const ptr_fx_fy_cx_cy, const T* const ptr_k1_k2_p1_p2_k3_k4_k5_k6,
                  T* residuals) const;

  Eigen::Matrix<double, 4, 4>* camera_from_rig;
  double observed_x;
  double observed_y;
  bool symmetric_focal_length;
};

// Declare explicit instantiations
extern template bool calculatePredictedObservation<double>(const double* const, const double* const,
                                                           const double* const, const double* const,
                                                           const double* const, const Eigen::Matrix<double, 4, 4>&,
                                                           double*, double*, bool);

extern template bool calculatePredictedObservation<ceres::Jet<double, 21>>(
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const,
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const,
    const Eigen::Matrix<ceres::Jet<double, 21>, 4, 4>&, ceres::Jet<double, 21>*, ceres::Jet<double, 21>*, bool);

extern template bool ReprojectionError::operator()<double>(const double* const, const double* const,
                                                           const double* const, const double* const,
                                                           const double* const, double*) const;

extern template bool ReprojectionError::operator()<ceres::Jet<double, 21>>(
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const,
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const, ceres::Jet<double, 21>*) const;

}  // namespace cuvslam::refinement::rational_polynomial
