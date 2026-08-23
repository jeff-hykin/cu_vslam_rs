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

#include "refinement/cost_rational_polynomial.h"
#include "refinement/refinement.h"

namespace cuvslam::refinement::rational_polynomial {

template <typename T>
bool calculatePredictedObservation(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_V_worldpoint,
                                   const T* const ptr_fx_fy_cx_cy, const T* const ptr_k1_k2_p1_p2_k3_k4_k5_k6,
                                   const Eigen::Matrix<T, 4, 4>& camera_from_rig, T* predicted_u, T* predicted_v,
                                   bool symmetric_focal_length) {
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

  // Normalize coordinates (x' = X/Z, y' = Y/Z)
  T x_p = V_camera_point[0] / V_camera_point[2];  // x'
  T y_p = V_camera_point[1] / V_camera_point[2];  // y'

  // Calculate r^2 = x'^2 + y'^2
  T r2 = x_p * x_p + y_p * y_p;
  T r4 = r2 * r2;
  T r6 = r4 * r2;

  // Get distortion coefficients
  T k1 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[0];
  T k2 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[1];
  T p1 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[2];
  T p2 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[3];
  T k3 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[4];
  T k4 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[5];
  T k5 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[6];
  T k6 = ptr_k1_k2_p1_p2_k3_k4_k5_k6[7];

  // Calculate radial distortion factor
  T numerator = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
  T denominator = T(1) + k4 * r2 + k5 * r4 + k6 * r6;

  // While theoretically the denominator could be zero if k4*r² + k5*r⁴ + k6*r⁶ = -1,
  // this should never happen in practice with a properly calibrated camera as it would:
  // 1. Create a mathematical singularity in the distortion model
  // 2. Represent infinite distortion at that radius, which is physically meaningless
  // This check is defensive programming for invalid camera calibration parameters
  if (ceres::abs(denominator) < T(kEPS)) {
    return false;
  }

  T radial_factor = numerator / denominator;

  // Calculate tangential distortion terms
  T xy = x_p * y_p;     // x'y'
  T x_p_2 = x_p * x_p;  // x'^2
  T y_p_2 = y_p * y_p;  // y'^2

  // Apply distortion (OpenCV's exact formulation)
  // x'' = x'*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + 2*p1*x'*y' + p2*(r^2 + 2*x'^2)
  // y'' = y'*(1 + k1*r^2 + k2*r^4 + k3*r^6)/(1 + k4*r^2 + k5*r^4 + k6*r^6) + p1*(r^2 + 2*y'^2) + 2*p2*x'*y'
  T x_pp = x_p * radial_factor + T(2) * p1 * xy + p2 * (r2 + T(2) * x_p_2);
  T y_pp = y_p * radial_factor + p1 * (r2 + T(2) * y_p_2) + T(2) * p2 * xy;

  // Project to pixel coordinates (u = fx*x'' + cx, v = fy*y'' + cy)
  // If symmetric_focal_length is true, use fx for both dimensions
  const T& fx = ptr_fx_fy_cx_cy[0];
  const T& fy = ptr_fx_fy_cx_cy[1];
  const T& cx = ptr_fx_fy_cx_cy[2];
  const T& cy = ptr_fx_fy_cx_cy[3];
  *predicted_u = fx * x_pp + cx;
  *predicted_v = symmetric_focal_length ? fx * y_pp + cy : fy * y_pp + cy;

  return true;
}

// Explicit template instantiations
template bool calculatePredictedObservation<double>(const double* const, const double* const, const double* const,
                                                    const double* const, const double* const,
                                                    const Eigen::Matrix<double, 4, 4>&, double*, double*, bool);

template bool calculatePredictedObservation<ceres::Jet<double, 21>>(
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const,
    const ceres::Jet<double, 21>* const, const ceres::Jet<double, 21>* const,
    const Eigen::Matrix<ceres::Jet<double, 21>, 4, 4>&, ceres::Jet<double, 21>*, ceres::Jet<double, 21>*, bool);

template <typename T>
bool ReprojectionError::operator()(const T* const ptr_angleaxis_rig_from_world,
                                   const T* const ptr_translation_rig_from_world, const T* const ptr_point,
                                   const T* const ptr_fx_fy_cx_cy, const T* const ptr_k1_k2_p1_p2_k3_k4_k5_k6,
                                   T* residuals) const {
  T predicted_x;
  T predicted_y;

  const Eigen::Matrix<T, 4, 4> camera_from_rig_cast = camera_from_rig->template cast<T>();
  if (!calculatePredictedObservation(ptr_angleaxis_rig_from_world, ptr_translation_rig_from_world, ptr_point,
                                     ptr_fx_fy_cx_cy, ptr_k1_k2_p1_p2_k3_k4_k5_k6, camera_from_rig_cast, &predicted_x,
                                     &predicted_y, symmetric_focal_length)) {
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

template bool ReprojectionError::operator()(double const*, double const*, double const*, double const*, double const*,
                                            double*) const;
template bool ReprojectionError::operator()(ceres::Jet<double, 21> const*, ceres::Jet<double, 21> const*,
                                            ceres::Jet<double, 21> const*, ceres::Jet<double, 21> const*,
                                            ceres::Jet<double, 21> const*, ceres::Jet<double, 21>*) const;

}  // namespace cuvslam::refinement::rational_polynomial
