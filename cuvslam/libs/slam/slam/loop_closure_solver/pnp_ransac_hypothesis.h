
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

#include "Eigen/Geometry"

#include "camera/camera.h"
#include "camera/observation.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "math/ransac.h"
#include "pnp/multicam_pnp.h"

namespace cuvslam::slam {

struct PnpRansacTrackData {
  Vector3T xyz;
  camera::Observation obs;

  PnpRansacTrackData(const Vector3T& xyz, const camera::Observation& obs) {
    this->xyz = xyz;
    this->obs = obs;
  }
};
class PnpRansacHypothesis : public math::HypothesisBase<float, PnpRansacTrackData, Isometry3T, 3> {
public:
  PnpRansacHypothesis(const camera::Rig& rig) : pnp_(rig), rig_(rig) {}

  void setThreshold(float reprojection_threshold) {
    square_reprojection_threshold_ = reprojection_threshold * reprojection_threshold;
  }
  void setInitialRigFromWorld(const Isometry3T& rig_from_world) { initial_rig_from_world_ = rig_from_world; }

protected:
  pnp::PNPSolver pnp_;
  camera::Rig rig_;
  float square_reprojection_threshold_ = 1e-3f;
  Isometry3T initial_rig_from_world_ = Isometry3T::Identity();

  mutable std::unordered_map<TrackId, Vector3T> pnp_landmarks;
  mutable std::vector<camera::Observation> pnp_observations;

protected:
  // Required method for Ransac, called by Ransac operator()
  template <typename _ItType>
  bool evaluate(Isometry3T& rig_from_world, _ItType beginIt, _ItType endIt) const {
    pnp_landmarks.clear();
    pnp_observations.clear();

    for_each(beginIt, endIt, [&](const PnpRansacTrackData& i) {
      pnp_landmarks[i.obs.id] = i.xyz;
      pnp_observations.emplace_back(i.obs);
    });

    return Pnp(rig_from_world);
  }
  bool Pnp(Isometry3T& rig_from_world) const {
    Matrix6T precision_local;
    Isometry3T rig_from_world_guess = initial_rig_from_world_;

    bool rig_from_world_res_local = pnp_.solve(rig_from_world_guess, precision_local, pnp_observations, pnp_landmarks,
                                               pnp::PNPSettings::SLAMRansacSettings());
    if (!rig_from_world_res_local) {
      return false;
    }

    rig_from_world = rig_from_world_guess;
    return true;
  }

  // Required method for Ransac, called by main Ransac method.
  template <typename _ItType>
  size_t countInliers(const Isometry3T& rig_from_world, const _ItType beginIt, const _ItType endIt) const {
    size_t inliers = 0;
    for_each(beginIt, endIt, [&](const PnpRansacTrackData& i) {
      if (isInlier(rig_from_world, i)) {
        inliers++;
      }
    });
    return inliers;
  }

public:
  bool isInlier(const Isometry3T& rig_from_world, const PnpRansacTrackData& i) const {
    auto camera_from_world = rig_.camera_from_rig[i.obs.cam_id] * rig_from_world;
    auto local_coord = camera_from_world * i.xyz;
    const float z = AvoidZero(local_coord.z());
    Vector2T projected_point = local_coord.head(2) / z;
    double square_reprojection_error = (projected_point - i.obs.xy).squaredNorm();
    return (square_reprojection_error < this->square_reprojection_threshold_);
  }

protected:
};

}  // namespace cuvslam::slam
