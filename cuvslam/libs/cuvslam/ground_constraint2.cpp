
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

#include "cuvslam/ground_constraint2.h"

#include <utility>

#include "common/isometry.h"
#include "cuvslam/internal.h"
#include "odometry/ground_integrator.h"

namespace cuvslam {

namespace {

inline Isometry3T ToInternal(const Pose& pose) { return ConvertPoseToIsometry(pose); }

inline Pose ToPublic(const Isometry3T& iso) { return ConvertIsometryToPose(iso); }

}  // namespace

struct GroundConstraint::Impl {
  odom::GroundIntegrator integrator;
  Impl(const Pose& world_from_ground, const Pose& initial_pose_on_ground, const Pose& initial_pose_in_space)
      : integrator(ToInternal(world_from_ground), ToInternal(initial_pose_on_ground),
                   ToInternal(initial_pose_in_space)) {}
};

GroundConstraint::GroundConstraint(const Pose& world_from_ground, const Pose& initial_pose_on_ground,
                                   const Pose& initial_pose_in_space)
    : impl(std::make_unique<Impl>(world_from_ground, initial_pose_on_ground, initial_pose_in_space)) {}

GroundConstraint::~GroundConstraint() = default;

GroundConstraint::GroundConstraint(GroundConstraint&&) noexcept = default;
GroundConstraint& GroundConstraint::operator=(GroundConstraint&&) noexcept = default;

void GroundConstraint::AddNextPose(const Pose& next_pose_in_space) {
  impl->integrator.AddNextPose(ToInternal(next_pose_in_space));
}

Pose GroundConstraint::GetPoseOnGround() const { return ToPublic(impl->integrator.GetPoseOnGround()); }

}  // namespace cuvslam
