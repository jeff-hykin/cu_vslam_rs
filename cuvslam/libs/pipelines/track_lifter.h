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

#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "cuda_modules/cuda_helper.h"
#include "pipelines/track.h"
#include "pnp/multisensor_pose_estimator.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::pipelines {

// Self-contained mono-track lifter: turns 2D observations into 3D world-frame landmarks
// by sampling a per-camera depth pyramid via the cuda::lift_opencv kernel.
//
// Holds its own CUDA stream and pinned scratch buffers so it can be invoked independently of
// MultisensorPoseEstimator (which it used to live inside). Used by DepthMapsContext to lift mono
// tracks at keyframe time.
class TrackLifter {
public:
  explicit TrackLifter(const camera::Rig& rig);

  // Lift `observations` through `depth_info`'s level-0 depth texture and emit world-frame
  // landmarks via `landmarks` (cleared on entry). Drops points with non-positive depth.
  void lift_tracks(const pnp::RGBDInfo& depth_info, const Isometry3T& world_from_rig,
                   const std::vector<camera::Observation>& observations, std::vector<Landmark>& landmarks) const;

private:
  const camera::Rig& rig_;
  mutable cuda::Stream stream_{false};
  mutable cuda::GPUArrayPinned<float2> gpu_observation_xy_;
  mutable cuda::GPUArrayPinned<float3> d_lifted_;

  profiler::PnPProfiler::DomainHelper profiler_domain_ = profiler::PnPProfiler::DomainHelper("TrackLifter");
};

}  // namespace cuvslam::pipelines
