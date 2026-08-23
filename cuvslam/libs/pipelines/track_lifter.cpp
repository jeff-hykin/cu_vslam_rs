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

#include "pipelines/track_lifter.h"

#include <algorithm>

#include "common/vector_2t.h"
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::pipelines {

namespace {

// Matches MultisensorPoseEstimator's per-camera observation cap (kept in sync to share the same
// allocation contract with the PnP solver path).
constexpr size_t kMaxObsPerCamera = 270;

struct CameraParams {
  float2 focal;
  float2 principal;
  cudaTextureObject_t depth_tex;
};

CameraParams extract_camera_params(const camera::Rig& rig, CameraId cam_id, const pnp::RGBDInfo& info) {
  const auto& intrinsics = *rig.intrinsics[cam_id];
  Vector2T focal_v = intrinsics.getFocal();
  Vector2T principal_v = intrinsics.getPrincipal();
  return {
      {focal_v.x(), focal_v.y()}, {principal_v.x(), principal_v.y()}, info.curr_depth[0].get_texture_filter_point()};
}

}  // namespace

TrackLifter::TrackLifter(const camera::Rig& rig)
    : rig_(rig),
      gpu_observation_xy_(kMaxObsPerCamera * rig.num_cameras),
      d_lifted_(kMaxObsPerCamera * rig.num_cameras) {}

void TrackLifter::lift_tracks(const pnp::RGBDInfo& depth_info, const Isometry3T& world_from_rig,
                              const std::vector<camera::Observation>& observations,
                              std::vector<Landmark>& landmarks) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("lift_tracks");
  landmarks.clear();

  const size_t n = std::min(observations.size(), gpu_observation_xy_.size());
  if (n == 0) {
    return;
  }

  for (size_t i = 0; i < n; i++) {
    gpu_observation_xy_[i] = {observations[i].xy.x(), observations[i].xy.y()};
  }

  cudaStream_t s = stream_.get_stream();
  gpu_observation_xy_.copy_top_n(cuda::ToGPU, n, s);

  auto cp = extract_camera_params(rig_, depth_info.depth_id, depth_info);

  CUDA_CHECK(cuda::lift_opencv(reinterpret_cast<const float2*>(gpu_observation_xy_.ptr()), cp.depth_tex, cp.focal.x,
                               cp.focal.y, cp.principal.x, cp.principal.y, d_lifted_.ptr(), n, s));

  d_lifted_.copy_top_n(cuda::ToCPU, n, s);
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Pose chain: world_from_depth = world_from_rig * (camera_from_rig)^{-1}
  const Isometry3T world_from_depth = world_from_rig * rig_.camera_from_rig[depth_info.depth_id].inverse();

  landmarks.reserve(n);
  for (size_t i = 0; i < n; i++) {
    const float3& pt = d_lifted_[i];
    if (pt.z > 1e-3f) {
      Vector3T lm(pt.x, pt.y, pt.z);
      landmarks.push_back({observations[i].id, world_from_depth * lm});
    }
  }
}

}  // namespace cuvslam::pipelines
