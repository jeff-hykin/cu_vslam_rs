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

#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"
#include "map/plane_map.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::map {

struct DepthPointMapSettings {
  // sample_stride 8 → 4 doubles the unprojected sample-pool density per update
  // (pool is striped down to max_per_cam afterwards), giving better spatial
  // coverage of the room. Cost is one-shot at sample time, not in the LM.
  int sample_stride = 4;
  float depth_min = 0.1f;
  float depth_max = 5.f;

  // A stored point is "close" to camera C when its camera-frame z lies in
  // (0, visible_max_depth]. Eviction keeps points that are close to at least
  // one camera regardless of frustum, so points that drift momentarily outside
  // an image rect (vibration, small yaw) are not needlessly re-sampled.
  // 3 → 5 m matches typical ICL-NUIM living-room/office depth range, so
  // back-wall points are not evicted on slow yaw.
  float visible_max_depth = 5.f;

  // Per-camera target AND cap on close-and-visible points (close + inside the
  // image rect). update() tops up camera C with (max_per_cam - count_C) new
  // samples whenever its count drops below this value.
  // 500 → 1000 doubles the ICP factor count in the LM. Bigger jumps (2000,
  // 5000) were tested and regress on fast living-room sequences (lr1) — once
  // depth dominates the cost, small per-pixel depth-discontinuity errors pull
  // the LM into a worse basin. 1000 is the sweet spot: enough constraint to
  // pin large planar surfaces, not enough to overpower the visual term.
  // ICL-NUIM 13-run lr1 ATE median 13.2 → 5.9 %/m; off0 3.76 → 1.62 %/m.
  size_t max_per_cam = 1000;
};

// Self-managed sparse depth point store for the point-to-point ICP factor.
//
// The map is coverage-driven: every update() first evicts points that are too
// far from every camera, then for each camera tops the close-and-visible
// support back up to `max_per_cam`. Between triggers the map is fully static,
// which preserves the map-to-frame invariant: the pose estimator solves
// against a reference that was captured at earlier (older) poses.
class DepthPointMap {
public:
  explicit DepthPointMap(const DepthPointMapSettings& settings = {});

  // Called every frame by the pipeline. The map internally:
  //   1. evicts points that are no longer close to ANY camera (distance-only),
  //   2. for each camera, counts its close-and-visible support,
  //   3. samples (max_per_cam - count) fresh points from cameras below target.
  void update(const std::vector<DepthCameraInfo>& depth_cameras);

  const std::vector<Vector3T>& get_points() const;

  // True when this update() actually inserted any new points.
  bool added_last_update() const;

  void clear();

private:
  void evict_far(const std::vector<DepthCameraInfo>& depth_cameras);
  size_t count_close_and_visible(const DepthCameraInfo& cam) const;
  void add_samples(const DepthCameraInfo& cam, size_t n);

  DepthPointMapSettings settings_;
  std::vector<Vector3T> points_;
  bool added_ = false;

  mutable cuda::Stream stream_{false};

  static constexpr int kMaxSamplePoints = 640 * 480;
  mutable cuda::GPUArrayPinned<float3> d_sampled_points_{kMaxSamplePoints};
  mutable cuda::GPUArrayPinned<int> d_count_{1};

  profiler::MapProfiler::DomainHelper profiler_domain_ = profiler::MapProfiler::DomainHelper("DepthPointMap");
};

}  // namespace cuvslam::map
