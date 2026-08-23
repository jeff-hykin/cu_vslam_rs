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

#include <optional>
#include <vector>

#include <texture_types.h>
#include <vector_types.h>

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

namespace cuvslam::map {

// A 3D point with an associated unit surface normal (rig-frame).
struct SurfacePoint {
  Vector3T point;
  Vector3T normal;
};

// A detected or merged plane in world coordinates.
struct Plane {
  Vector3T centroid;
  Vector3T normal;
  std::vector<Vector3T> convex_hull;
  int num_inliers = 0;
};

// Input data for one depth camera: GPU texture, intrinsics, and world pose.
struct DepthCameraInfo {
  cudaTextureObject_t depth_tex;
  float2 focal;
  float2 principal;
  int2 image_size;
  Isometry3T world_from_cam;
};

// Configuration for plane detection, merging, and map maintenance.
struct PlaneMapSettings {
  int max_planes = 64;

  // Depth filtering
  float depth_min = 0.1f;
  float depth_max = 5.f;
  int unproject_stride = 2;  // sample every Nth pixel for plane detection (1 = full res)

  // RANSAC core
  int ransac_max_iterations = 300;  // max RANSAC iterations per plane
  float ransac_confidence = 0.99f;  // desired probability of finding the best plane
  float inlier_thresh = 2e-2f;      // max point-to-plane distance for an inlier [m]
  int min_plane_inliers = 5e3;      // planes with fewer inliers are discarded; also stops extraction loop

  // Spatial density filter (applied to inliers before convex hull)
  float grid_cell_size = 0.05f;  // 2D grid cell side length for density filtering [m]
  int min_cell_count = 3;        // cells with fewer points are discarded

  // Area and hull
  float min_plane_area = 0.5f;  // planes with convex-hull area below this are discarded [m^2]

  // Plane merging / NMS thresholds
  float merge_normal_thresh = 0.9f;  // min normal dot-product for two planes to be merge-compatible
  float merge_dist_thresh = 0.1f;    // max centroid-to-plane distance for merge compatibility [m]
  float nms_overlap_thresh = 0.1f;   // min convex-hull overlap ratio to suppress a redundant plane

  float max_plane_dist = 15.f;  // evict planes whose centroid is farther than this from all cameras [m]

  // Cap on the number of inliers fed into the CPU-side spatial_filter and convex_hull.
  // When refined_count exceeds this, inliers are stride-subsampled on the GPU before
  // download.  Compaction (active-set removal) still uses the full refined inlier set,
  // so plane growth is unaffected.  1500 keeps spatial coverage (every ~10 cm at typical
  // indoor depth densities), preserves the convex hull shape (extremes are statistically
  // retained), and shrinks both CPU stages roughly 10x on a wall-sized plane.
  int max_inliers_for_geometry = 1500;
};

// Persistent plane map: detects planes from depth cameras via GPU-accelerated RANSAC
// and maintains a merged set of planes across keyframes.
class PlaneMap {
public:
  explicit PlaneMap(const PlaneMapSettings& settings = {});

  void update_at_keyframe(const std::vector<DepthCameraInfo>& depth_cameras);

  const std::vector<Plane>& get_planes() const;

  std::vector<SurfacePoint> get_plane_surface_points() const;

  void clear();

private:
  PlaneMapSettings settings_;
  std::vector<Plane> planes_;

  mutable cuda::Stream stream_{false};

  static constexpr int kMaxOutputPoints = 640 * 480;
  mutable cuda::GPUArrayPinned<float3> d_points_{kMaxOutputPoints};
  mutable cuda::GPUArrayPinned<int> d_active_indices_{kMaxOutputPoints};
  mutable cuda::GPUArrayPinned<int> d_inlier_indices_{kMaxOutputPoints};
  // Active-set positions of the inliers gathered alongside d_inlier_indices_; used to
  // scatter removal flags into d_keep_flags_ without round-tripping inlier indices to CPU.
  mutable cuda::GPUArrayPinned<int> d_inlier_positions_{kMaxOutputPoints};
  // Stride-subsampled inlier indices (point ids) for the geometry stages.
  mutable cuda::GPUArrayPinned<int> d_inlier_indices_geom_{kMaxOutputPoints};
  // Tight gather of d_points_ at d_inlier_indices_geom_; the only inlier buffer copied DtoH.
  mutable cuda::GPUArrayPinned<float3> d_inlier_pts_compact_{kMaxOutputPoints};
  mutable cuda::GPUArrayPinned<int> d_count_{1};

  static constexpr int kRansacBatchSize = 128;
  mutable cuda::GPUArrayPinned<float4> d_hypotheses_{kRansacBatchSize};
  mutable cuda::GPUArrayPinned<int> d_hyp_counts_{kRansacBatchSize};
  mutable cuda::GPUArrayPinned<int3> d_triplets_{kRansacBatchSize};

  // 9 floats: centroid(3) + upper-triangle covariance(6)
  mutable cuda::GPUArrayPinned<float> d_stats_{9};

  // Flags and double-buffer for GPU-side active index compaction
  mutable cuda::GPUArrayPinned<uint8_t> d_keep_flags_{kMaxOutputPoints};
  mutable cuda::GPUArrayPinned<int> d_active_indices_b_{kMaxOutputPoints};

  std::vector<Plane> detect_planes(const std::vector<DepthCameraInfo>& depth_cameras) const;

  std::vector<Plane> detect_planes_single_camera(const DepthCameraInfo& cam) const;

  void merge_planes(const std::vector<Plane>& detected);

  static float convex_hull_area(const std::vector<Vector3T>& hull);
  static std::vector<Vector3T> compute_convex_hull(const std::vector<Vector3T>& points, const Vector3T& normal,
                                                   const Vector3T& centroid);
  static void merge_two_planes(Plane& target, const Plane& source);
  std::optional<size_t> find_best_match(const Plane& new_plane) const;
  static float hull_overlap_ratio(const Plane& a, const Plane& b);

  std::vector<Plane> suppress_redundant(std::vector<Plane>& planes) const;

  profiler::MapProfiler::DomainHelper profiler_domain_ = profiler::MapProfiler::DomainHelper("PlaneMap");
};

}  // namespace cuvslam::map
