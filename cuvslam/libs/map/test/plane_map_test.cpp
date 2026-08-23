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

#include <cmath>
#include <functional>
#include <vector>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/vector_3t.h"
#include "cuda_modules/cuda_helper.h"

#include "map/plane_map.h"

namespace test::map {

using namespace cuvslam;

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr float kFx = 320.f;
constexpr float kFy = 320.f;
constexpr float kCx = 320.f;
constexpr float kCy = 240.f;

// Synthesises a depth image from a per-pixel function and exposes it as a
// DepthCameraInfo backed by a GPU texture.
struct DepthCam {
  cuvslam::cuda::GPUImage<float> depth_img;

  explicit DepthCam(const std::function<float(int u, int v)>& depth_fn) : depth_img(kWidth, kHeight) {
    std::vector<float> host(kWidth * kHeight);
    for (int v = 0; v < kHeight; ++v) {
      for (int u = 0; u < kWidth; ++u) {
        host[v * kWidth + u] = depth_fn(u, v);
      }
    }
    depth_img.copy(cuvslam::cuda::ToGPU, host.data(), nullptr);
    cudaDeviceSynchronize();
  }

  cuvslam::map::DepthCameraInfo info(const Isometry3T& world_from_cam) {
    return {depth_img.get_texture_filter_point(), {kFx, kFy}, {kCx, kCy}, {kWidth, kHeight}, world_from_cam};
  }
};

// Tilted plane in camera frame: n^T x = c, with n a unit vector and c > 0.
// For camera-frame point p_cam = depth * (px, py, 1) where px = (u - cx)/fx,
// solving n^T p_cam = c gives depth = c / (n.x*px + n.y*py + n.z).
float depth_for_tilted_plane(int u, int v, const Vector3T& n_cam, float c) {
  const float px = (static_cast<float>(u) - kCx) / kFx;
  const float py = (static_cast<float>(v) - kCy) / kFy;
  const float denom = n_cam.x() * px + n_cam.y() * py + n_cam.z();
  if (denom <= 1e-3f) {
    return 0.f;  // unprojection skips zero/non-positive depth
  }
  return c / denom;
}

}  // namespace

// Single tilted plane filling the full frame must be detected, with the
// recovered normal and centroid close to the synthetic ground truth.
TEST(PlaneMap, DetectsLargePlaneOnSyntheticDepth) {
  cuvslam::map::PlaneMapSettings settings;
  cuvslam::map::PlaneMap pm(settings);

  // Slightly tilted plane: normal pointing back toward camera with a small +x component.
  Vector3T n_cam = Vector3T(0.1f, 0.f, 1.f).normalized();
  const float c = 2.f;  // plane offset in camera frame

  DepthCam cam([&](int u, int v) { return depth_for_tilted_plane(u, v, n_cam, c); });
  std::vector<cuvslam::map::DepthCameraInfo> cams{cam.info(Isometry3T::Identity())};

  pm.update_at_keyframe(cams);

  ASSERT_FALSE(pm.get_planes().empty()) << "Single fronto-tilted plane should be detected.";
  // World == camera frame here (identity pose), so the world-frame plane normal
  // should equal -n_cam (cuvslam convention: plane normal points away from the
  // camera, opposite of the surface-towards-camera direction).
  const Vector3T expected_normal = -n_cam;
  const Vector3T got_normal = pm.get_planes().front().normal;
  EXPECT_GT(std::abs(got_normal.dot(expected_normal)), 0.95f)
      << "Recovered normal " << got_normal.transpose() << " differs from expected " << expected_normal.transpose();

  const Vector3T centroid = pm.get_planes().front().centroid;
  // Centroid should lie on the plane: |n . centroid - c| small.
  const float plane_residual = std::abs(n_cam.dot(centroid) - c);
  EXPECT_LT(plane_residual, 0.05f) << "Centroid " << centroid.transpose() << " is not on the plane (residual "
                                   << plane_residual << " m).";
}

// Two parallel planes ~30 cm apart in depth must both be detected: the second
// plane will only be reachable if the first plane's full refined inlier set
// has been removed from the active list (the algorithmic-change guard).
TEST(PlaneMap, RemovesAllRefinedInliersFromActiveSet) {
  cuvslam::map::PlaneMapSettings settings;
  cuvslam::map::PlaneMap pm(settings);

  // Top half of the image: plane at depth 2.0 m. Bottom half: plane at depth 2.3 m.
  // The 0.3 m gap is much larger than inlier_thresh (2 cm), so the two planes
  // are unambiguously separated.
  DepthCam cam([&](int u, int v) {
    (void)u;
    return v < kHeight / 2 ? 2.0f : 2.3f;
  });
  std::vector<cuvslam::map::DepthCameraInfo> cams{cam.info(Isometry3T::Identity())};

  pm.update_at_keyframe(cams);

  const auto& planes = pm.get_planes();
  ASSERT_GE(planes.size(), 2u) << "Both parallel planes should be detected; got " << planes.size();

  // Both planes are fronto-parallel, so their world-frame centroids differ
  // primarily in z.
  std::vector<float> centroid_zs;
  centroid_zs.reserve(planes.size());
  for (const auto& p : planes) {
    centroid_zs.push_back(p.centroid.z());
  }
  std::sort(centroid_zs.begin(), centroid_zs.end());
  EXPECT_NEAR(centroid_zs.front(), 2.0f, 0.1f);
  EXPECT_NEAR(centroid_zs.back(), 2.3f, 0.1f);
}

// With max_inliers_for_geometry deliberately set very low, a single huge plane
// must still be detected: the cap only applies to the CPU-side geometry stages
// (spatial_filter and convex_hull), while RANSAC counting and active-set
// compaction continue to use the full inlier set.
TEST(PlaneMap, RespectsMaxInliersForGeometryCap) {
  cuvslam::map::PlaneMapSettings settings;
  // A tiny cap forces the stride-subsample path; correctness must survive it.
  settings.max_inliers_for_geometry = 200;
  cuvslam::map::PlaneMap pm(settings);

  // Fronto-parallel plane at 2 m: a full 640x480 frame yields ~76k inliers
  // after unproject_stride=2 sampling, so the geometry cap is exercised.
  DepthCam cam([](int /*u*/, int /*v*/) { return 2.f; });
  std::vector<cuvslam::map::DepthCameraInfo> cams{cam.info(Isometry3T::Identity())};

  pm.update_at_keyframe(cams);

  ASSERT_GE(pm.get_planes().size(), 1u) << "Detection should still succeed under a low geometry cap.";
  const auto& plane = pm.get_planes().front();
  // Reported num_inliers reflects the full refined inlier set, not the
  // subsampled geometry input.
  EXPECT_GT(plane.num_inliers, settings.max_inliers_for_geometry)
      << "Reported num_inliers should reflect the full refined set, not the geometry cap.";
  EXPECT_NEAR(plane.centroid.z(), 2.f, 0.05f);
}

}  // namespace test::map
