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

#include "map/plane_map.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "common/log.h"
#include "common/vector_2t.h"
#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::map {

namespace {

// PCA-based plane fitting: the smallest eigenvalue's eigenvector of the point
// covariance matrix gives the plane normal (the direction of least variance).
void eigen_symmetric_3x3(const Matrix3T& C, Vector3T& eigenvalues, Matrix3T& eigenvectors) {
  Eigen::SelfAdjointEigenSolver<Matrix3T> solver(C);
  eigenvalues = solver.eigenvalues();
  eigenvectors = solver.eigenvectors();
}

void build_plane_frame(const Vector3T& normal, Vector3T& u, Vector3T& v) {
  if (std::abs(normal.x()) < 0.9f) {
    u = normal.cross(Vector3T::UnitX()).normalized();
  } else {
    u = normal.cross(Vector3T::UnitY()).normalized();
  }
  v = normal.cross(u).normalized();
}

std::vector<Vector2T> project_hull_2d(const std::vector<Vector3T>& hull, const Vector3T& u, const Vector3T& v,
                                      const Vector3T& origin) {
  std::vector<Vector2T> out(hull.size());
  for (size_t i = 0; i < hull.size(); ++i) {
    const Vector3T d = hull[i] - origin;
    out[i] = Vector2T(d.dot(u), d.dot(v));
  }
  return out;
}

// Shoelace formula: area = 0.5 * |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)|
float polygon_area_2d(const std::vector<Vector2T>& poly) {
  if (poly.size() < 3) {
    return 0.f;
  }
  float area = 0.f;
  for (size_t i = 0; i < poly.size(); ++i) {
    size_t j = (i + 1) % poly.size();
    area += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
  }
  return std::abs(area) * 0.5f;
}

// Sutherland-Hodgman: clip subject polygon by one half-plane defined by edge (a -> b).
// Points on the left side of (a -> b) are kept.
std::vector<Vector2T> clip_by_edge(const std::vector<Vector2T>& subject, const Vector2T& a, const Vector2T& b) {
  if (subject.empty()) {
    return {};
  }

  auto side = [&](const Vector2T& p) -> float {
    return (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
  };

  auto intersect = [&](const Vector2T& p, const Vector2T& q) -> Vector2T {
    float sp = side(p);
    float sq = side(q);
    float denom = sp - sq;
    // Guard against degenerate cases (p == q, both points on the edge, NaN/denormal inputs).
    // Callers only invoke this for sign-straddling pairs, so denom should normally be non-zero.
    if (std::fabs(denom) < 1e-8f) {
      return Vector2T(0.5f * (p.x() + q.x()), 0.5f * (p.y() + q.y()));
    }
    float t = sp / denom;
    t = std::max(0.f, std::min(1.f, t));
    return Vector2T(p.x() + t * (q.x() - p.x()), p.y() + t * (q.y() - p.y()));
  };

  std::vector<Vector2T> out;
  out.reserve(subject.size() + 1);
  for (size_t i = 0; i < subject.size(); ++i) {
    const Vector2T& cur = subject[i];
    const Vector2T& prev = subject[(i + subject.size() - 1) % subject.size()];
    float s_cur = side(cur);
    float s_prev = side(prev);

    if (s_cur >= 0.f) {
      if (s_prev < 0.f) {
        out.push_back(intersect(prev, cur));
      }
      out.push_back(cur);
    } else if (s_prev >= 0.f) {
      out.push_back(intersect(prev, cur));
    }
  }
  return out;
}

// Sutherland-Hodgman polygon intersection
std::vector<Vector2T> polygon_intersection(const std::vector<Vector2T>& subject, const std::vector<Vector2T>& clip) {
  if (subject.size() < 3 || clip.size() < 3) {
    return {};
  }

  std::vector<Vector2T> result = subject;
  for (size_t i = 0; i < clip.size() && !result.empty(); ++i) {
    result = clip_by_edge(result, clip[i], clip[(i + 1) % clip.size()]);
  }
  return result;
}

// Adaptive RANSAC iteration count.
// k = ceil(log(1-p) / log(1-w^3)) where w = inlier ratio, p = desired confidence,
// and 3 is the minimum sample size (three points define a plane).
int ransac_required_iterations(float inlier_ratio, float confidence) {
  if (inlier_ratio <= 0.f) {
    return std::numeric_limits<int>::max();
  }
  if (inlier_ratio >= 1.f) {
    return 1;
  }
  const double w3 = static_cast<double>(inlier_ratio) * inlier_ratio * inlier_ratio;
  const double denom = std::log(1.0 - w3);
  if (std::abs(denom) < 1e-15) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(std::ceil(std::log(1.0 - confidence) / denom));
}

// Pack 2D grid indices into a single int64: upper 32 bits = ix, lower 32 bits = iy (unsigned).
int64_t pack_cell(int ix, int iy) { return (static_cast<int64_t>(ix) << 32) | static_cast<uint32_t>(iy); }

// Binary search for `key` in a sorted array of int64s; returns index or -1 if absent.
int find_sorted(const int64_t* keys, int n, int64_t key) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (lo < n && keys[lo] == key) ? lo : -1;
}

// Spatial density filter: projects inlier points onto the plane's 2D frame (u,v),
// bins them into a grid, discards sparse cells (< min_cell_count points), then
// extracts the largest 4-connected component of dense cells via BFS.
//
// Implementation note: the previous version used unordered_map / unordered_set / queue
// which dominated plane_map_update at keyframes (~3.7 ms per keyframe profiled).  The
// version below replaces all hash containers with a single radix-friendly sort over
// (cell_key, point_idx) pairs and reuses preallocated scratch buffers, which keeps the
// hot path allocation-free after the first call.
std::vector<int> filter_spatial_outliers(const std::vector<Vector3T>& inlier_pts, const Vector3T& centroid,
                                         const Vector3T& normal, float cell_size, int min_cell_count) {
  const int n = static_cast<int>(inlier_pts.size());
  if (n < 3 || cell_size <= 0.f) {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  Vector3T u, v;
  build_plane_frame(normal, u, v);

  const float inv_cell = 1.f / cell_size;

  // Bin points into grid cells: pair (cell_key, original_point_index).  Sorting groups
  // points belonging to the same cell into contiguous runs without any hashing.
  std::vector<std::pair<int64_t, int>> cell_pt(n);
  for (int i = 0; i < n; ++i) {
    const Vector3T d = inlier_pts[i] - centroid;
    const int ix = static_cast<int>(std::floor(d.dot(u) * inv_cell));
    const int iy = static_cast<int>(std::floor(d.dot(v) * inv_cell));
    cell_pt[i] = {pack_cell(ix, iy), i};
  }
  std::sort(cell_pt.begin(), cell_pt.end(),
            [](const std::pair<int64_t, int>& a, const std::pair<int64_t, int>& b) { return a.first < b.first; });

  // Run-length encode: collect every dense cell as (key, run_start_in_cell_pt, run_len).
  // Sparse cells are skipped here -- they never participate in BFS.
  struct DenseCell {
    int64_t key;
    int start;
    int len;
  };
  std::vector<DenseCell> dense;
  dense.reserve(n);

  int run_start = 0;
  while (run_start < n) {
    int run_end = run_start + 1;
    while (run_end < n && cell_pt[run_end].first == cell_pt[run_start].first) {
      ++run_end;
    }
    const int run_len = run_end - run_start;
    if (run_len >= min_cell_count) {
      dense.push_back({cell_pt[run_start].first, run_start, run_len});
    }
    run_start = run_end;
  }

  if (dense.empty()) {
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  // dense is already sorted by key (since cell_pt was sorted), so neighbor lookups
  // become a binary search instead of a hash probe.  We also reuse a small flat
  // 'visited' bitmap addressed by dense-array index, not by cell key.
  const int num_dense = static_cast<int>(dense.size());
  std::vector<int64_t> dense_keys(num_dense);
  for (int i = 0; i < num_dense; ++i) {
    dense_keys[i] = dense[i].key;
  }

  std::vector<uint8_t> visited(num_dense, 0);
  std::vector<int> bfs_stack;
  bfs_stack.reserve(num_dense);
  std::vector<int> component_dense_idx;
  component_dense_idx.reserve(num_dense);

  // Track only point counts during BFS; materialize the winning component once at the end.
  int best_size = 0;
  std::vector<int> best_component_dense_idx;
  best_component_dense_idx.reserve(num_dense);

  for (int seed = 0; seed < num_dense; ++seed) {
    if (visited[seed]) {
      continue;
    }
    bfs_stack.clear();
    component_dense_idx.clear();

    bfs_stack.push_back(seed);
    visited[seed] = 1;
    int component_size = 0;

    while (!bfs_stack.empty()) {
      const int cur = bfs_stack.back();
      bfs_stack.pop_back();
      component_dense_idx.push_back(cur);
      component_size += dense[cur].len;

      const int64_t cur_key = dense_keys[cur];
      const int cx = static_cast<int>(cur_key >> 32);
      const int cy = static_cast<int>(cur_key & 0xFFFFFFFF);
      const int64_t nbs[4] = {pack_cell(cx - 1, cy), pack_cell(cx + 1, cy), pack_cell(cx, cy - 1),
                              pack_cell(cx, cy + 1)};
      for (int k = 0; k < 4; ++k) {
        const int idx = find_sorted(dense_keys.data(), num_dense, nbs[k]);
        if (idx >= 0 && !visited[idx]) {
          visited[idx] = 1;
          bfs_stack.push_back(idx);
        }
      }
    }

    if (component_size > best_size) {
      best_size = component_size;
      best_component_dense_idx = component_dense_idx;
    }
  }

  // Materialise winning component's point ids and sort to match the previous output order.
  std::vector<int> result;
  result.reserve(best_size);
  for (int di : best_component_dense_idx) {
    const DenseCell& dc = dense[di];
    for (int k = 0; k < dc.len; ++k) {
      result.push_back(cell_pt[dc.start + k].second);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

// Reconstruct centroid and covariance from GPU-accumulated sums.
// stats layout: [sum_x, sum_y, sum_z, sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz]
// cov_ij = sum_ij/N - mean_i * mean_j  (unbiased scatter matrix)
void covariance_from_gpu_stats(const float* stats, int num_points, Vector3T& centroid, Matrix3T& covariance) {
  const float inv_n = 1.f / static_cast<float>(num_points);
  centroid = Vector3T(stats[0] * inv_n, stats[1] * inv_n, stats[2] * inv_n);

  covariance(0, 0) = stats[3] * inv_n - centroid.x() * centroid.x();
  covariance(0, 1) = covariance(1, 0) = stats[4] * inv_n - centroid.x() * centroid.y();
  covariance(0, 2) = covariance(2, 0) = stats[5] * inv_n - centroid.x() * centroid.z();
  covariance(1, 1) = stats[6] * inv_n - centroid.y() * centroid.y();
  covariance(1, 2) = covariance(2, 1) = stats[7] * inv_n - centroid.y() * centroid.z();
  covariance(2, 2) = stats[8] * inv_n - centroid.z() * centroid.z();
}

}  // namespace

PlaneMap::PlaneMap(const PlaneMapSettings& settings) : settings_(settings) {}

void PlaneMap::update_at_keyframe(const std::vector<DepthCameraInfo>& depth_cameras) {
  TRACE_EVENT ev = profiler_domain_.trace_event("update_at_keyframe");

  {
    TRACE_EVENT evict_ev = profiler_domain_.trace_event("evict_planes");
    const float max_dist_sq = settings_.max_plane_dist * settings_.max_plane_dist;
    planes_.erase(std::remove_if(planes_.begin(), planes_.end(),
                                 [&](const Plane& p) {
                                   for (const auto& cam : depth_cameras) {
                                     Vector3T cam_pos = cam.world_from_cam.translation();
                                     if ((p.centroid - cam_pos).squaredNorm() <= max_dist_sq) {
                                       return false;
                                     }
                                   }
                                   return true;
                                 }),
                  planes_.end());
  }

  std::vector<Plane> detected;
  {
    TRACE_EVENT detect_ev = profiler_domain_.trace_event("detect_planes");
    detected = detect_planes(depth_cameras);
  }

  {
    TRACE_EVENT merge_ev = profiler_domain_.trace_event("merge_planes");
    merge_planes(detected);
  }
}

const std::vector<Plane>& PlaneMap::get_planes() const { return planes_; }

std::vector<SurfacePoint> PlaneMap::get_plane_surface_points() const {
  std::vector<SurfacePoint> out;
  out.reserve(planes_.size());
  for (const auto& p : planes_) {
    out.push_back({p.centroid, p.normal});
  }
  return out;
}

void PlaneMap::clear() { planes_.clear(); }

// ─── RANSAC plane detection (per camera) ──────────────────────────────────────

std::vector<Plane> PlaneMap::detect_planes_single_camera(const DepthCameraInfo& cam) const {
  cudaStream_t s = stream_.get_stream();

  // Stage 1: Unproject depth to 3D point cloud on GPU.
  // The d_count_ counter is zeroed device-side via cudaMemsetAsync so we don't pay
  // a host->device 4-byte copy + sync just to reset it.  The single sync below is
  // unavoidable: we need total_points on the host to size all subsequent loops.
  int total_points = 0;
  {
    TRACE_EVENT unproject_ev = profiler_domain_.trace_event("unproject");
    CUDA_CHECK(cudaMemsetAsync(d_count_.ptr(), 0, sizeof(int), s));

    CUDA_CHECK(cuda::unproject_depth_points(cam.depth_tex, cam.focal, cam.principal, cam.image_size,
                                            settings_.depth_min, settings_.depth_max, d_points_.ptr(), d_count_.ptr(),
                                            kMaxOutputPoints, settings_.unproject_stride, s));

    d_count_.copy(cuda::ToCPU, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    total_points = std::min(d_count_[0], kMaxOutputPoints);
  }
  if (total_points < settings_.min_plane_inliers) {
    return {};
  }

  // Build initial active index set entirely on GPU.  Previously this was a host
  // iota over up to 76800 entries followed by a memcpy + sync; replaced with a
  // single fill_iota kernel chained on the same stream (no sync needed -- the
  // first downstream consumer is a kernel that runs on the same stream).
  {
    TRACE_EVENT build_active_ev = profiler_domain_.trace_event("build_active_set");
    CUDA_CHECK(cuda::fill_iota(d_active_indices_.ptr(), total_points, s));
  }

  // We use a double-buffer for active indices to avoid GPU self-copy
  int* d_active_cur = d_active_indices_.ptr();
  int* d_active_next = d_active_indices_b_.ptr();

  int num_active = total_points;
  std::mt19937 rng(42);

  std::vector<Plane> result;

  // Stage 2: Iterative RANSAC -- extract planes largest-first
  for (int plane_iter = 0; plane_iter < settings_.max_planes && num_active >= settings_.min_plane_inliers;
       ++plane_iter) {
    TRACE_EVENT plane_iter_ev = profiler_domain_.trace_event("plane_iter");
    float3 best_normal = {0, 0, 0};
    float best_d = 0.f;
    int best_inlier_count = 0;
    int adaptive_max = settings_.ransac_max_iterations;

    for (int iter = 0; iter < adaptive_max;) {
      TRACE_EVENT ransac_batch_ev = profiler_domain_.trace_event("ransac_batch");
      const int batch = std::min(kRansacBatchSize, adaptive_max - iter);

      // Generate random triplet indices on CPU, upload to GPU for hypothesis generation
      {
        TRACE_EVENT triplets_ev = profiler_domain_.trace_event("triplets_gen_upload");
        std::uniform_int_distribution<int> dist(0, num_active - 1);
        for (int b = 0; b < batch; ++b) {
          int i0 = dist(rng);
          int i1 = dist(rng);
          while (i1 == i0) {
            i1 = dist(rng);
          }
          int i2 = dist(rng);
          while (i2 == i0 || i2 == i1) {
            i2 = dist(rng);
          }
          d_triplets_[b] = {i0, i1, i2};
        }
        d_triplets_.copy_top_n(cuda::ToGPU, batch, s);
      }

      {
        TRACE_EVENT eval_ev = profiler_domain_.trace_event("ransac_eval_gpu");
        CUDA_CHECK(
            cuda::generate_hypotheses(d_points_.ptr(), d_active_cur, d_triplets_.ptr(), d_hypotheses_.ptr(), batch, s));

        CUDA_CHECK(cuda::ransac_batch_evaluate(d_points_.ptr(), d_active_cur, num_active, d_hypotheses_.ptr(),
                                               settings_.inlier_thresh, d_hyp_counts_.ptr(), batch, s));
      }

      {
        TRACE_EVENT d2h_ev = profiler_domain_.trace_event("ransac_d2h_sync");
        d_hyp_counts_.copy_top_n(cuda::ToCPU, batch, s);
        d_hypotheses_.copy_top_n(cuda::ToCPU, batch, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
      }

      {
        TRACE_EVENT pick_ev = profiler_domain_.trace_event("ransac_pick_best");
        for (int b = 0; b < batch; ++b) {
          const float4& h = d_hypotheses_[b];
          if (h.x == 0.f && h.y == 0.f && h.z == 0.f) {
            continue;
          }
          const int inlier_count = d_hyp_counts_[b];
          if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_normal = {h.x, h.y, h.z};
            best_d = h.w;

            const float w = static_cast<float>(inlier_count) / static_cast<float>(num_active);
            adaptive_max =
                std::min(settings_.ransac_max_iterations, ransac_required_iterations(w, settings_.ransac_confidence));
          }
        }
      }

      iter += batch;
    }

    if (best_inlier_count < settings_.min_plane_inliers) {
      break;
    }

    // Gather inlier indices on GPU.
    // We do NOT read d_count_ back here: count_plane_inliers is called with the same
    // (best_normal, best_d, inlier_thresh) used by RANSAC's evaluate kernel, so it
    // produces exactly best_inlier_count inliers (clamped to kMaxOutputPoints).  Skipping
    // the device->host counter copy lets us merge this stage's sync with the next one,
    // saving a stream round-trip per plane_iter.
    Vector3T centroid_cam;
    const int gathered = std::min(best_inlier_count, kMaxOutputPoints);
    {
      TRACE_EVENT count_init_ev = profiler_domain_.trace_event("count_inliers_initial");
      CUDA_CHECK(cudaMemsetAsync(d_count_.ptr(), 0, sizeof(int), s));

      CUDA_CHECK(cuda::count_plane_inliers(d_points_.ptr(), d_active_cur, num_active, best_normal, best_d,
                                           settings_.inlier_thresh, d_count_.ptr(), d_inlier_indices_.ptr(),
                                           /*gather_positions=*/nullptr, kMaxOutputPoints, s));

      // Compute centroid + covariance on GPU (no D2H of inlier points)
      CUDA_CHECK(cuda::compute_plane_stats(d_points_.ptr(), d_inlier_indices_.ptr(), gathered, d_stats_.ptr(), s));

      d_stats_.copy(cuda::ToCPU, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
    }

    Matrix3T cov;
    covariance_from_gpu_stats(&d_stats_[0], gathered, centroid_cam, cov);

    Vector3T refined_normal;
    {
      TRACE_EVENT pca_ev = profiler_domain_.trace_event("pca_refine");
      Vector3T eigenvalues;
      Matrix3T eigenvectors;
      eigen_symmetric_3x3(cov, eigenvalues, eigenvectors);

      refined_normal = eigenvectors.col(0);

      if (refined_normal.dot(centroid_cam) > 0) {
        refined_normal = -refined_normal;
      }
    }

    // Plane equation: n . x + d = 0, so d = -(n . centroid)
    // Re-count inliers against the refined plane on GPU
    const float refined_d = -refined_normal.dot(centroid_cam);
    const float3 refined_n_f3 = {static_cast<float>(refined_normal.x()), static_cast<float>(refined_normal.y()),
                                 static_cast<float>(refined_normal.z())};

    int refined_count = 0;
    {
      TRACE_EVENT count_ref_ev = profiler_domain_.trace_event("count_inliers_refined");
      CUDA_CHECK(cudaMemsetAsync(d_count_.ptr(), 0, sizeof(int), s));
      // Extended count_plane_inliers: writes both point ids (d_inlier_indices_) and
      // active-set positions (d_inlier_positions_) in lockstep, so the downstream
      // compact_active step can scatter removal flags without a CPU reverse map.
      CUDA_CHECK(cuda::count_plane_inliers(d_points_.ptr(), d_active_cur, num_active, refined_n_f3,
                                           static_cast<float>(refined_d), settings_.inlier_thresh, d_count_.ptr(),
                                           d_inlier_indices_.ptr(), d_inlier_positions_.ptr(), kMaxOutputPoints, s));
      d_count_.copy(cuda::ToCPU, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      refined_count = std::min(d_count_[0], kMaxOutputPoints);
    }

    if (refined_count < settings_.min_plane_inliers) {
      break;
    }

    // Stride-subsample the refined inliers down to a fixed cap before any DtoH.
    // The geometry stages (spatial_filter, convex_hull) only need representative
    // coverage; compaction below still uses the full refined set so we never
    // strand inliers in the active list.
    const int n_geom = std::min(refined_count, settings_.max_inliers_for_geometry);
    const int stride = std::max(1, refined_count / n_geom);

    std::vector<Vector3T> inlier_pts(n_geom);
    {
      TRACE_EVENT dl_ev = profiler_domain_.trace_event("download_inliers");
      CUDA_CHECK(cuda::stride_subsample_indices(d_inlier_indices_.ptr(), d_inlier_indices_geom_.ptr(), stride, n_geom,
                                                refined_count, s));
      CUDA_CHECK(cuda::gather_inlier_points(d_points_.ptr(), d_inlier_indices_geom_.ptr(), n_geom,
                                            d_inlier_pts_compact_.ptr(), s));
      d_inlier_pts_compact_.copy_top_n(cuda::ToCPU, n_geom, s);
      CUDA_CHECK(cudaStreamSynchronize(s));

      for (int i = 0; i < n_geom; ++i) {
        const float3& p = d_inlier_pts_compact_[i];
        inlier_pts[i] = Vector3T(p.x, p.y, p.z);
      }
    }

    // Spatial density filter on the (already capped) inlier subsample.  The
    // hard "min_plane_inliers" robustness check was already applied to the full
    // refined_count above; here we only require enough surviving points to
    // build a meaningful convex hull (at least 3).
    //
    // The filter's grid_cell_size + min_cell_count thresholds were tuned for
    // the full unsubsampled inlier set.  When we subsample by `stride`, the
    // expected count per cell drops by the same factor, which would erase
    // every cell on the plane.  Compensate by enlarging the cell area by
    // `stride` (side length scales by sqrt(stride)), so the per-cell density
    // assumption is preserved.
    const float density_scale = std::sqrt(static_cast<float>(stride));
    const float effective_cell_size = settings_.grid_cell_size * density_scale;
    {
      TRACE_EVENT spatial_ev = profiler_domain_.trace_event("spatial_filter");
      auto keep = filter_spatial_outliers(inlier_pts, centroid_cam, refined_normal, effective_cell_size,
                                          settings_.min_cell_count);
      if (keep.size() < inlier_pts.size()) {
        std::vector<Vector3T> filtered_pts;
        filtered_pts.reserve(keep.size());
        for (int ki : keep) {
          filtered_pts.push_back(inlier_pts[ki]);
        }
        inlier_pts = std::move(filtered_pts);
      }
      if (inlier_pts.size() < 3) {
        break;
      }
    }

    const Isometry3T& w_from_c = cam.world_from_cam;
    const Matrix3T R = w_from_c.linear();

    centroid_cam = Vector3T::Zero();
    for (const auto& pt : inlier_pts) {
      centroid_cam += pt;
    }
    centroid_cam /= static_cast<float>(inlier_pts.size());

    Plane plane;
    plane.centroid = w_from_c * centroid_cam;
    plane.normal = (R * refined_normal).normalized();
    // Report the full refined inlier count, not the subsampled-and-filtered one,
    // since compaction below removes the full set from the active pool.
    plane.num_inliers = refined_count;

    {
      TRACE_EVENT hull_ev = profiler_domain_.trace_event("convex_hull");
      // Compute the convex hull in the camera frame on the inlier_pts we already have,
      // then transform only the (small, typically O(10) vertices) hull to world.
      // The previous version transformed all ~1500 inlier points into world before the
      // hull, paying ~9 mul + 6 add per point for nothing -- the hull topology is
      // invariant under the rigid transform.
      auto hull_cam = compute_convex_hull(inlier_pts, refined_normal, centroid_cam);

      plane.convex_hull.resize(hull_cam.size());
      for (size_t i = 0; i < hull_cam.size(); ++i) {
        plane.convex_hull[i] = w_from_c * hull_cam[i];
      }

      const float area = convex_hull_area(plane.convex_hull);
      if (area >= settings_.min_plane_area) {
        result.push_back(std::move(plane));
      }
    }

    // Fully GPU-side active-set compaction.  The full refined inlier set (not just
    // the spatially-filtered subset) is removed: any sparse-cell or secondary-component
    // inliers that the spatial filter dropped from this plane's geometry would
    // otherwise re-seed the same plane on the next plane_iter.  inter-camera NMS and
    // the existing merge_planes path already absorb any genuinely-missed fragments.
    //
    // Sync-free fast path: count_plane_inliers writes `refined_count` unique active-set
    // positions into d_inlier_positions_ (each thread writes its own tid exactly once),
    // so the compaction is guaranteed to remove exactly that many entries.  We can
    // therefore update num_active on the host without a device->host counter readback
    // and let the next plane_iter's RANSAC kernel pick up the freshly compacted indices
    // through the same stream's implicit ordering.  This eliminated round-trip is the
    // single biggest win for plane_map_update at keyframes.
    {
      TRACE_EVENT compact_ev = profiler_domain_.trace_event("compact_active");
      CUDA_CHECK(cudaMemsetAsync(d_keep_flags_.ptr(), 1, sizeof(uint8_t) * num_active, s));
      CUDA_CHECK(cudaMemsetAsync(d_count_.ptr(), 0, sizeof(int), s));
      CUDA_CHECK(cuda::mark_positions_for_removal(d_keep_flags_.ptr(), d_inlier_positions_.ptr(), refined_count, s));
      CUDA_CHECK(cuda::compact_active_indices(d_active_cur, d_keep_flags_.ptr(), num_active, d_active_next,
                                              d_count_.ptr(), s));

      num_active -= refined_count;
      std::swap(d_active_cur, d_active_next);
    }
  }

  // Intra-camera NMS removed: any duplicate planes from a single camera are also
  // caught by the inter-camera suppress_redundant pass in detect_planes().
  return result;
}

std::vector<Plane> PlaneMap::detect_planes(const std::vector<DepthCameraInfo>& depth_cameras) const {
  std::vector<Plane> all_planes;

  for (size_t ci = 0; ci < depth_cameras.size(); ++ci) {
    TRACE_EVENT cam_ev = profiler_domain_.trace_event("cam");
    auto cam_planes = detect_planes_single_camera(depth_cameras[ci]);
    std::move(cam_planes.begin(), cam_planes.end(), std::back_inserter(all_planes));
  }

  TRACE_EVENT nms_ev = profiler_domain_.trace_event("inter_cam_nms");
  auto merged = suppress_redundant(all_planes);
  return merged;
}

// ─── Plane merging into the persistent map ────────────────────────────────────

std::optional<size_t> PlaneMap::find_best_match(const Plane& new_plane) const {
  for (size_t i = 0; i < planes_.size(); ++i) {
    const float normal_compat = std::abs(planes_[i].normal.dot(new_plane.normal));
    if (normal_compat < settings_.merge_normal_thresh) {
      continue;
    }
    const float dist = std::abs(planes_[i].normal.dot(new_plane.centroid - planes_[i].centroid));
    if (dist > settings_.merge_dist_thresh) {
      continue;
    }
    return i;
  }
  return std::nullopt;
}

void PlaneMap::merge_planes(const std::vector<Plane>& detected) {
  for (size_t di = 0; di < detected.size(); ++di) {
    const auto& new_plane = detected[di];
    auto match = find_best_match(new_plane);

    if (match) {
      merge_two_planes(planes_[*match], new_plane);
    } else if (static_cast<int>(planes_.size()) < settings_.max_planes) {
      planes_.push_back(new_plane);
    } else {
      auto worst = std::min_element(planes_.begin(), planes_.end(),
                                    [](const Plane& a, const Plane& b) { return a.num_inliers < b.num_inliers; });
      if (worst != planes_.end() && new_plane.num_inliers > worst->num_inliers) {
        *worst = new_plane;
      }
    }
  }

  planes_ = suppress_redundant(planes_);
}

// Merge source plane into target using exponential smoothing for the centroid
// (alpha = 0.2 blends toward the source) and inlier-count-weighted averaging
// for the normal. The convex hull is recomputed from the union of hull vertices.
void PlaneMap::merge_two_planes(Plane& target, const Plane& source) {
  const float w_t = static_cast<float>(target.num_inliers);
  const float w_s = static_cast<float>(source.num_inliers);
  const float w_total = w_t + w_s;

  const float alpha = 0.2f;
  target.centroid = target.centroid * alpha + source.centroid * (1.f - alpha);
  Vector3T merged_n = (target.normal * w_t + source.normal * w_s) / w_total;
  target.normal = merged_n.normalized();
  target.num_inliers = static_cast<int>(w_total);

  std::vector<Vector3T> all_hull_pts;
  all_hull_pts.reserve(target.convex_hull.size() + source.convex_hull.size());
  std::copy(target.convex_hull.begin(), target.convex_hull.end(), std::back_inserter(all_hull_pts));
  std::copy(source.convex_hull.begin(), source.convex_hull.end(), std::back_inserter(all_hull_pts));
  target.convex_hull = compute_convex_hull(all_hull_pts, target.normal, target.centroid);
}

// ─── Convex hull utilities ────────────────────────────────────────────────────

std::vector<Vector3T> PlaneMap::compute_convex_hull(const std::vector<Vector3T>& points, const Vector3T& normal,
                                                    const Vector3T& centroid) {
  if (points.size() < 3) {
    return points;
  }

  Vector3T u, v;
  build_plane_frame(normal, u, v);

  struct Pt2 {
    Vector2T xy;
    int idx;
  };
  std::vector<Pt2> pts2d(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const Vector3T d = points[i] - centroid;
    pts2d[i] = {Vector2T(d.dot(u), d.dot(v)), static_cast<int>(i)};
  }

  // Andrew's monotone chain
  std::sort(pts2d.begin(), pts2d.end(), [](const Pt2& a, const Pt2& b) {
    return a.xy.x() < b.xy.x() || (a.xy.x() == b.xy.x() && a.xy.y() < b.xy.y());
  });

  auto cross2d = [](const Pt2& O, const Pt2& A, const Pt2& B) -> float {
    return (A.xy.x() - O.xy.x()) * (B.xy.y() - O.xy.y()) - (A.xy.y() - O.xy.y()) * (B.xy.x() - O.xy.x());
  };

  const int n = static_cast<int>(pts2d.size());
  std::vector<int> hull_idx(2 * n);
  int k = 0;

  for (int i = 0; i < n; ++i) {
    while (k >= 2 && cross2d(pts2d[hull_idx[k - 2]], pts2d[hull_idx[k - 1]], pts2d[i]) <= 0) {
      --k;
    }
    hull_idx[k++] = i;
  }

  for (int i = n - 2, t = k + 1; i >= 0; --i) {
    while (k >= t && cross2d(pts2d[hull_idx[k - 2]], pts2d[hull_idx[k - 1]], pts2d[i]) <= 0) {
      --k;
    }
    hull_idx[k++] = i;
  }
  hull_idx.resize(k - 1);

  std::vector<Vector3T> hull;
  hull.reserve(hull_idx.size());
  for (int hi : hull_idx) {
    hull.push_back(points[pts2d[hi].idx]);
  }
  return hull;
}

// Area of a coplanar 3D polygon via triangle-fan cross products.
// For ordered hull vertices p_0..p_{n-1}: A = 0.5 * sum_i ||(p_i - p_0) x (p_{i+1} - p_0)||
float PlaneMap::convex_hull_area(const std::vector<Vector3T>& hull) {
  if (hull.size() < 3) {
    return 0.f;
  }

  float area = 0.f;
  const Vector3T& p0 = hull[0];
  for (size_t i = 1; i + 1 < hull.size(); ++i) {
    area += (hull[i] - p0).cross(hull[i + 1] - p0).norm();
  }
  return area * 0.5f;
}

// ─── Plane NMS utilities ──────────────────────────────────────────────────────

// Compute overlap between two convex hulls projected onto a common 2D plane.
// Returns max(IoU, IoMin) where IoU = intersection/union and IoMin = intersection/min_area.
// The 1e-8f epsilon prevents division by zero for degenerate polygons.
float PlaneMap::hull_overlap_ratio(const Plane& a, const Plane& b) {
  if (a.convex_hull.size() < 3 || b.convex_hull.size() < 3) {
    return 0.f;
  }

  Vector3T n_avg = (a.normal + b.normal).normalized();
  Vector3T u, v;
  build_plane_frame(n_avg, u, v);

  Vector3T origin = (a.centroid + b.centroid) * 0.5f;

  auto poly_a = project_hull_2d(a.convex_hull, u, v, origin);
  auto poly_b = project_hull_2d(b.convex_hull, u, v, origin);

  auto intersection = polygon_intersection(poly_a, poly_b);

  float area_inter = polygon_area_2d(intersection);
  float area_a = polygon_area_2d(poly_a);
  float area_b = polygon_area_2d(poly_b);

  float union_area = area_a + area_b - area_inter;
  float iou = (union_area > 1e-8f) ? area_inter / union_area : 0.f;

  float min_area = std::min(area_a, area_b);
  float iomin = (min_area > 1e-8f) ? area_inter / min_area : 0.f;

  return std::max(iou, iomin);
}

std::vector<Plane> PlaneMap::suppress_redundant(std::vector<Plane>& planes) const {
  if (planes.size() <= 1) {
    return std::move(planes);
  }

  std::sort(planes.begin(), planes.end(), [](const Plane& a, const Plane& b) { return a.num_inliers > b.num_inliers; });

  std::vector<bool> suppressed(planes.size(), false);

  for (size_t i = 0; i < planes.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    for (size_t j = i + 1; j < planes.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      const float normal_compat = std::abs(planes[i].normal.dot(planes[j].normal));
      if (normal_compat < settings_.merge_normal_thresh) {
        continue;
      }

      const float dist = std::abs(planes[i].normal.dot(planes[j].centroid - planes[i].centroid));
      if (dist > settings_.merge_dist_thresh) {
        continue;
      }

      const float overlap = hull_overlap_ratio(planes[i], planes[j]);
      if (overlap < settings_.nms_overlap_thresh) {
        continue;
      }
      merge_two_planes(planes[i], planes[j]);
      suppressed[j] = true;
    }
  }
  std::vector<Plane> result;
  result.reserve(planes.size());
  for (size_t i = 0; i < planes.size(); ++i) {
    if (!suppressed[i]) {
      result.push_back(std::move(planes[i]));
    }
  }
  return result;
}

}  // namespace cuvslam::map
