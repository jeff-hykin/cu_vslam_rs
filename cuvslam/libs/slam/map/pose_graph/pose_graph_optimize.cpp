
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

#include <exception>

#include "common/log_types.h"
#include "common/stopwatch.h"
#include "math/twist.h"
#include "profiler/profiler.h"

#include "slam/map/pose_graph/pose_graph.h"

namespace cuvslam::slam {
using namespace cuvslam::math;

bool PoseGraph::OptimizeSubgraph(const std::vector<KeyFrameId>& keyframes_to_optimize,
                                 const std::vector<EdgeId>& edges_to_optimize,
                                 const std::vector<KeyFrameId>& constraint_keyframes,
                                 const PoseGraphHypothesis& pose_graph_hypothesis_src, bool planar_constraint,
                                 PoseGraphHypothesis& pose_graph_hypothesis_dst, Isometry3T& vo_to_head) const {
  TRACE_EVENT te_pgo = profiler_domain_.trace_event("PGO", profiler_color_);
  vo_to_head = Isometry3T::Identity();

  pose_graph_hypothesis_src.CopyTo(pose_graph_hypothesis_dst);
  Stopwatch sw_full;
  StopwatchScope ssw_full(sw_full);

  if (edges_to_optimize.empty()) {
    return false;
  }
  if (keyframes_to_optimize.empty()) {
    return false;
  }

  // fill remap & initial
  TRACE_EVENT te_fill = profiler_domain_.trace_event("Fill graph", profiler_color_);

  inputs_.clear();

  inputs_.poses.reserve(keyframes_to_optimize.size());
  std::unordered_map<int, KeyFrameId> pose_to_kf;
  std::unordered_map<KeyFrameId, int> kf_to_pose;

  for (KeyFrameId keyframe_id : keyframes_to_optimize) {
    const auto keyframe_pose = pose_graph_hypothesis_src.GetKeyframePose(keyframe_id);
    if (!keyframe_pose) {
      return false;
    }
    // copy existing keyframes
    pose_graph_hypothesis_dst.SetKeyframePose(keyframe_id, *keyframe_pose);

    // don't add standalone nodes
    int edges_count = 0;
    QueryKeyframeEdges(keyframe_id, [&](KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&) { edges_count++; });
    if (edges_count == 0) {
      continue;
    }

    auto m = *keyframe_pose;

    pose_to_kf[inputs_.poses.size()] = keyframe_id;
    kf_to_pose[keyframe_id] = inputs_.poses.size();
    inputs_.poses.push_back(m);
  }

  if (planar_constraint) {
    inputs_.use_planar_constraint = true;
    inputs_.plane_normal = {0, 1.f, 0, 0};  // y-axis is the normal for the plane
    inputs_.planar_weight = 1e3;
  }

  // constraint first node
  for (KeyFrameId constrainedKey : constraint_keyframes) {
    // check if constrainedKey exists
    auto it_initial = kf_to_pose.find(constrainedKey);
    if (it_initial == kf_to_pose.end()) {
      SlamStderr("Initial pose for constrained key %lu not found.\n", constrainedKey);
    } else {
      inputs_.constrained_pose_ids.insert(it_initial->second);
    }
  }

  inputs_.robustifier = 0.5f;

  inputs_.deltas.reserve(edges_to_optimize.size());
  for (EdgeId edge_id : edges_to_optimize) {
    auto& edge = edges_.at(edge_id);
    auto& m = edge.from_to;
    auto& cov = edge.from_to_covariance;

    if (kf_to_pose.find(edge.from_keyframe) == kf_to_pose.end()) {
      SlamStderr("Initial pose for keyframe %zd not found.\n", edge.from_keyframe);
      continue;
    }
    if (kf_to_pose.find(edge.to_keyframe) == kf_to_pose.end()) {
      SlamStderr("Initial pose for keyframe %zd not found.\n", edge.to_keyframe);
      continue;
    }

    const int p1id = kf_to_pose[edge.from_keyframe];
    const int p2id = kf_to_pose[edge.to_keyframe];

    inputs_.deltas.push_back({p1id, p2id, m, cov.ldlt().solve(Matrix6T::Identity())});
  }
  te_fill.Pop();

  try {
    TRACE_EVENT te_opt = profiler_domain_.trace_event("optimizer.optimize()", profiler_color_);
    if (!pgo.run(inputs_, 10)) {
      TraceError("PoseGraph optimization failed.");
      return false;
    }
    te_opt.Pop();
  } catch (const std::exception& e) {
    TraceError("PoseGraph optimization crashed: %s\n", e.what());
    return false;
  } catch (...) {
    TraceError("PoseGraph optimization crashed (unknown exception).");
    return false;
  }

  TRACE_EVENT te_store = profiler_domain_.trace_event("store result", profiler_color_);
  KeyFrameId head_keyframe;
  GetHeadKeyframe(head_keyframe);

  for (size_t i = 0; i < inputs_.poses.size(); i++) {
    const Isometry3T& m = inputs_.poses[i];
    const auto keyframe_id = pose_to_kf[i];

    const auto keyframe_pose = pose_graph_hypothesis_src.GetKeyframePose(keyframe_id);
    if (!keyframe_pose) {
      return false;
    }

    if (head_keyframe == keyframe_id) {
      // VO correction: new_keyframe_pose = keyframe_pose * vo_to_head

      /*/
      Isometry3T vo_to_head_draft = keyframe_pose->inverse() * m;
      Matrix3T mat_rotation, mat_scaling;
      vo_to_head_draft.computeRotationScaling(&mat_rotation, &mat_scaling);

      vo_to_head.translate(vo_to_head_draft.translation());
      vo_to_head.rotate(mat_rotation);
      //*/

      vo_to_head = keyframe_pose->inverse() * m;
      RemoveScaleFromTransform(vo_to_head);
    }
    // update pose in hypothesis
    if (&pose_graph_hypothesis_dst != &pose_graph_hypothesis_src) {
      pose_graph_hypothesis_dst.SetKeyframePose(keyframe_id, m);
    }
  }
  te_store.Pop();
  ssw_full.Stop();

  return true;
}

// select which nodes/edges to optimize with what constraint and run optimization
bool PoseGraph::Optimize(const PoseGraphHypothesis& pose_graph_hypothesis_src,
                         PoseGraphHypothesis& pose_graph_hypothesis_dst, bool planar_constraint,
                         Isometry3T& vo_to_head) const {
  keyframes_to_optimize_.clear();
  keyframes_to_optimize_.reserve(keyframes_.size());
  for (const auto& it : keyframes_) {
    keyframes_to_optimize_.push_back(it.first);
  }

  edges_to_optimize_.clear();
  edges_to_optimize_.reserve(edges_.size());
  for (const auto& it : edges_) {
    edges_to_optimize_.push_back(it.first);
  }

  constrained_keyframes_.clear();
  KeyFrameId min_keyframe_id = InvalidKeyFrameId;
  for (const KeyFrameId keyframe_id : keyframes_to_optimize_) {
    if (min_keyframe_id == InvalidKeyFrameId || keyframe_id < min_keyframe_id) {
      min_keyframe_id = keyframe_id;
    }
  }
  if (min_keyframe_id == InvalidKeyFrameId) {
    return false;  // nothing to optimize
  }
  constrained_keyframes_.push_back(min_keyframe_id);

  return OptimizeSubgraph(keyframes_to_optimize_, edges_to_optimize_, constrained_keyframes_, pose_graph_hypothesis_src,
                          planar_constraint, pose_graph_hypothesis_dst, vo_to_head);
}

}  // namespace cuvslam::slam
