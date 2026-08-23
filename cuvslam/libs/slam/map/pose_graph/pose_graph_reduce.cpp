
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

#include "common/log_types.h"
#include "math/pgo.h"
#include "math/twist.h"

#include "slam/map/pose_graph/pose_graph.h"

namespace cuvslam::slam {

using namespace cuvslam::math;

KeyFrameId PoseGraph::GetSmallestVarianceEdgeId() const {
  EdgeId min_edge_id = InvalidEdgeId;
  float min_value = std::numeric_limits<float>::max();

  KeyFrameId head_keyframe;
  GetHeadKeyframe(head_keyframe);

  for (auto it = edges_.begin(); it != edges_.end(); ++it) {
    const EdgeId edge_id = it->first;
    const Edge& edge = it->second;

    // check edges from to_keyframe
    int to_keyframe_edges = 0;
    QueryKeyframeEdges(edge.to_keyframe,
                       [&](KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&) { to_keyframe_edges++; });
    if (to_keyframe_edges < 2) {
      continue;
    }
    if (head_keyframe == edge.to_keyframe || head_keyframe == edge.from_keyframe) {
      continue;
    }

    Vector3T variance_xyz(std::abs(edge.from_to_covariance(3, 3)), std::abs(edge.from_to_covariance(4, 4)),
                          std::abs(edge.from_to_covariance(5, 5)));
    const float v = sqrt(variance_xyz[0] + variance_xyz[1] + variance_xyz[2]);
    if (v < min_value) {
      // is in latest_keyframes_
      auto it_latest_keyframes = std::find(latest_keyframes_.begin(), latest_keyframes_.end(), edge.to_keyframe);
      if (it_latest_keyframes != latest_keyframes_.end()) {
        continue;
      }
      min_value = v;
      min_edge_id = edge_id;
    }
  }
  return min_edge_id;
}

bool PoseGraph::ReduceSingleEdge(EdgeId edge_id, const PoseGraphHypothesis& pose_graph_hypothesis,
                                 const std::function<OnUpdateLandmarkRelation>& func_update_landmark_relation) {
  const auto it = edges_.find(edge_id);
  if (it == edges_.end()) {
    return false;
  }
  const auto& edge = it->second;

  log::Message<LogFrames>(log::kInfo, "ReduceSingleEdge(%zd, %zd)", edge.from_keyframe, edge.to_keyframe);

  // select all keyframes connected to edge.from_keyframe
  std::vector<KeyFrameId> keyframes;
  QueryKeyframeEdges(edge.from_keyframe,
                     [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T&, const Matrix6T&) {
                       if (from_keyframe != edge.from_keyframe && from_keyframe != edge.to_keyframe) {
                         keyframes.push_back(from_keyframe);
                       }
                       if (to_keyframe != edge.from_keyframe && to_keyframe != edge.to_keyframe) {
                         keyframes.push_back(to_keyframe);
                       }
                     });

  // select all keyframes connected to edge.to_keyframe
  // Add to keyframes and required_edges
  std::vector<OptimizeEdgeInfo> required_edges;
  required_edges.reserve(10);
  QueryKeyframeEdges(edge.to_keyframe,
                     [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, [[maybe_unused]] const Isometry3T& from_to,
                         const Matrix6T& from_to_covariance) {
                       if (from_keyframe != edge.to_keyframe && from_keyframe != edge.from_keyframe) {
                         keyframes.push_back(from_keyframe);

                         OptimizeEdgeInfo edge_info;
                         edge_info.from_id = from_keyframe;
                         edge_info.to_id = edge.from_keyframe;  // to_keyframe -> edge.from_keyframe
                         edge_info.from_to_covariance = from_to_covariance + edge.from_to_covariance;
                         required_edges.push_back(edge_info);
                       }
                       if (to_keyframe != edge.to_keyframe && to_keyframe != edge.from_keyframe) {
                         keyframes.push_back(to_keyframe);

                         OptimizeEdgeInfo edge_info;
                         edge_info.from_id = edge.from_keyframe;  // from_keyframe -> edge.from_keyframe
                         edge_info.to_id = to_keyframe;
                         edge_info.from_to_covariance = from_to_covariance + edge.from_to_covariance;
                         required_edges.push_back(edge_info);
                       }
                     });
  // fill from_to with current pose_graph_hypothesis
  for (auto& edge_info : required_edges) {
    auto* from_pose = pose_graph_hypothesis.GetKeyframePose(edge_info.from_id);
    auto* to_pose = pose_graph_hypothesis.GetKeyframePose(edge_info.to_id);
    if (from_pose && to_pose) {
      const Isometry3T m = from_pose->inverse() * (*to_pose);
      edge_info.from_to = m;
    }
  }

  std::sort(keyframes.begin(), keyframes.end());
  const auto it_last_unique_keyframe = std::unique(keyframes.begin(), keyframes.end());
  keyframes.erase(it_last_unique_keyframe, keyframes.end());

  // add keyframe_id
  keyframes.push_back(edge.to_keyframe);
  keyframes.push_back(edge.from_keyframe);

  for (auto& [from_id, to_id, from_to, from_to_covariance] : required_edges) {
    AddEdge(pose_graph_hypothesis, from_id, to_id, from_to, from_to_covariance);
  }

  // Move landmarks from edge.to_keyframe -> edge.from_keyframe
  const auto from_pose = pose_graph_hypothesis.GetKeyframePose(edge.from_keyframe);
  const auto to_pose = pose_graph_hypothesis.GetKeyframePose(edge.to_keyframe);

  if (from_pose && to_pose) {
    const auto it_keyframes = keyframes_.find(edge.from_keyframe);
    if (it_keyframes != keyframes_.end()) {
      PoseGraphKeyFrame& keyframe = it_keyframes->second;
      keyframe.AddKeyframeTransform(*from_pose, *to_pose);
    }
  }

  {
    auto& stable_keyframe = keyframes_[edge.from_keyframe];
    auto& removed_keyframe = keyframes_[edge.to_keyframe];

    if (from_pose && to_pose) {
      const Isometry3T from_to = from_pose->inverse() * (*to_pose);

      // add all landmarks from "to" to "from"
      stable_keyframe.landmarks.insert(stable_keyframe.landmarks.end(), removed_keyframe.landmarks.begin(),
                                       removed_keyframe.landmarks.end());
      // sort and unique
      std::sort(stable_keyframe.landmarks.begin(), stable_keyframe.landmarks.end());
      const auto it_last = std::unique(stable_keyframe.landmarks.begin(), stable_keyframe.landmarks.end());
      stable_keyframe.landmarks.erase(it_last, stable_keyframe.landmarks.end());

      if (func_update_landmark_relation) {
        func_update_landmark_relation(edge.to_keyframe, edge.from_keyframe, removed_keyframe.landmarks, from_to);
      }

      removed_keyframe.landmarks.clear();
    }
  }
  // call RemoveNodeCB
  if (remove_node_cb_) {
    Isometry3T m = Isometry3T::Identity();
    if (from_pose && to_pose) {
      m = from_pose->inverse() * (*to_pose);
    }
    remove_node_cb_(edge.to_keyframe, edge.from_keyframe, m);
  }

  // remove keyframe and edges
  RemoveKeyframe(edge.to_keyframe);

  SlamStdout("|");
  return true;
}

}  // namespace cuvslam::slam
