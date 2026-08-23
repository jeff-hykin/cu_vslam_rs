
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

#include <memory>
#include <mutex>
#include <vector>

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

namespace cuvslam::slam {

struct ViewPose {
  Matrix3T r;
  Vector3T t;

  void set(const Isometry3T& pose) {
    r = pose.linear();
    t = pose.translation();
  }

  operator Isometry3T() const {
    Isometry3T pose;
    pose.linear() = r;
    pose.translation() = t;
    return pose;
  }
};
// view for LC landmarks
struct ViewPoseGraphNode {
  uint64_t id;
  ViewPose node_pose;
};
struct ViewPoseGraphEdge {
  uint64_t node_from;  // node id
  uint64_t node_to;    // node id
  ViewPose transform;
  Matrix6T covariance;
};

struct ViewPoseGraph {
  uint64_t timestamp_ns;
  std::vector<ViewPoseGraphNode> nodes;
  std::vector<ViewPoseGraphEdge> edges;

public:
  ViewPoseGraph(uint32_t max_count) : timestamp_ns(0) {
    nodes.reserve(max_count);
    edges.reserve(max_count);
  }
  uint64_t get_timestamp() const { return timestamp_ns; }
};

}  // namespace cuvslam::slam
