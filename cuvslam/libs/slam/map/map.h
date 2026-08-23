
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

#include "camera/rig.h"
#include "slam/map/database/slam_database.h"
#include "slam/map/pose_graph/pose_graph.h"
#include "slam/map/spatial_index/lsi_grid.h"

namespace cuvslam::slam {

enum class FeatureDescriptorType { kNone, kShiTomasi2, kShiTomasi6 };

struct Map {
  Map(const camera::Rig& rig, FeatureDescriptorType descriptor_type, bool use_gpu);

  std::shared_ptr<ISlamDatabase> database_;
  PoseGraph pose_graph_;
  PoseGraphHypothesis pose_graph_hypothesis_;
  PoseGraphHypothesis pose_graph_hypothesis_for_swap_;
  std::shared_ptr<LSIGrid> landmarks_spatial_index_;
  std::shared_ptr<IFeatureDescriptorOps> feature_descriptor_ops_;  // operations over descriptors: Match/Serialize

  bool AttachDatabase(std::shared_ptr<ISlamDatabase> database, bool load_data);
  void DetachDatabase(bool copy_all_from_db);

  const PoseGraph& GetPoseGraph() const;
  const PoseGraphHypothesis& GetPoseGraphHypothesis() const;
  std::shared_ptr<const LSIGrid> GetLandmarksSpatialIndex() const;

  static Matrix6T GetHardEdgeDefaultCovariance();

  void logPoseGraph() const;

  std::pair<KeyFrameId, Isometry3T> GetRootKeyframe() const;
};

}  // namespace cuvslam::slam
