
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

#include "slam/map/spatial_index/visualizer.h"

#include "visualizer/visualizer.hpp"

namespace cuvslam::slam {

void LSIGridVisualizer::LogCell(const LSIGrid& lsi, HashCellId hash_cell_id,
                                const PoseGraphHypothesis& pose_graph_hypothesis) {
  const CellId cell_id = lsi.HashCellIdToCellId(hash_cell_id);
  const Vector3T xyz = lsi.CellIdToXYZ(cell_id);
  const std::string name = lsi.CellIdToString(cell_id);
  const auto& rec = visualizer::RerunVisualizer::getInstance().getRecordingStream();
  const rerun::components::Position3D center = {xyz.x(), xyz.y(), xyz.z()};
  const float cell_size = lsi.GetCellSize();
  const std::vector<rerun::datatypes::Vec3D> size = {{cell_size, cell_size, cell_size}};
  const rerun::Rgba32 color(rand() % 256, rand() % 256, rand() % 256);
  rec.log("world/lsi_cells/" + name, rerun::Boxes3D::from_centers_and_sizes(center, size).with_colors({color}));

  const LSIGrid::Cell& cell = lsi.cells_.at(hash_cell_id);
  const std::vector<LandmarkId>& landmark_ids_in_cell = cell.landmarks_in_cell;
  std::vector<rerun::Position3D> positions;
  positions.reserve(landmark_ids_in_cell.size());
  for (const LandmarkId id : landmark_ids_in_cell) {
    const Vector3T p = lsi.GetLandmarkOrStagedCoords(id, pose_graph_hypothesis);
    positions.emplace_back(p.x(), p.y(), p.z());
  }
  rec.log("world/lsi_landmarks/" + name + "/p", rerun::Points3D(positions).with_colors({color}));
}

}  // namespace cuvslam::slam
