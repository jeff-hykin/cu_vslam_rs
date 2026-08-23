
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

#include "slam/map/map.h"
#include "slam/slam/slam.h"
#include "slam/view/view_landmarks.h"
#include "slam/view/view_pose_graph.h"

namespace cuvslam::slam {
// Do it smart - select sparse up to view.landmarks.size()
void PublishAllLandmarksToView(const Map& map, int64_t timestamp_ns, ViewLandmarks& view);
void PublishLandmarksToView(const Map& map, int64_t timestamp_ns,
                            const std::unordered_map<LandmarkId, uint32_t>& landmarks, uint32_t max_landmarks,
                            ViewLandmarks& view);
void PublishPoseGraphToView(const Map& map, int64_t timestamp_ns, ViewPoseGraph& view);
void PublishLoopClosureToView(const Map& map, const std::vector<LandmarkInSolver>& landmarks, ViewLandmarks& view);
}  // namespace cuvslam::slam
