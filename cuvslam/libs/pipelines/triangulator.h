
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

#include <random>
#include <unordered_set>
#include <vector>

#include "camera/frustum_intersection_graph.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/vector_3t.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "pipelines/track.h"

namespace cuvslam::pipelines {

class MulticamTriangulator {
public:
  MulticamTriangulator(const camera::Rig& rig);

  void reset();

  std::vector<Landmark> triangulate(const Isometry3T& world_from_rig,
                                    const std::vector<camera::Observation>& observations);

private:
  camera::Rig rig_;
  camera::FrustumIntersectionGraph fig_;

  std::unordered_set<TrackId> visible_tracks_;
  std::unordered_set<TrackId> visible_traingulated_tracks_;
  std::unordered_set<TrackId> triangulated_tracks_;

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("MulticamTriangulator");
};

}  // namespace cuvslam::pipelines
