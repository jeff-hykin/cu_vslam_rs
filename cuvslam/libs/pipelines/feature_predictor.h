
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

#include "camera/rig.h"
#include "map/map.h"
#include "sof/feature_prediction_interface.h"

namespace cuvslam::pipelines {

class FeaturePredictor : public sof::IFeaturePredictor {
public:
  FeaturePredictor(const map::UnifiedMap& map_, const camera::Rig& rig_);

  void predictObservations(const Isometry3T& world_from_rig, int cameraIndex, const std::vector<TrackId>& ids,
                           sof::Prediction& uvs) const final;

private:
  const map::UnifiedMap& map_;
  camera::Rig rig_;
  mutable map::TrackIdMap map_landmarks_;

  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("FeaturePredictor");
};

}  // namespace cuvslam::pipelines
