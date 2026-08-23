
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
#include <optional>
#include <vector>

#include "common/isometry.h"
#include "common/vector_2t.h"

namespace cuvslam::sof {

using Prediction = std::vector<std::optional<Vector2T>>;  // list of predicted uvs

// This class is responsible for generating predictions for 2D tracks
// based on track ids.
// It relies on predicted pose and existing positions of 3D points.
class IFeaturePredictor {
public:
  // Caller is responsible for allocating the memory (resizing uvs vector),
  virtual void predictObservations(const Isometry3T& world_from_rig, int cameraIndex, const std::vector<TrackId>& ids,
                                   Prediction& uvs) const = 0;

  virtual ~IFeaturePredictor() = default;
};

using FeaturePredictorPtr = std::shared_ptr<IFeaturePredictor>;

}  // namespace cuvslam::sof
