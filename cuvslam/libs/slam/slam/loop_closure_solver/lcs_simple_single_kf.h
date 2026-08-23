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
#include "slam/slam/loop_closure_solver/lcs_simple.h"

namespace cuvslam::slam {

class LoopClosureSolverSimpleSingleKF : public LoopClosureSolverSimple {
public:
  LoopClosureSolverSimpleSingleKF(const camera::Rig& rig, RansacType ransac_type, bool randomized = true);
  ~LoopClosureSolverSimpleSingleKF() override;

  virtual void SelectLandmarksCandidates(const LoopClosureTask& task, const LSIGrid& landmarks_spatial_index,
                                         const IFeatureDescriptorOps* feature_descriptor_ops,
                                         std::vector<LandmarkInfo>& landmark_candidates,
                                         DiscardLandmarkCB* discard_landmark_cb,
                                         KeyframeInSightCB* keyframe_in_sight_cb) const override;

private:
  camera::Rig rig_;
};

}  // namespace cuvslam::slam
