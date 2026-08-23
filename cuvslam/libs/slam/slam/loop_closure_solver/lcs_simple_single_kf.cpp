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

#include "slam/slam/loop_closure_solver/lcs_simple_single_kf.h"

namespace cuvslam::slam {

LoopClosureSolverSimpleSingleKF::LoopClosureSolverSimpleSingleKF(const camera::Rig& rig, RansacType ransac_type,
                                                                 bool randomized)
    : LoopClosureSolverSimple(rig, ransac_type, randomized), rig_(rig) {}

LoopClosureSolverSimpleSingleKF::~LoopClosureSolverSimpleSingleKF() {}

void LoopClosureSolverSimpleSingleKF::SelectLandmarksCandidates(const LoopClosureTask& task,
                                                                const LSIGrid& landmarks_spatial_index,
                                                                const IFeatureDescriptorOps* feature_descriptor_ops,
                                                                std::vector<LandmarkInfo>& landmark_candidates,
                                                                DiscardLandmarkCB* discard_landmark_cb,
                                                                KeyframeInSightCB* keyframe_in_sight_cb) const {
  LoopClosureSolverSimple::SelectLandmarksCandidates(task, landmarks_spatial_index, feature_descriptor_ops,
                                                     landmark_candidates, discard_landmark_cb, keyframe_in_sight_cb);

  // filter landmark_candidates
  struct Sum {
    int value = 0;
  };
  std::map<KeyFrameId, Sum> keyframes;

  // fill keyframes
  for (auto& landmark_info : landmark_candidates) {
    landmarks_spatial_index.QueryLandmarkRelations(landmark_info.id, [&](KeyFrameId keyframe_id, const Vector2T&) {
      keyframes[keyframe_id].value++;
      return true;
    });
  }

  // select keyframe with max landmarks
  KeyFrameId selected_keyframe_id = InvalidKeyFrameId;
  int selected_landmarks_count = 0;
  for (auto& it_keyframe : keyframes) {
    if (it_keyframe.second.value > selected_landmarks_count) {
      selected_landmarks_count = it_keyframe.second.value;
      selected_keyframe_id = it_keyframe.first;
    }
  }
  if (selected_keyframe_id == InvalidKeyFrameId) {
    return;
  }

  // remove others landmarks
  auto pred = [&](const LandmarkInfo& landmark_info) {
    bool mine = false;
    landmarks_spatial_index.QueryLandmarkRelations(landmark_info.id, [&](KeyFrameId keyframe_id, const Vector2T&) {
      mine |= (selected_keyframe_id == keyframe_id);
      return true;
    });
    return !mine;
  };
  auto remove_it = std::remove_if(landmark_candidates.begin(), landmark_candidates.end(), pred);
  landmark_candidates.erase(remove_it, landmark_candidates.end());
}

}  // namespace cuvslam::slam
