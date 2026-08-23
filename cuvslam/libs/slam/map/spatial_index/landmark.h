
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

#include <algorithm>

#include "common/unaligned_types.h"
#include "common/vector_3t.h"

#include "slam/common/slam_common.h"
#include "slam/map/descriptor/feature_descriptor.h"

namespace cuvslam::slam {

struct LandmarkInKeyframe {
  KeyFrameId keyframe_id = 0;                                               //
  storage::Vec3<float> xyz_in_keyframe = storage::Vec3<float>::Zero();      // landmark's point in keyframe space
  storage::Vec2<float> uv_norm_in_keyframe = storage::Vec2<float>::Zero();  // uv of landmark in this frame
  storage::Vec3<float> eye_in_keyframe = storage::Vec3<float>::Zero();      // eye point in keyframe space
  bool has_link = false;                                                    // is landmark binded to this keyframe
  bool has_xyz = false;                                                     // is xyz_in_keyframe valid
};

struct Landmark {
  std::vector<LandmarkInKeyframe> keyframes;
  FeatureDescriptor fd;

public:
  KeyFrameId GetKeyFrame() const {
    for (auto& kfp : keyframes) {
      if (kfp.has_link) {
        return kfp.keyframe_id;
      }
    }
    return InvalidKeyFrameId;
  }

  void AddLandmarkInKeyframe(const LandmarkInKeyframe& data) { this->keyframes.push_back(data); }

  const LandmarkInKeyframe* GetLandmarkInKeyframe(KeyFrameId keyframe_id) const {
    for (const auto& data : this->keyframes) {
      if (data.keyframe_id == keyframe_id) {
        return &data;
      }
    }
    return nullptr;
  }

  // const Isometry3T* FUNC(keyframe_id)
  template <class FUNC>
  bool IsInsideObserverSphere(const Vector3T& pt, float radius_scale /*=1.25*/, float radius_bias /*=0*/,
                              FUNC&& func) const {
    Eigen::AlignedBox3f observer_box;
    for (auto& kfp : keyframes) {
      const Isometry3T* kf_pose = func(kfp.keyframe_id);
      if (!kf_pose) {
        continue;
      }
      Vector3T t = kf_pose->translation();
      observer_box.extend(t);
    }
    Vector3T center = observer_box.center();
    double radius2 = observer_box.diagonal().squaredNorm() * (radius_scale * radius_scale);
    if ((pt - center).squaredNorm() >= radius2 + (radius_bias * radius_bias)) {
      return false;
    }
    return true;
  }

  // const Isometry3T* FUNC(keyframe_id)
  template <class FUNC>
  Vector3T xyz_world(FUNC&& func) const {
    Vector3T xyz(0, 0, 0);
    int count = 0;
    for (auto& kfp : keyframes) {
      if (!kfp.has_link) {
        continue;
      }
      const Isometry3T* kf_pose = func(kfp.keyframe_id);
      if (!kf_pose) {
        continue;
      }
      xyz += *kf_pose * kfp.xyz_in_keyframe;
      count++;
    }
    if (!count) {
      return Vector3T(0, 0, 0);
    }
    xyz /= (float)count;
    return xyz;
  }
};

}  // namespace cuvslam::slam
