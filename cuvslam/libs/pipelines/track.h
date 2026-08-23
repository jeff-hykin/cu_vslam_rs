
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

#include "common/frame_id.h"
#include "common/unaligned_types.h"
#include "common/vector_3t.h"

namespace cuvslam::pipelines {

struct Landmark {
  TrackId id;
  Vector3T point_w;
};

enum class TrackState {
  kNone = 0,
  kTriangulated,
  kOptimized,  // was in SBA
  kWinzorised,
};

class Track {
  FrameId last_frame_of_visibility = 0;  // for mono purging vanishing, time wise, tracks
  TrackState state_ = TrackState::kNone;
  Vector3T loc3d_;  // track 3D location

public:
  Track() = default;

  void setLastFrameOfVisibility(FrameId lvf) { last_frame_of_visibility = lvf; }

  FrameId getLastFrameOfVisibility() const { return last_frame_of_visibility; }

  void setLocation3D(const Vector3T& loc, const TrackState ts) {
    loc3d_ = loc;
    state_ = ts;
  }

  bool hasLocation() const { return state_ == TrackState::kTriangulated || state_ == TrackState::kOptimized; }

  // return track 3D world coordinates
  const Vector3T& getLocation3D() const {
    assert(hasLocation());
    return loc3d_;
  }

  TrackState getState() const { return state_; }

  void disableTriangulation() { state_ = TrackState::kNone; }

  void winsorize() { state_ = TrackState::kWinzorised; }
};

}  // namespace cuvslam::pipelines
