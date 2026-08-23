
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

namespace cuvslam::edex {

// keep start and end frames, take care about reverse ordering
class Timeline {
  FrameId firstFrame_;
  FrameId lastFrame_;

public:
  void set(const FrameId& frameStart, const FrameId& frameEnd) {
    firstFrame_ = frameStart;
    lastFrame_ = frameEnd;
  }
  bool reverse() const { return lastFrame_ < firstFrame_; }
  FrameId firstFrame() const { return firstFrame_; }
  FrameId lastFrame() const { return lastFrame_; }
  bool checkFrameId(const FrameId& frameId) const {
    if (!reverse()) {
      return firstFrame_ <= frameId && frameId <= lastFrame_;
    } else {
      return lastFrame_ <= frameId && frameId <= firstFrame_;
    }
  }
  size_t nFrames() const {
    if (!reverse()) {
      return lastFrame_ - firstFrame_ + 1;
    } else {
      return firstFrame_ - lastFrame_ + 1;
    }
  }
};

// take care about current frameId
class TimeControl {
  Timeline timeline_;
  FrameId currentFrame_;

public:
  TimeControl(const Timeline& timeline) : timeline_(timeline) { rewindToStart(); }
  void rewindToStart() { currentFrame_ = timeline_.firstFrame(); }
  // return false if no more frames
  bool nextFrame() {
    if (currentFrame_ == timeline_.lastFrame()) {
      return false;
    }

    if (!timeline_.reverse()) {
      assert(currentFrame_ < timeline_.lastFrame());
      ++currentFrame_;
    } else {
      assert(currentFrame_ > timeline_.lastFrame());
      --currentFrame_;
    }

    return true;
  }
  FrameId currentFrame() const { return currentFrame_; }
  void set(const Timeline& timeline) {
    timeline_ = timeline;
    rewindToStart();
  }
};

}  // namespace cuvslam::edex
