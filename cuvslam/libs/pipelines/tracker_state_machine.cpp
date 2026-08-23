
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

#include "pipelines/tracker_state_machine.h"

#include "common/log.h"

namespace cuvslam::pipelines {

void StateMachine::set_gravity_init_fn(GravityInitFn fn) { gravity_init_fn_ = std::move(fn); }

void StateMachine::reset() {
  keyframe_timeline_.clear();
  last_successful_frame_time_ns_ = -1;
  last_gravity_estimation_time_ns_ = -1;
  state_ = State::Uninitialized;
}

bool StateMachine::ready_to_init(int64_t ts_ns, const StateMachineSettings& s) const {
  if (keyframe_timeline_.size() < s.min_num_kf_for_gravity) {
    return false;
  }
  if (ts_ns - keyframe_timeline_.front() < s.min_time_period_ns) {
    return false;
  }
  return true;
}

StateMachine::State StateMachine::on_event(Event e, int64_t ts_ns, bool is_keyframe, const StateMachineSettings& s) {
  switch (e) {
    case Event::Reset:
      reset();
      return state_;

    case Event::FrameDropped:
    case Event::FrameFailed: {
      // Either visual tracking failed (FrameFailed) or IMU coverage was incomplete (FrameDropped).
      // Behaviour depends on whether init has completed:
      //   * state_ == Ok : keep the keyframe timeline; transient hiccups should not invalidate
      //                    the gravity / bias estimate. Subsequent IMU-only integration still
      //                    works and gravity stays valid.
      //   * state_ != Ok : clear the keyframe timeline so the gravity callback fires only on a
      //                    contiguous stretch of healthy visual frames. KFs straddling a tracking
      //                    or IMU failure bias the gravity solver toward a bad basin during init.
      if (state_ != State::Ok) {
        keyframe_timeline_.clear();
      }
      // Integration timeout: if too much time has elapsed since the last successful frame, fully
      // reset. This matches the legacy SM, which only triggered the timeout on consecutive failed
      // frames (never on a successful frame, where last_successful is updated to ts_ns).
      if (ts_ns - last_successful_frame_time_ns_ > s.max_integration_time_ns) {
        reset();
      }
      return state_;
    }

    case Event::FrameOk:
      break;
  }

  // FrameOk: append (if keyframe) and slide window.
  if (is_keyframe) {
    keyframe_timeline_.push_back(ts_ns);
    while (!keyframe_timeline_.empty() &&
           keyframe_timeline_.back() - keyframe_timeline_.front() > s.max_time_period_ns) {
      keyframe_timeline_.pop_front();
    }
  }
  last_successful_frame_time_ns_ = ts_ns;

  // Already initialized — only re-run gravity estimation if the refresh period has elapsed.
  if (state_ == State::Ok) {
    if (gravity_init_fn_ && last_gravity_estimation_time_ns_ >= 0 &&
        ts_ns - last_gravity_estimation_time_ns_ > s.gravity_update_period_ns && ready_to_init(ts_ns, s)) {
      if (gravity_init_fn_(s.min_num_kf_for_gravity)) {
        last_gravity_estimation_time_ns_ = ts_ns;
      }
      // On failure we keep state_ == Ok and preserve the timeline; the next keyframe simply
      // retries. Periodic refinement should never demote a healthy SM back to Uninitialized.
    }
    return state_;
  }

  // Not yet enough keyframes / time span — stay in current state.
  if (!ready_to_init(ts_ns, s)) {
    return state_;
  }

  // We can attempt gravity init. Promote to Initializing the first time we cross the threshold.
  if (state_ == State::Uninitialized) {
    state_ = State::Initializing;
  }

  if (gravity_init_fn_ && gravity_init_fn_(s.min_num_kf_for_gravity)) {
    state_ = State::Ok;
    last_gravity_estimation_time_ns_ = ts_ns;
  }
  // On failure we stay in Initializing — the keyframe timeline is preserved so the next keyframe
  // simply retries the callback. This is the key behavioural change vs. the old SM that did a full
  // reset() on every callback failure.
  return state_;
}

}  // namespace cuvslam::pipelines
