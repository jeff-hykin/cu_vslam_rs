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
#include "common/include_gtest.h"

namespace cuvslam::pipelines {
namespace {

constexpr int64_t kFrameDt = static_cast<int64_t>(150e6);  // 150 ms KF spacing

// Test settings sized for short, deterministic tests: 5 KFs minimum, 500 ms minimum span.
// 5 KFs at 150 ms spacing -> 600 ms span (exceeds the 500 ms min_time_period_ns).
StateMachineSettings small_settings() {
  StateMachineSettings s;
  s.min_num_kf_for_gravity = 5;
  s.min_time_period_ns = static_cast<int64_t>(500e6);
  s.max_time_period_ns = static_cast<int64_t>(5e9);
  s.max_integration_time_ns = static_cast<int64_t>(1e9);
  // Effectively disable periodic re-estimation by default; individual tests override as needed.
  s.gravity_update_period_ns = static_cast<int64_t>(5e12);
  return s;
}

class CallbackCounter {
public:
  CallbackCounter(bool result) : result_(result) {}
  bool operator()(size_t /*num_kfs*/) {
    ++calls_;
    return result_;
  }
  int calls() const { return calls_; }
  void set_result(bool r) { result_ = r; }

private:
  bool result_;
  int calls_ = 0;
};

TEST(StateMachineTest, UninitializedToOkOnSuccessfulInit) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
  int64_t ts = 0;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s),
              StateMachine::State::Uninitialized);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 0);
  EXPECT_EQ(sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 1);
}

TEST(StateMachineTest, FailedCallbackKeepsTimelineAndStaysInitializing) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/false);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Initializing);
  EXPECT_EQ(cb.calls(), 1);

  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  ts += kFrameDt;
  EXPECT_EQ(sm.state(), StateMachine::State::Initializing);
  EXPECT_EQ(cb.calls(), 2);

  cb.set_result(true);
  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 3);
}

TEST(StateMachineTest, FrameFailedDuringInitClearsTimeline) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 4; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);

  sm.on_event(StateMachine::Event::FrameFailed, ts, /*is_keyframe=*/false, s);
  ts += kFrameDt;

  for (int i = 0; i < 4; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 0);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);

  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  EXPECT_EQ(cb.calls(), 1);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
}

TEST(StateMachineTest, FrameFailedAfterOkKeepsTimelineAndState) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  sm.on_event(StateMachine::Event::FrameFailed, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 1);
}

TEST(StateMachineTest, FrameDroppedTimeoutResetsToUninitialized) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  ts += static_cast<int64_t>(2e9);  // 2s gap > max_integration_time_ns
  sm.on_event(StateMachine::Event::FrameDropped, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
}

TEST(StateMachineTest, ResetEventDropsToUninitialized) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  sm.on_event(StateMachine::Event::Reset, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
}

TEST(StateMachineTest, PeriodicGravityRefinementWhenOk) {
  StateMachine sm;
  StateMachineSettings s = small_settings();
  s.gravity_update_period_ns = static_cast<int64_t>(1e9);  // refine every 1s while Ok
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  // Drive into Ok with the initial 5 KFs (600 ms elapsed).
  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  ASSERT_EQ(sm.state(), StateMachine::State::Ok);
  ASSERT_EQ(cb.calls(), 1);

  // Within the refresh period: callback must NOT fire again.
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 1);

  // Advance > refresh period from the last successful estimation -> callback fires once.
  ts += static_cast<int64_t>(2e9);
  // Use a small frame gap for max_integration_time_ns to allow a 2s jump only via a Reset would
  // normally trigger; bump the SM forward via a successful keyframe at the new timestamp. We need
  // max_integration_time_ns large enough to not reset on this artificial jump.
  StateMachineSettings s2 = s;
  s2.max_integration_time_ns = static_cast<int64_t>(5e9);
  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s2);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 2);
}

}  // namespace
}  // namespace cuvslam::pipelines
