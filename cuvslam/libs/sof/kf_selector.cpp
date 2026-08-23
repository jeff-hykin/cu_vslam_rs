
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

#include "sof/kf_selector.h"

#include "common/log_types.h"

namespace cuvslam::sof {

#ifndef NDEBUG
bool IsSorted(const TracksVector& v) {
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i - 1].id() >= v[i].id()) {
      return false;
    }
  }

  return true;
}
#endif

// p_sum_motion - average motion in pixels
static uint32_t CalcNSurvivors(const TracksVector& frame1, const TracksVector& frame2) {
  assert(IsSorted(frame1));
  assert(IsSorted(frame2));
  const auto size1 = frame1.size();
  const auto size2 = frame2.size();

  uint32_t i1 = 0;
  uint32_t i2 = 0;
  uint32_t n_survivors = 0;  // number of alive tracks between frames

  while (i1 < size1) {
    const Track& t1 = frame1[i1];
    ++i1;

    if (t1.dead()) {
      continue;
    }

    while (i2 < size2) {
      const Track& t2 = frame2[i2];

      if (!t2.dead() && t1.id() == t2.id()) {
        ++n_survivors;
      }

      if ((i2 + 1 < size2) && (frame2[i2 + 1].id() <= t1.id())) {
        ++i2;
      } else {
        break;
      }
    }
  }

  return n_survivors;
}

// return percent (0.f-100.f) of survivor tracks
static float CalcSurvivorTracksPercentage(const TracksVector& frame1, uint32_t n_survivors) {
  const size_t n_frame1_tracks = frame1.get_num_alive();

  assert(n_survivors <= n_frame1_tracks);

  if (n_frame1_tracks == 0) {
    return 0;
  }

  const float percentage = 100.f * n_survivors / n_frame1_tracks;
  assert(0.f <= percentage && percentage <= 100.f);
  return percentage;
}

KFSelector::KFSelector([[maybe_unused]] const odom::KeyFrameSettings& kf_settings) {}

bool KFSelector::select(const TracksVector& cur_frame_tracks, const int64_t current_timestamp_ns,
                        const TracksVector& last_kf_tracks, const int64_t last_kf_timestamp,
                        const odom::KeyFrameSettings& kf_settings) {
  TRACE_EVENT ev = profiler_domain_.trace_event("KFSelector::select()", profiler_color_);
  assert(IsSorted(cur_frame_tracks));

  // kf_override_frame_selection is applied here for multicamera, where the keyframe decision is a
  // single global one. The mono path applies the same override in MonoSOFBase::SelectKeyframe
  // (guarded by is_mono_mode); the two are mutually exclusive — multicamera constructs
  // MonoSOFFrameSettings with is_mono_mode=false so the per-camera MonoSOF never double-applies it.
  if (const std::optional<bool> override_frame_selection = kf_settings.override_frame_selection) {
    if (*override_frame_selection) {
      first_kf_selected_ = true;
    }
    return *override_frame_selection;
  }

  if (!first_kf_selected_) {
    first_kf_selected_ = true;  // first frame is always keyframe
    return true;                // all tracks are dead, need new keyframe ASAP
  }

  const uint32_t n_survivors_from_last = CalcNSurvivors(last_kf_tracks, cur_frame_tracks);
  if (n_survivors_from_last == 0) {
    return true;  // all tracks are dead, need new keyframe ASAP
  }

  assert(CalcNSurvivors(last_kf_tracks, cur_frame_tracks) == n_survivors_from_last);

  if (CalcSurvivorTracksPercentage(last_kf_tracks, n_survivors_from_last) < kf_settings.survivor_from_last) {
    return true;
  }

  // Even if there are no significant changes in the tracks,
  // it's necessary to force  keyframe selection after a certain time duration
  // to deal with the stationary cases
  if (current_timestamp_ns - last_kf_timestamp > kf_settings.max_timedelta_between_kfs_s * 1e9) {
    return true;
  }

  return false;
}

void KFSelector::reset() { first_kf_selected_ = false; }

bool KFSelector::first_kf_selected() const { return first_kf_selected_; }

}  // namespace cuvslam::sof
