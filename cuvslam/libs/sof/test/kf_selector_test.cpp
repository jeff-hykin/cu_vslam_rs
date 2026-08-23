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

#include "common/include_gtest.h"
#include "odometry/svo_config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace test::sof {
namespace {

using namespace cuvslam;

cuvslam::sof::TracksVector MakeTracks(size_t count) {
  cuvslam::sof::TracksVector tracks;
  std::vector<Vector2T> points;
  points.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    points.emplace_back(static_cast<float>(i), static_cast<float>(i));
  }
  tracks.add(0, points);
  tracks.sort();
  return tracks;
}

cuvslam::odom::KeyFrameSettings ForcedSettings(bool frame_is_keyframe) {
  cuvslam::odom::KeyFrameSettings settings;
  settings.override_frame_selection = frame_is_keyframe;
  return settings;
}

bool HaveNoCommonTrackIds(const cuvslam::sof::TracksVector& lhs, const cuvslam::sof::TracksVector& rhs) {
  auto lhs_it = lhs.cbegin();
  auto rhs_it = rhs.cbegin();
  while (lhs_it != lhs.cend() && rhs_it != rhs.cend()) {
    if (lhs_it->id() == rhs_it->id()) {
      return false;
    }
    if (lhs_it->id() < rhs_it->id()) {
      ++lhs_it;
    } else {
      ++rhs_it;
    }
  }
  return true;
}

}  // namespace

TEST(KFSelector, DefaultAutomaticSelectionStillSelectsFirstFrameOnly) {
  cuvslam::odom::KeyFrameSettings settings;
  cuvslam::sof::KFSelector selector(settings);
  const cuvslam::sof::TracksVector empty_tracks;
  const auto last_kf_tracks = MakeTracks(50);

  EXPECT_TRUE(selector.select(last_kf_tracks, 0, empty_tracks, 0, settings));
  EXPECT_TRUE(selector.first_kf_selected());

  const auto current_tracks = last_kf_tracks;
  EXPECT_FALSE(selector.select(current_tracks, 1, last_kf_tracks, 0, settings));
}

TEST(KFSelector, ForcedKeyframeSkipsAutomaticChecks) {
  cuvslam::odom::KeyFrameSettings settings;
  cuvslam::sof::KFSelector selector(settings);
  const cuvslam::sof::TracksVector empty_tracks;
  const auto last_kf_tracks = MakeTracks(50);

  EXPECT_TRUE(selector.select(last_kf_tracks, 0, empty_tracks, 0, settings));

  const auto current_tracks = last_kf_tracks;
  EXPECT_TRUE(selector.select(current_tracks, 1, last_kf_tracks, 0, ForcedSettings(true)));
  EXPECT_TRUE(selector.first_kf_selected());
}

TEST(KFSelector, ForcedNonKeyframeCanSkipFirstFrameSelection) {
  cuvslam::odom::KeyFrameSettings settings;
  cuvslam::sof::KFSelector selector(settings);
  const cuvslam::sof::TracksVector empty_tracks;
  const auto current_tracks = MakeTracks(50);

  EXPECT_FALSE(selector.select(current_tracks, 0, empty_tracks, 0, ForcedSettings(false)));
  EXPECT_FALSE(selector.first_kf_selected());

  EXPECT_TRUE(selector.select(current_tracks, 1, empty_tracks, 0, settings));
}

TEST(KFSelector, ForcedNonKeyframeSkipsZeroSurvivorAndNextAutomaticFrameUsesPreviousKeyframe) {
  cuvslam::odom::KeyFrameSettings settings;
  cuvslam::sof::KFSelector selector(settings);
  const cuvslam::sof::TracksVector empty_tracks;
  const auto last_kf_tracks = MakeTracks(50);
  // A fresh MakeTracks() call creates a disjoint track-id range, so current_tracks has zero survivors
  // relative to last_kf_tracks when automatic selection resumes below.
  const auto current_tracks = MakeTracks(50);

  EXPECT_TRUE(HaveNoCommonTrackIds(last_kf_tracks, current_tracks));
  EXPECT_TRUE(selector.select(last_kf_tracks, 0, empty_tracks, 0, settings));
  EXPECT_FALSE(selector.select(current_tracks, 1, last_kf_tracks, 0, ForcedSettings(false)));

  EXPECT_TRUE(selector.select(current_tracks, 2, last_kf_tracks, 0, settings));
}

TEST(KFSelector, ForcedNonKeyframeSkipsMaxTimeRule) {
  cuvslam::odom::KeyFrameSettings settings;
  settings.max_timedelta_between_kfs_s = 1;
  cuvslam::sof::KFSelector selector(settings);
  const cuvslam::sof::TracksVector empty_tracks;
  const auto last_kf_tracks = MakeTracks(50);
  const auto current_tracks = last_kf_tracks;

  EXPECT_TRUE(selector.select(last_kf_tracks, 0, empty_tracks, 0, settings));

  constexpr int64_t kLateTimestampNs = 2'000'000'001;
  EXPECT_FALSE(selector.select(current_tracks, kLateTimestampNs, last_kf_tracks, 0, ForcedSettings(false)));
  EXPECT_TRUE(selector.select(current_tracks, kLateTimestampNs, last_kf_tracks, 0, settings));
}

}  // namespace test::sof
