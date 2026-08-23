
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

#include "sof/sof.h"

#include <atomic>
#include <string>

#include "sof/klt_tracker.h"
#include "sof/lk_tracker.h"
#include "sof/st_tracker.h"

namespace cuvslam::sof {

namespace {
// Global track id counter shared by all trackers in the process. Starts at 1 (0 is reserved).
// Exposed through accessors so Odometry state serialization can checkpoint and restore it:
// after a restore, newly allocated track ids must not collide with ids in the restored state.
std::atomic<TrackId> g_next_raw_track_id{1};
}  // namespace

TrackId AllocateRawTrackId() { return g_next_raw_track_id++; }

TrackId GetNextRawTrackId() { return g_next_raw_track_id.load(); }

void SetNextRawTrackId(TrackId next_id) { g_next_raw_track_id.store(next_id); }

TrackerType ParseTrackerType(const std::string& name) {
  if (name == "lk") return TrackerType::LK;
  if (name == "klt") return TrackerType::KLT;
  if (name == "lk_horizontal") return TrackerType::LKHorizontal;
  if (name == "klt_horizontal") return TrackerType::KLTHorizontal;
  throw std::invalid_argument{"Unknown tracker type: " + name};
}

std::unique_ptr<IFeatureTracker> CreateTracker(TrackerType type) {
  switch (type) {
    case TrackerType::LK:
      return std::make_unique<LKFeatureTracker>();
    case TrackerType::KLT:
      return std::make_unique<KLTTracker>();
    case TrackerType::LKHorizontal:
      return std::make_unique<LKTrackerHorizontal>();
    case TrackerType::KLTHorizontal:
      return std::make_unique<KLTTrackerHorizontal>();
  }
  return nullptr;
}

}  // namespace cuvslam::sof
