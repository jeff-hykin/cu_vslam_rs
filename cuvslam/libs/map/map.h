
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

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "jg/dense_hash_map.hpp"

#include "camera/observation.h"
#include "common/fixed_array.h"
#include "common/isometry.h"
#include "common/state_serial.h"
#include "common/track_id.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "imu/imu_preintegration.h"
#include "pipelines/track.h"
#include "profiler/profiler_enable.h"

#include "map/keyframe.h"

namespace cuvslam::map {

using namespace cuvslam::sba_imu;
using PreintegPtr = std::shared_ptr<IMUPreintegration>;
using KeyframePtr = std::shared_ptr<KeyFrame>;
using LandmarkPtr = std::shared_ptr<Landmark>;

template <class U, class V>
using Map = std::unordered_map<U, V>;

template <class U, class V>
using FastMap = jg::dense_hash_map<U, V>;

template <class U>
using Set = std::unordered_set<U>;
using ObservPtr = std::shared_ptr<camera::Observation>;

struct TrackIdHash {
  std::size_t operator()(const TrackId& track_id) const noexcept { return track_id; }
};

using TrackIdMap = jg::dense_hash_map<TrackId, Vector3T, TrackIdHash>;

class UnifiedMap {
public:
  UnifiedMap(size_t capacity);
  void add_keyframe(int64_t time_ns, const State& state,
                    const IMUPreintegration& preint,  // for first fk simply pass empty prein-n. The map wont store it.
                    const std::vector<camera::Observation>& observations,
                    const std::vector<pipelines::Landmark>& triangulated_tracks);

  // for pnp
  Map<TrackId, Vector3T> get_recent_landmarks() const;
  void get_recent_landmarks(Map<TrackId, Vector3T>& out) const;
  TrackIdMap get_recent_landmarks(CameraId cam_id) const;
  void get_recent_landmarks(CameraId cam_id, TrackIdMap& out) const;
  struct KeyframeWithPreint {
    KeyframeWithPreint(const KeyframeWithPreint& other) = default;
    KeyframeWithPreint(KeyframeWithPreint&& other) = default;
    KeyframeWithPreint& operator=(KeyframeWithPreint&& other) = default;

    KeyframePtr keyframe;

    // contains measurements SINCE this keyframe
    // i.e. last keyframe in map ALWAYS has nullptr as preintegration
    PreintegPtr preintegration = nullptr;
  };

  struct LandmarkAndObserv {
    LandmarkPtr landmark;
    FixedArray<camera::Observation, 3> observations;
  };

  struct SubMap {
    std::vector<KeyframeWithPreint> consecutive_keyframes;
    // pass to SBA not more then 600 landmarks per keyframe
    std::vector<FixedArray<LandmarkAndObserv, 600>> landmark_and_obs;
  };

  // needed for sba;
  /*
  filter_landmarks enables filtering out landmarks visible only from a single keyframe.
  Such landmarks should not affect the SBA, but, neveretheless it affects Fusion SBA.
  So this filtering is only enabled for regular SBA.
  */
  SubMap get_recent_submap(size_t max_last_keyframes, bool filter_landmarks = false) const;

  std::deque<KeyframeWithPreint> get_consecutive_keyframes() const;

  void set_gravity(const Vector3T& gravity);
  void reset_gravity();
  std::optional<Vector3T> get_gravity() const;

  size_t size() const;
  size_t num_landmarks() const;
  size_t capacity() const;
  bool empty() const;

  void clear();

  // Checkpoint support: serialize/restore the full keyframe/landmark graph. Landmark object
  // identity is keyed by track id (one shared Landmark instance per alive track id), keyframe
  // identity by position in the consecutive-keyframes deque.
  void save_state(serial::Writer& w) const;
  void load_state(serial::Reader& r);

private:
  void remove_tail_keyframe_thread_unsafe();

  const size_t capacity_;

  mutable std::mutex gravity_mutex_;
  std::optional<Vector3T> gravity_ = std::nullopt;
  mutable std::mutex map_mutex_;

  // last value should always have preint == nullptr!!!
  std::deque<KeyframeWithPreint> consecutive_keyframes_;

  Map<KeyframePtr, FastMap<TrackId, LandmarkAndObserv>> landmarks_from_keyframe_;
  FastMap<TrackId, Set<KeyframePtr>> keyframes_from_landmark_;

  profiler::MapProfiler::DomainHelper profiler_domain_ = profiler::MapProfiler::DomainHelper("Map");
};

}  // namespace cuvslam::map
