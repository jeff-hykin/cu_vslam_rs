
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

#include "map/map.h"

#include <map>

#include "common/log.h"

namespace cuvslam::map {
using namespace cuvslam::camera;

UnifiedMap::UnifiedMap(size_t capacity) : capacity_(capacity) {
  if (capacity_ == 0) {
    throw std::runtime_error("UnifiedMap capacity cant be zero!");
  }
}

std::deque<UnifiedMap::KeyframeWithPreint> UnifiedMap::get_consecutive_keyframes() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_;
}

void UnifiedMap::add_keyframe(int64_t time_ns, const State& state, const IMUPreintegration& preint,
                              const std::vector<camera::Observation>& observations,
                              const std::vector<pipelines::Landmark>& triangulated_tracks) {
  TRACE_EVENT ev = profiler_domain_.trace_event("add_keyframe");
  // create new keyframe
  auto new_kf = std::make_shared<KeyFrame>(state, time_ns);

  FastMap<TrackId, LandmarkAndObserv> new_landmarks;

  std::lock_guard<std::mutex> lock(map_mutex_);

  for (const auto& [track_id, point] : triangulated_tracks) {
    auto it = keyframes_from_landmark_.find(track_id);
    if (it != keyframes_from_landmark_.end()) {
      const auto& kf_set = it->second;
      auto& visible_landmarks = landmarks_from_keyframe_[*kf_set.begin()];
      auto& [landmark, _] = visible_landmarks[track_id];
      if (!landmark->get_pose()) {
        landmark->set_pose(point);
      }
    } else {
      new_landmarks[track_id].landmark = std::make_shared<Landmark>(track_id, point);
    }
  }

  FastMap<TrackId, LandmarkAndObserv> old_landmarks;
  for (const auto& obs : observations) {
    auto it = keyframes_from_landmark_.find(obs.id);
    if (it == keyframes_from_landmark_.end()) {
      // obs of new landmark
      auto& l_o = new_landmarks[obs.id];
      l_o.observations.try_push_back(obs);

      if (!l_o.landmark) {
        l_o.landmark = std::make_shared<Landmark>(obs.id);
      }

    } else {
      // obs of old landmark
      const auto& kf_set = it->second;
      auto& visible_landmarks = landmarks_from_keyframe_[*kf_set.begin()];
      auto& [landmark, _] = visible_landmarks[obs.id];

      auto& old_lm = old_landmarks[obs.id];
      old_lm.landmark = landmark;
      old_lm.observations.try_push_back(obs);
    }
  }

  auto& landmarks = landmarks_from_keyframe_[new_kf];
  landmarks = std::move(new_landmarks);
  std::move(old_landmarks.begin(), old_landmarks.end(), std::inserter(landmarks, landmarks.begin()));

  for (auto& [track_id, l_o] : landmarks) {
    keyframes_from_landmark_[track_id].insert(new_kf);
  }

  if (!consecutive_keyframes_.empty()) {
    auto& [kf, p] = consecutive_keyframes_.back();
    p = std::make_shared<IMUPreintegration>(preint);
  }

  consecutive_keyframes_.push_back({new_kf});

  while (consecutive_keyframes_.size() > capacity_) {
    remove_tail_keyframe_thread_unsafe();
  }
}

Map<TrackId, Vector3T> UnifiedMap::get_recent_landmarks() const {
  Map<TrackId, Vector3T> out;
  get_recent_landmarks(out);
  return out;
}

void UnifiedMap::get_recent_landmarks(Map<TrackId, Vector3T>& out) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_landmarks");
  std::lock_guard<std::mutex> lock(map_mutex_);
  out.clear();
  if (consecutive_keyframes_.empty()) {
    return;
  }

  const auto& recent_landmarks = landmarks_from_keyframe_.at(consecutive_keyframes_.back().keyframe);
  out.reserve(recent_landmarks.size());
  for (const auto& [track_id, lm_with_obs] : recent_landmarks) {
    const std::optional<Vector3T>& point_3d = lm_with_obs.landmark->get_pose();
    if (point_3d) {
      out.insert({track_id, *point_3d});
    }
  }
}

TrackIdMap UnifiedMap::get_recent_landmarks(CameraId cam_id) const {
  TrackIdMap out;
  get_recent_landmarks(cam_id, out);
  return out;
}

void UnifiedMap::get_recent_landmarks(CameraId cam_id, TrackIdMap& out) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_landmarks(cam_id)");
  std::lock_guard<std::mutex> lock(map_mutex_);
  out.clear();
  if (consecutive_keyframes_.empty()) {
    return;
  }

  const auto& recent_landmarks = landmarks_from_keyframe_.at(consecutive_keyframes_.back().keyframe);
  out.reserve(recent_landmarks.size());
  for (const auto& [track_id, lm_with_obs] : recent_landmarks) {
    const std::optional<Vector3T>& point_3d = lm_with_obs.landmark->get_pose();
    if (point_3d) {
      for (const auto& obs : lm_with_obs.observations) {
        if (obs.cam_id == cam_id) {
          out.insert({track_id, *point_3d});
          break;
        }
      }
    }
  }
}

void UnifiedMap::remove_tail_keyframe_thread_unsafe() {
  // thread unsafe means that mutex is locked in the outer scope

  KeyframePtr kf = consecutive_keyframes_.front().keyframe;

  for (const auto& [track_id, landmark] : landmarks_from_keyframe_.at(kf)) {
    auto& kf_set = keyframes_from_landmark_[track_id];
    kf_set.erase(kf);
    if (kf_set.empty()) {
      keyframes_from_landmark_.erase(track_id);
    }
  }
  assert(!landmarks_from_keyframe_.empty());
  landmarks_from_keyframe_.erase(kf);
  consecutive_keyframes_.pop_front();
}

UnifiedMap::SubMap UnifiedMap::get_recent_submap(size_t max_last_keyframes, bool filter_landmarks) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_submap");
  SubMap sub_map;
  sub_map.consecutive_keyframes.reserve(max_last_keyframes);

  std::lock_guard<std::mutex> lock(map_mutex_);
  {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("add keyframes");
    // add recent keyframes
    auto it = consecutive_keyframes_.rbegin();
    for (size_t i = 0; i < max_last_keyframes; i++) {
      if (it == consecutive_keyframes_.rend()) {
        break;
      }
      sub_map.consecutive_keyframes.push_back(*it);
      it++;
    }
    std::reverse(sub_map.consecutive_keyframes.begin(), sub_map.consecutive_keyframes.end());
  }

  {
    sub_map.landmark_and_obs.resize(sub_map.consecutive_keyframes.size());
    TRACE_EVENT ev1 = profiler_domain_.trace_event("add landmarks");

    for (size_t i = 0; i < sub_map.consecutive_keyframes.size(); i++) {
      const auto& kf = sub_map.consecutive_keyframes[i].keyframe;
      auto& lms = sub_map.landmark_and_obs[i];
      for (const auto& [track_id, x] : landmarks_from_keyframe_.at(kf)) {
        if (lms.full()) {
          break;
        }
        if (!filter_landmarks) {
          lms.try_push_back(x);
          continue;
        }

        if (x.landmark->get_pose() && keyframes_from_landmark_.at(track_id).size() > 1) {
          lms.try_push_back(x);
        }
      }
    }
  }
  return sub_map;
}

size_t UnifiedMap::size() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_.size();
}

size_t UnifiedMap::num_landmarks() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return keyframes_from_landmark_.size();
}

bool UnifiedMap::empty() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_.empty();
}

void UnifiedMap::set_gravity(const Vector3T& gravity) {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  gravity_ = gravity;
}

void UnifiedMap::reset_gravity() {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  gravity_ = std::nullopt;
}

std::optional<Vector3T> UnifiedMap::get_gravity() const {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  return gravity_;
}

size_t UnifiedMap::capacity() const { return capacity_; }

void UnifiedMap::clear() {
  std::scoped_lock lock(map_mutex_, gravity_mutex_);

  gravity_ = std::nullopt;
  landmarks_from_keyframe_.clear();
  consecutive_keyframes_.clear();
  keyframes_from_landmark_.clear();
}

void UnifiedMap::save_state(serial::Writer& w) const {
  std::scoped_lock lock(map_mutex_, gravity_mutex_);
  w.write_tag(0x554D4150);  // "UMAP"

  w.write_bool(gravity_.has_value());
  if (gravity_.has_value()) {
    w.write_eigen(*gravity_);
  }

  // Keyframes in deque order; the deque index becomes the keyframe identity in the stream.
  w.write_size(consecutive_keyframes_.size());
  Map<KeyframePtr, size_t> kf_index;
  kf_index.reserve(consecutive_keyframes_.size());
  for (const KeyframeWithPreint& kf_p : consecutive_keyframes_) {
    kf_index.emplace(kf_p.keyframe, kf_index.size());
    w.write_pod<int64_t>(kf_p.keyframe->time_ns());
    const State state = kf_p.keyframe->get_state();
    w.write_isometry(state.rig_from_w);
    w.write_eigen(state.velocity);
    w.write_eigen(state.acc_bias);
    w.write_eigen(state.gyro_bias);
    w.write_bool(kf_p.preintegration != nullptr);
    if (kf_p.preintegration != nullptr) {
      kf_p.preintegration->save_state(w);
    }
  }

  // Landmark table keyed (and ordered) by track id — one shared Landmark instance per track id.
  std::map<TrackId, LandmarkPtr> landmark_table;
  for (const auto& [kf, landmarks] : landmarks_from_keyframe_) {
    for (const auto& [track_id, lm_obs] : landmarks) {
      landmark_table.emplace(track_id, lm_obs.landmark);
    }
  }
  w.write_size(landmark_table.size());
  for (const auto& [track_id, landmark] : landmark_table) {
    w.write_pod(track_id);
    const std::optional<Vector3T> pose = landmark->get_pose();
    w.write_bool(pose.has_value());
    if (pose.has_value()) {
      w.write_eigen(*pose);
    }
  }

  // Per-keyframe landmark entries, keyframes in deque order, entries sorted by track id.
  w.write_size(landmarks_from_keyframe_.size());
  for (const KeyframeWithPreint& kf_p : consecutive_keyframes_) {
    const auto it = landmarks_from_keyframe_.find(kf_p.keyframe);
    if (it == landmarks_from_keyframe_.end()) {
      continue;
    }
    w.write_size(kf_index.at(kf_p.keyframe));
    // Entries are written in the FastMap's own iteration order and re-inserted in that order on
    // load: open-addressing probe chains (and therefore iteration order, which fixes the SBA
    // floating-point summation order) depend on insertion order, not just content.
    w.write_size(it->second.size());
    for (const auto& [track_id, lm_obs] : it->second) {
      w.write_pod(track_id);
      w.write_size(lm_obs.observations.size());
      for (const camera::Observation& obs : lm_obs.observations) {
        w.write_pod(obs.cam_id);
        w.write_pod(obs.id);
        w.write_eigen(obs.xy);
        w.write_eigen(obs.xy_info);
      }
    }
  }
}

void UnifiedMap::load_state(serial::Reader& r) {
  std::scoped_lock lock(map_mutex_, gravity_mutex_);
  r.expect_tag(0x554D4150, "UnifiedMap");

  gravity_ = std::nullopt;
  if (r.read_bool()) {
    Vector3T g;
    r.read_eigen(g);
    gravity_ = g;
  }

  consecutive_keyframes_.clear();
  landmarks_from_keyframe_.clear();
  keyframes_from_landmark_.clear();

  const size_t num_keyframes = r.read_size();
  for (size_t i = 0; i < num_keyframes; ++i) {
    const int64_t time_ns = r.read_pod<int64_t>();
    State state;
    r.read_isometry(state.rig_from_w);
    r.read_eigen(state.velocity);
    r.read_eigen(state.acc_bias);
    r.read_eigen(state.gyro_bias);
    consecutive_keyframes_.push_back({std::make_shared<KeyFrame>(state, time_ns)});
    if (r.read_bool()) {
      auto preint = std::make_shared<IMUPreintegration>();
      preint->load_state(r);
      consecutive_keyframes_.back().preintegration = std::move(preint);
    }
  }

  Map<TrackId, LandmarkPtr> landmark_table;
  const size_t num_landmarks = r.read_size();
  landmark_table.reserve(num_landmarks);
  for (size_t i = 0; i < num_landmarks; ++i) {
    const TrackId track_id = r.read_pod<TrackId>();
    LandmarkPtr landmark;
    if (r.read_bool()) {
      Vector3T pose;
      r.read_eigen(pose);
      landmark = std::make_shared<Landmark>(track_id, pose);
    } else {
      landmark = std::make_shared<Landmark>(track_id);
    }
    landmark_table.emplace(track_id, std::move(landmark));
  }

  const size_t num_kf_entries = r.read_size();
  for (size_t i = 0; i < num_kf_entries; ++i) {
    const size_t kf_idx = r.read_size();
    if (kf_idx >= consecutive_keyframes_.size()) {
      throw std::runtime_error("cuVSLAM state deserialization: keyframe index out of range");
    }
    const KeyframePtr& kf = consecutive_keyframes_[kf_idx].keyframe;
    auto& landmarks = landmarks_from_keyframe_[kf];
    const size_t num_entries = r.read_size();
    for (size_t j = 0; j < num_entries; ++j) {
      const TrackId track_id = r.read_pod<TrackId>();
      LandmarkAndObserv& lm_obs = landmarks[track_id];
      const auto lm_it = landmark_table.find(track_id);
      if (lm_it == landmark_table.end()) {
        throw std::runtime_error("cuVSLAM state deserialization: landmark id missing from table");
      }
      lm_obs.landmark = lm_it->second;
      const size_t num_obs = r.read_size();
      for (size_t k = 0; k < num_obs; ++k) {
        camera::Observation obs;
        obs.cam_id = r.read_pod<CameraId>();
        obs.id = r.read_pod<TrackId>();
        r.read_eigen(obs.xy);
        r.read_eigen(obs.xy_info);
        lm_obs.observations.try_push_back(obs);
      }
      keyframes_from_landmark_[track_id].insert(kf);
    }
  }
}

}  // namespace cuvslam::map
