
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

#include <atomic>
#include <memory>
#include <string>

#include "camera/camera.h"
#include "camera/observation.h"
#include "common/state_serial.h"
#include "common/types.h"
#include "common/unaligned_types.h"

namespace cuvslam::sof {

enum class TrackerType { LK, KLT, LKHorizontal, KLTHorizontal };

TrackerType ParseTrackerType(const std::string& name);

class IFeatureTracker;
std::unique_ptr<IFeatureTracker> CreateTracker(TrackerType type);

// Global track id counter accessors. The raw id does not include the camera bits that
// TracksVector::newTrackId() ORs on top. Used by Odometry state serialization to checkpoint and
// restore the counter so that ids allocated after a restore do not collide with restored state.
TrackId AllocateRawTrackId();
TrackId GetNextRawTrackId();
void SetNextRawTrackId(TrackId next_id);

// keep track position in uv space (in float pixels')
class Track {
public:
  Track(const CameraId cam_id, const TrackId id, const Vector2T& position)
      : cam_id_(cam_id), id_(id), position_(position), age_(1) {
    constexpr auto stdPx = float{3};
    info_ << 1.f / (stdPx * stdPx), 0.f, 0.f, 1.f / (stdPx * stdPx);
  }
  const Vector2T& position() const {
    assert(!dead());
    return position_;
  }
  const Matrix2T& info() const {
    assert(!dead());
    return info_;
  }
  void setPosition(const Vector2T& uv) {
    assert(!dead());
    position_ = uv;
  }
  void setInfo(const Matrix2T& uv_info) {
    assert(!dead());
    info_ = uv_info;
  }
  TrackId id() const { return id_; }
  CameraId cam_id() const { return cam_id_; }
  size_t age() const {
    assert(!dead());
    return age_;
  }
  void IncrementAge() {
    assert(!dead());
    age_++;
  }
  bool kill() {
    if (age_ == 0) {
      return false;
    }
    age_ = 0;
    return true;
  }
  bool dead() const { return age_ == 0; }

  void save_state(serial::Writer& w) const {
    w.write_pod(cam_id_);
    w.write_pod(id_);
    w.write_eigen(position_);
    w.write_eigen(info_);
    w.write_pod<uint64_t>(age_);
  }

  static Track load_state(serial::Reader& r) {
    const CameraId cam_id = r.read_pod<CameraId>();
    const TrackId id = r.read_pod<TrackId>();
    Vector2T position;
    r.read_eigen(position);
    Track t(cam_id, id, position);
    r.read_eigen(t.info_);
    t.age_ = static_cast<size_t>(r.read_pod<uint64_t>());
    return t;
  }

private:
  Track() = delete;

  CameraId cam_id_;
  TrackId id_;
  Vector2T position_;
  Matrix2T info_;
  size_t age_;  // start with 1 and ++ in each frame we can track it, set to 0 when dead
};

class TracksVector {
public:
  size_t get_num_alive() const { return num_alive_; }
  size_t size() const { return tracks_.size(); }
  void remove_dead_tracks() {
    if (num_alive_ < tracks_.size()) {
      auto pred = [](const Track& t) { return t.dead(); };
      tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), pred), tracks_.end());
    }
    assert(num_alive_ == tracks_.size());
  }

  void export_to_observations_vector(const camera::ICameraModel& intrinsics,
                                     std::vector<camera::Observation>& frameTracks) const {
    frameTracks.clear();
    frameTracks.reserve(get_num_alive());
    for (const Track& t : tracks_) {
      if (t.dead()) {
        continue;
      }

      Vector2T xy;
      if (!intrinsics.normalizePoint(t.position(), xy)) {
        continue;
      }

      frameTracks.push_back(
          {t.cam_id(), t.id(), xy, camera::ObservationInfoUVToXY(intrinsics, t.position(), xy, t.info())});
    }
  }

  void add(CameraId cam_id, std::vector<Vector2T>& newPoints) {
    tracks_.reserve(tracks_.size() + newPoints.size());
    for (const Vector2T& point : newPoints) {
      tracks_.emplace_back(cam_id, newTrackId(cam_id), point);
    }
    num_alive_ += newPoints.size();
  }
  void kill(size_t index) {
    if (tracks_[index].kill()) {
      --num_alive_;
    }
  }
  void IncrementAge(size_t index) { tracks_[index].IncrementAge(); }
  void sort() {
    std::sort(tracks_.begin(), tracks_.end(), [](const Track& lhs, const Track& rhs) { return lhs.id() < rhs.id(); });
  }
  auto cbegin() const { return tracks_.cbegin(); }
  auto cend() const { return tracks_.cend(); }
  void add(const TracksVector& newTracks) {
    tracks_.reserve(tracks_.size() + newTracks.size());
    std::copy(newTracks.cbegin(), newTracks.cend(), std::back_inserter(tracks_));
    num_alive_ += newTracks.get_num_alive();
  }
  void setPosition(size_t index, const Vector2T& uv) { tracks_[index].setPosition(uv); }
  void setInfo(size_t index, const Matrix2T& uv_info) { tracks_[index].setInfo(uv_info); }
  const Track& operator[](size_t index) const { return tracks_[index]; }
  void reset() {
    tracks_.clear();
    num_alive_ = 0;
  }

  void save_state(serial::Writer& w) const {
    w.write_tag(0x54524B56);  // "TRKV"
    w.write_size(tracks_.size());
    for (const Track& t : tracks_) {
      t.save_state(w);
    }
    w.write_pod<uint64_t>(num_alive_);
  }

  void load_state(serial::Reader& r) {
    r.expect_tag(0x54524B56, "TracksVector");
    const size_t size = r.read_size();
    tracks_.clear();
    tracks_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      tracks_.push_back(Track::load_state(r));
    }
    num_alive_ = static_cast<size_t>(r.read_pod<uint64_t>());
  }

private:
  TrackId newTrackId(CameraId cam_id) const {
    TrackId id_with_cam = AllocateRawTrackId();

    // top 8 bits contain camera id, the rest is for track id.
    id_with_cam = id_with_cam | (TrackId)(cam_id) << 8 * (sizeof(TrackId) - sizeof(CameraId));

    static_assert(sizeof(TrackId) == 8);
    static_assert(sizeof(CameraId) == 1);
    return id_with_cam;
  }

  std::vector<Track> tracks_;
  size_t num_alive_ = 0;  // number of non-dead tracks
};

}  // namespace cuvslam::sof
