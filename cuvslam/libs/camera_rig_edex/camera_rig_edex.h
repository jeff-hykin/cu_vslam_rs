
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

#include <filesystem>

#include "camera/camera.h"
#include "common/imu_measurement.h"
#include "common/interfaces.h"
#include "edex/edex.h"
#include "edex/timeline.h"

namespace cuvslam::camera_rig_edex {

class ICameraRigReplay : public ICameraRig {
public:
  virtual const std::vector<CameraId>& getCameraIds() const = 0;
  virtual bool setCurrentFrame(int frame) = 0;
};

class CameraRigEdex : public ICameraRigReplay {
protected:
  enum EventType { Frame, IMU };
  struct FrameMetadataPerCam {
    std::string filename;
    int64_t timestamp_ns;
    std::optional<uint8_t> depth_id;
    std::optional<std::string> depth_filename;
  };
  struct FrameMetadata {
    FrameId frame_id;
    std::unordered_map<CameraId, FrameMetadataPerCam> camera;
    // check ho it used
  };
  struct Event {
    EventType type;
    imu::ImuMeasurement imu_measurement;
    FrameMetadata frame_metadata;
  };
  struct CameraEdex {
    std::unique_ptr<camera::ICameraModel> intrinsic;
    Isometry3T transform;
    std::vector<uint8_t> data;
    std::vector<uint8_t> mask_data;
    std::optional<uint8_t> depth_id;
    std::vector<float> depth;
    bool has_depth = false;
  };

  edex::EdexFile edex_;
  std::string edexFileName_;
  std::filesystem::path seqPath_;
  edex::Timeline timeline_;
  std::vector<CameraEdex> cameras_;
  std::vector<CameraId> useCameras_;
  std::vector<Event> events_;
  size_t eventIndex_;
  size_t frameIndex_;  // just a number of the replayed frames
  bool replayForward_;
  std::function<void(const imu::ImuMeasurement&)> imuCallback_;

  bool read_frame_metadata(const std::string& filename);
  bool read_imu_data(const std::string& filename, const Matrix3T& transform);
  bool load_edex();
  int64_t get_event_timestamp(const Event& event);

public:
  // Paths stored inside the edex (image sequences, IMU log, frame-metadata) are resolved relative
  // to the edex file's directory.
  explicit CameraRigEdex(const std::string& edexFileName, const std::vector<CameraId>& useCameras = {});
  int64_t getFirstTimestamp() const;
  int64_t getLastTimestamp() const;
  bool isFinished() const { return eventIndex_ >= events_.size(); };
  // timestamps will be the same regardless of the direction of the replay, so they will decrease when replaying
  // backwards
  void setReplayForward(bool forward);
  bool getReplayForward() const;

  // ICameraRig
  ErrorCode start() override;
  ErrorCode stop() override;
  ErrorCode getFrame(Sources& sources, Metas& metas, Sources& masks_sources, DepthSources& depth_sources) override;
  uint32_t getCamerasNum() const override;
  std::vector<CameraId> getCamerasWithDepth() const override;
  const camera::ICameraModel& getIntrinsic(uint32_t index) const override;
  const Isometry3T& getExtrinsic(uint32_t index) const override;
  void registerIMUCallback(const std::function<void(const imu::ImuMeasurement&)>& func) override;

  // ICameraRigReplay
  const std::vector<CameraId>& getCameraIds() const override { return useCameras_; };
  bool setCurrentFrame(int frame) override;
};

}  // namespace cuvslam::camera_rig_edex
