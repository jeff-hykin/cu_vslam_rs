
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

// note - we expect only float data in json,
// it possible to use doubles in json, in future releases
// we assume data filled frame by frame.
#include <mutex>

#include "common/isometry_utils.h"
#include "common/types.h"
#include "common/vector_3t.h"

#include "edex/edex_types.h"
#include "edex/timeline.h"

namespace Json {
class Value;
}

namespace cuvslam::edex {

/**
 * Load poses from KITTI format file.
 * Each line contains 12 floats: row-major 3x4 transformation matrix.
 * @param file_name Path to the poses file
 * @param poses Output vector of poses
 * @return true if poses were loaded successfully, false otherwise
 */
bool loadPoses(const std::string& file_name, Isometry3TVector& poses);

/**
 * Write poses to KITTI format file.
 * @param file_name Path to the poses file
 * @param poses Input vector of poses
 * @return true if poses were written successfully, false otherwise
 */
bool writePoses(const std::string& file_name, const Isometry3TVector& poses);

class EdexFile {
public:
  std::string version_;
  Timeline timeline_;
  FrameSet keyFrames_;
  FrameSet failedFrames_;
  Cameras cameras_;  // 0 - left, 1 - right in case of stereo rig
  IMU imu_;
  CameraMap rigPositions_;            // per frame
  CameraMap loop_closure_positions_;  // loop closure poses per frame and timestamp
  Tracks3DMap tracks3D_;              // per track
  RotationStyle rotStyle_;
  std::string frame_meta_log_path_;  // default is empty string
  double fps_ = 0.0;                 // valid only if frame_meta_log_path_ is empty

  EdexFile();
  explicit EdexFile(RotationStyle rs);

  bool read(const std::string& fileName);
  bool write(const std::string& fileName) const;

private:
  std::mutex& getMutex() const;
  bool readHeader(const Json::Value& header);
  bool readBody(const Json::Value& body);
  void writeHeader(Json::Value& header) const;
  void writeBody(Json::Value& body) const;
};

}  // namespace cuvslam::edex
