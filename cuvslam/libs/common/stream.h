
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

#include <cstdint>
#include <experimental/iterator>
#include <iomanip>
#include <iostream>
#include <tuple>

#include "common/coordinate_system.h"

namespace cuvslam {

enum class PoseFormat : uint8_t { MATRIX, TUM };

class PoseIOManip {
public:
  PoseIOManip(PoseFormat format, bool ros_conversion) : format_(format), ros_conversion_(ros_conversion) {}

  // store Pose IO settings in ostream
  friend std::ostream& operator<<(std::ostream& os, const PoseIOManip& manip) {
    os.iword(getIndex()) = manip.toLong();
    return os;
  }

  // get stored settings from a stream; to be used by operator<< implementations
  static std::tuple<PoseFormat, bool> getSettings(std::ios_base& os) {
    long settings = os.iword(getIndex());
    return {static_cast<PoseFormat>(settings >> 1), ((settings & 1) == 1)};
  }

private:
  // combine settings into a long value to store in stream's iword
  long toLong() const { return (static_cast<uint8_t>(format_) << 1) + (ros_conversion_ ? 1 : 0); }

  // all objects of this class will use the same iword
  static int getIndex() {
    static int index = std::ios_base::xalloc();
    return index;
  }

private:
  PoseFormat format_;
  bool ros_conversion_;
};

std::ostream& operator<<(std::ostream& stream, const cuvslam::Isometry3T& pose) {
  const auto [format, ros_conversion] = PoseIOManip::getSettings(stream);
  cuvslam::Isometry3T iso_pose(pose);
  if (ros_conversion) {
    // Poses from the public API are in OpenCV coordinates (x-right, y-down, z-forward).
    // Convert to ROS coordinates (x-forward, y-left, z-up).
    // ROS_x = OpenCV_z, ROS_y = -OpenCV_x, ROS_z = -OpenCV_y
    static const Isometry3T kRosFromOpencv = [] {
      Isometry3T t = Isometry3T::Identity();
      t.linear() << 0, 0, 1, -1, 0, 0, 0, -1, 0;
      return t;
    }();
    static const Isometry3T kOpencvFromRos{kRosFromOpencv.inverse()};
    iso_pose = kRosFromOpencv * iso_pose * kOpencvFromRos;
  }
  if (format == PoseFormat::MATRIX) {
    auto t = iso_pose.translation();
    auto r = iso_pose.linear();
    std::array<float, 12> data = {r(0, 0), r(1, 0), r(2, 0), t(0),    r(0, 1), r(1, 1),
                                  r(2, 1), t(1),    r(0, 2), r(1, 2), r(2, 2), t(2)};
    std::copy(std::begin(data), std::end(data), std::experimental::make_ostream_joiner(stream, " "));
  } else if (format == PoseFormat::TUM) {
    auto t = iso_pose.translation();
    Eigen::Quaternionf q(iso_pose.linear());
    std::array<float, 7> data = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
    std::copy(std::begin(data), std::end(data), std::experimental::make_ostream_joiner(stream, " "));
  } else {
    throw std::invalid_argument("Unknown output pose format " + std::to_string(static_cast<uint8_t>(format)));
  }
  return stream;
}

// int64_t timestamp wrapper to format in ostream according to PoseIOManip settings
struct Timestamp {
  int64_t ts_;
  Timestamp(int64_t ts) : ts_(ts) {}
};

std::ostream& operator<<(std::ostream& stream, const Timestamp& ts) {
  const auto [format, _] = PoseIOManip::getSettings(stream);
  if (format == PoseFormat::TUM) {
    stream << static_cast<double>(ts.ts_) / 1e9;
  } else {
    stream << ts.ts_;
  }
  return stream;
}

}  // namespace cuvslam
