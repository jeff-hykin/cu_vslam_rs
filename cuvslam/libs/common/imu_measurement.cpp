
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

#include "common/imu_measurement.h"

namespace cuvslam::imu {

ImuMeasurementStorage::ImuMeasurementStorage(size_t capacity) : CircularBuffer<ImuMeasurement>(capacity) {}

void ImuMeasurementStorage::iterate_since(int64_t since_time_ns, Lambda f) const {
  ImuMeasurement m;
  m.time_ns = since_time_ns;
  auto it = std::lower_bound(begin(), end(), m, [](const ImuMeasurement& lhs, const ImuMeasurement& rhs) {
    return lhs.time_ns < rhs.time_ns;
  });

  for (; it != end(); it++) {
    f(*it);
  };
}

void ImuMeasurementStorage::iterate_from_to(int64_t from_time_ns, int64_t to_time_ns, Lambda f) const {
  if (from_time_ns > to_time_ns) {
    return;
  }

  ImuMeasurement m;
  m.time_ns = from_time_ns;
  auto lower_it = std::lower_bound(begin(), end(), m, [](const ImuMeasurement& lhs, const ImuMeasurement& rhs) {
    return lhs.time_ns < rhs.time_ns;
  });

  m.time_ns = to_time_ns;
  auto upper_it = std::upper_bound(begin(), end(), m, [](const ImuMeasurement& lhs, const ImuMeasurement& rhs) {
    return lhs.time_ns < rhs.time_ns;
  });

  if (lower_it == upper_it && lower_it != end()) {
    f(*lower_it);
  }

  for (; lower_it != upper_it; lower_it++) {
    f(*lower_it);
  };
}

}  // namespace cuvslam::imu
