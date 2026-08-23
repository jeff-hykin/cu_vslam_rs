
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

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

#include "circular_buffer.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "common/vector_3t.h"

namespace cuvslam::imu {

struct ImuMeasurement {
  int64_t time_ns;               // in nanoseconds
  Vector3T linear_acceleration;  // in meters per squared second
  Vector3T angular_velocity;     // in radians per second
};

class ImuMeasurementStorage : public CircularBuffer<ImuMeasurement> {
public:
  explicit ImuMeasurementStorage(size_t capacity);
  using Lambda = const std::function<void(const ImuMeasurement& m)>&;
  void iterate_since(int64_t since_time_ns, Lambda f) const;
  void iterate_from_to(int64_t from_time_ns, int64_t to_time_ns, Lambda f) const;
};

}  // namespace cuvslam::imu
