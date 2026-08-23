
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

#include "imu/linear_filter.h"

namespace cuvslam::sba_imu {

LinearFilter::LinearFilter(const std::vector<float>& b_coeff, const std::vector<float>& a_coeff) : b_coeff_(b_coeff) {
  // skip first coeff since its always 1
  std::copy(std::next(a_coeff.begin()), a_coeff.end(), std::back_inserter(a_coeff_));
}

float LinearFilter::add_measurement(float m) {
  input_measurements_.push_front(m);
  while (input_measurements_.size() > b_coeff_.size()) {
    input_measurements_.pop_back();
  }
  float out = 0;
  {
    auto it = input_measurements_.begin();
    for (float b : b_coeff_) {
      out += b * (*it);
      it++;
    }
  }
  {
    auto it = output_measurements_.begin();
    for (float a : a_coeff_) {
      out -= a * (*it);
      it++;
    }
  }
  output_measurements_.push_front(out);
  while (output_measurements_.size() > a_coeff_.size()) {
    output_measurements_.pop_back();
  }
  return out;
}

LinearFilter3::LinearFilter3(const std::vector<float>& b_coeff, const std::vector<float>& a_coeff)
    : x_filter(b_coeff, a_coeff), y_filter(b_coeff, a_coeff), z_filter(b_coeff, a_coeff) {}

Vector3T LinearFilter3::add_measurement(const Vector3T& m) {
  return {x_filter.add_measurement(m.x()), y_filter.add_measurement(m.y()), z_filter.add_measurement(m.z())};
}

}  // namespace cuvslam::sba_imu
