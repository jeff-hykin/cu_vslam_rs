
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

#include <list>
#include <vector>

#include "common/vector_3t.h"

namespace cuvslam::sba_imu {

class LinearFilter {
public:
  LinearFilter() = default;
  LinearFilter(const std::vector<float>& b_coeff, const std::vector<float>& a_coeff);

  float add_measurement(float m);

private:
  std::list<float> input_measurements_;
  std::list<float> output_measurements_;
  std::vector<float> b_coeff_ = {0.01903746, 0.01304126, 0.06372939, 0.05650231, 0.10140835, 0.08570455,
                                 0.10140835, 0.05650231, 0.06372939, 0.01304126, 0.01903746};
  std::vector<float> a_coeff_ = {-3.15684036, 6.73821606,  -9.31537759, 9.88689107,  -7.7265588,
                                 4.69018777,  -2.07057786, 0.66622515,  -0.13270214, 0.01368564};
};

class LinearFilter3 {
public:
  LinearFilter3() = default;
  LinearFilter3(const std::vector<float>& b_coeff, const std::vector<float>& a_coeff);

  Vector3T add_measurement(const Vector3T& m);

private:
  LinearFilter x_filter;
  LinearFilter y_filter;
  LinearFilter z_filter;
};

}  // namespace cuvslam::sba_imu
