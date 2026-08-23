
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

#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace cuvslam {

template <typename Int>
std::vector<Int> StringToIntVector(const std::string& str, char delim) {
  std::vector<Int> vec;
  std::istringstream istr(str);
  std::string token;
  while (std::getline(istr, token, delim)) {
    auto llval = std::stoll(
        token);  // TODO: support `unsigned long long`? the only case is to input values not fitting in `long long`
    if (std::numeric_limits<Int>::min() > llval || llval > std::numeric_limits<Int>::max()) {
      throw std::out_of_range("'" + token + "' is out of range for Int type (" + typeid(Int).name() + ")");
    }
    Int val = llval;
    vec.push_back(val);
  }
  return vec;
}

}  // namespace cuvslam
