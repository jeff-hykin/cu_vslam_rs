
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

#include <cstddef>

namespace cuvslam {

/*
This class supports only push_back and const range-based iteration.
It is made so to keep it as small as possible for a fast copy.
*/

template <class T, size_t Capacity>
class FixedArray {
public:
  /*since "push_back" is supposed to thow exceptions, we call the method try_push_back*/
  bool try_push_back(const T& value) noexcept {
    if (size_ < Capacity) {
      array_[size_++] = value;
      return true;
    }
    return false;
  }

  size_t size() const { return size_; }
  bool full() const { return Capacity == size_; }

  const T* begin() const { return &array_[0]; }
  const T* end() const { return &array_[size_]; }

private:
  T array_[Capacity];
  size_t size_ = 0;
};

}  // namespace cuvslam
