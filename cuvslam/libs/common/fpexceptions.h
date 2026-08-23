
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

// Code in this file is largely based on Bruce Dawson's article
// on floating point exceptions:
// https://www.gamasutra.com/view/news/169203/Exceptional_floating_point.php
//
// See https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/controlfp-s
// for documentation on _controlfp_s
//
// For Linux version please refer
// https://linux.die.net/man/3/feenableexcept
//
// WARNING!
// Do not use any of the functionality in this file in the production code.
// It is only intended for development purposes.

#include <cfloat>

#ifndef _MSC_VER
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cfenv>
#endif

namespace cuvslam {
// Enables floating point exception for the duration of the object's lifetime.
class FPExceptionEnabler {
public:
  explicit FPExceptionEnabler(
#ifdef _MSC_VER
      unsigned int enable_bits = _EM_OVERFLOW | _EM_ZERODIVIDE | _EM_INVALID
#else
      int flags = FE_OVERFLOW | FE_DIVBYZERO | FE_INVALID
#endif
  ) {
#ifdef _MSC_VER
    _controlfp_s(&previous_control_value_, _MCW_EM, _MCW_EM);
    enable_bits &= _MCW_EM;

    // clear any pending exceptions
    _clearfp();
    _controlfp_s(0, ~enable_bits, _MCW_EM);
#else
    previous_value_ = fegetexcept();
    feclearexcept(flags);
    feenableexcept(flags);
    excepts_ = flags;
#endif
  }

  ~FPExceptionEnabler() {
#ifdef _MSC_VER
    _controlfp_s(0, previous_control_value_, _MCW_EM);
#else
    // disable newly enabled exceptions
    fedisableexcept(excepts_);
    // re-enable previously enabled exceptions (if any)
    feenableexcept(previous_value_);
#endif
  }

private:
#ifdef _MSC_VER
  unsigned int previous_control_value_;
#else
  int previous_value_;
  int excepts_;
#endif
};

#define CUVSLAM_ENABLE_FPE_IN_CURRENT_SCOPE(...) \
  ::cuvslam::FPExceptionEnabler fpe_enabler_##__COUNTER__ { __VA_ARGS__ }

}  // namespace cuvslam
