
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

#include "common/log.h"

#include <cassert>
#include <cstdarg>
#include <thread>

#include "common/thread_name.h"

namespace cuvslam::Trace {

// helper method to convert to std::string any object with stream out operator<<
template <class _T>
std::string ToString(const _T& t) {
  std::ostringstream stream;
  stream << t;
  assert(!stream.fail());
  return stream.str();
}

ThreadName::ThreadName() : name_(ToString(std::this_thread::get_id())) {}

std::string ThreadName::set(const std::string& name) {
  std::string old(name);
  std::swap(old, name_);
  return old;
}

const std::string& ThreadName::get() const { return name_; }

const char* ThreadName::c_str() const { return name_.c_str(); }

void PrintDecoratedMsg(const char* const prefix, const char* const strFormat, ...) {
  std::va_list args;
  va_start(args, strFormat);

  // TODO: (msmirnov) add support for proper logging APIs depending
  // on the platform (e.g. TraceLogging on Windows)
  // For now just printf to stdout: that's what old code was doing anyway
  printf("[%s] ", prefix);
  vprintf(strFormat, args);
  va_end(args);
  printf("\n");
}

Verbosity global_verbosity = Verbosity::None;

void SetVerbosity(Verbosity verbosity) { global_verbosity = verbosity; }

Verbosity GetVerbosity() { return global_verbosity; }

}  // end namespace cuvslam::Trace
