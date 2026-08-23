
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

#include <chrono>
#include <sstream>
#include <string>

#include "common/log.h"

namespace cuvslam {

/*
Usage:
#include "slam/stopwatch.h"
Stopwatch stopwatch;
for(;;) {
    cuvslam::StopwatchScope ssw(stopwatch);
    // Stopwatced code ...
}

std::cout << "Stopwatced code: " << stopwatch.Verbose() << std::endl;
log::Value<LogRoot>("Stopwatced_code", stopwatch);

*/

class Stopwatch {
public:
  unsigned int counts = 0;
  std::chrono::nanoseconds duration;

public:
  explicit Stopwatch() { duration = duration.zero(); }
  std::string Verbose() const {
    double d = Seconds();
    double mean = counts ? d / counts : 0;
    std::stringstream ss;
    ss << counts << " calls " << d << "sec: " << mean << " sec per call";
    return ss.str();
  }
  double Seconds() const { return duration.count() / double(1000000000); }
  unsigned int Times() const { return counts; }
};

class StopwatchScope {
  Stopwatch* stopwatch_ = nullptr;
  std::chrono::high_resolution_clock::time_point start_;

public:
  explicit StopwatchScope(Stopwatch& stopwatch) : stopwatch_(&stopwatch) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  double Stop() {
    if (!stopwatch_) {
      return 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds d = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
    stopwatch_->duration += d;
    stopwatch_->counts++;
    stopwatch_ = nullptr;
    return d.count() / double(1000000000);
  }
  ~StopwatchScope() { this->Stop(); }
};

namespace log {

template <class ENABLE = ENABLE_default, std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const char* name, const cuvslam::Stopwatch& value) {
  if (ILogger::current_logger_) {
    std::stringstream ss;
    ss << "[";
    ss << value.Seconds();
    ss << ", ";
    ss << value.Times();
    ss << "]";
    ILogger::current_logger_->Value(name, ss.str().c_str());
  }
}
}  // namespace log

}  // namespace cuvslam
