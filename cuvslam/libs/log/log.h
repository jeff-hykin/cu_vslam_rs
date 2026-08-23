
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

/*
Frontend Log/trace template library for cuVSLAM
This header is low cost. It can be included everywhere

Usage:
1. log::Message()         : formatted text
2. log::Value(key, value) : log key-value pair (function std::to_string(VALUE) must be exists)
3. log::Json(json)        : log json
4. log::Log([&](){ ... }) : call lambda. use it to suppress code which is not needed if log is off
5. log::Scoped sk;        : allow to group values and jsons records

Template argument ENABLE must be set to Log::Enable or Log::Disable.
In log::Disable case code with logging will not be in the build.

Sample:
using log_test = log::Enable<>;     // log_test is enabled

log::Message<log_test>(critical, "imageName");
log::Value<log_test>("imageName", filename);

for(;;)
{
    log::Scoped<log_test> frame("frame");
    log::Value<log_test>("attr1", attr1);
    log::Value<log_test>("attr2", attr2);
}

// here: code with Json record will not be exists in the build
log::Log<log_test>([&]() {
    Json::Value root;
    root["frameId"] = static_cast<int>(frameId);
    root["solutionFound"] = solutionFound ? 1 : 0;
    root["keyframe"] = stat.keyframe ? 1 : 0;
    Log::Json(root);
    }


Tips:
* Include log_eigen to use eigen clases in Value() function
#include "log/log_eigen.h"
log::Value("3d", vector);

*/

#include <cstdarg>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "log/log_enable.h"
#include "log/logger_interface.h"

namespace cuvslam::log {

// you must specify log type!!!
using ENABLE_default = int;

// message
template <class ENABLE = ENABLE_default, typename... Args, std::enable_if_t<ENABLE::enable_, int> = 0>
void Message(Levels level, const char* text, ...) {
  if (ILogger::current_logger_) {
    va_list args1;
    va_start(args1, text);
    va_list args2;
    va_copy(args2, args1);
    std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, text, args1));
    va_end(args1);
    std::vsnprintf(buf.data(), buf.size(), text, args2);
    va_end(args2);

    ILogger::current_logger_->Message(level, buf.data());
  }
}
// message:disabled
template <class ENABLE = ENABLE_default, typename... Args, std::enable_if_t<!ENABLE::enable_, int> = 0>
void Message(Levels, const char*, Args&&...) {}

// value
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, std::string>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const char* name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name, ("\"" + value + "\"").c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_convertible<T, const char*>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const char* name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name, ("\"" + std::string(value) + "\"").c_str());
}

// use std::to_string(value) for std::is_ariphmetic<>
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const char* name, const T& value) {
  if (ILogger::current_logger_) {
    std::ostringstream oss;
    oss.precision(std::numeric_limits<T>::max_digits10);
    oss << value;
    ILogger::current_logger_->Value(name, oss.str().c_str());
  }
}

// array via std::to_string()
template <class ENABLE = ENABLE_default, class Iterator, std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const char* name, Iterator begin, Iterator end) {
  if (ILogger::current_logger_) {
    std::string text = "[";
    for (auto it = begin; it != end; it++) {
      if (it != begin) text += ", ";
      text.append(std::to_string(*it));
    }
    text.append("]");
    ILogger::current_logger_->Value(name, text.c_str());
  }
}

// value:disabled
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<!ENABLE::enable_, int> = 0>
void Value(const char*, const T&) {}

// array value:disabled
template <class ENABLE = ENABLE_default, class Iterator, std::enable_if_t<!ENABLE::enable_, int> = 0>
void Value(const char*, Iterator, Iterator) {}

template <class ENABLE = ENABLE_default, std::enable_if_t<ENABLE::enable_, int> = 0>
void Json(const Json::Value& root) {
  if (ILogger::current_logger_) ILogger::current_logger_->Json(root);
}
template <class ENABLE = ENABLE_default, std::enable_if_t<!ENABLE::enable_, int> = 0>
void Json(const Json::Value& /* root */) {}

// custom. Via lambda
template <class ENABLE = ENABLE_default, class FUNC, std::enable_if_t<ENABLE::enable_, int> = 0>
void Log(FUNC&& lambda) {
  if (ILogger::current_logger_) {
    lambda();
  }
}
// log: disabled
template <class ENABLE = ENABLE_default, class FUNC, std::enable_if_t<!ENABLE::enable_, int> = 0>
void Log(FUNC&&) {}

// scoped
template <class ENABLE = ENABLE_default>
class Scoped {
public:
  Scoped(const char* text) { this->Begin<ENABLE>(text); }

  ~Scoped() { this->End<ENABLE>(); }

private:
  template <class T, std::enable_if_t<T::enable_, int> = 0>
  void Begin(const char* text) {
    if (ILogger::current_logger_) {
      ILogger::current_logger_->BeginScope(text);
    }
  }
  template <class T, std::enable_if_t<!T::enable_, int> = 0>
  void Begin(const char*) {}
  template <class T, std::enable_if_t<T::enable_, int> = 0>
  void End() {
    if (ILogger::current_logger_) {
      ILogger::current_logger_->EndScope();
    }
  }
  template <class T, std::enable_if_t<!T::enable_, int> = 0>
  void End() {}
};

}  // namespace cuvslam::log
