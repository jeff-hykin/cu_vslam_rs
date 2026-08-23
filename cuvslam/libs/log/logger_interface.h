
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

#include <memory>

/*
Backend Log/trace interface for cuVSLAM

Implement ILogger interface and pass instance to cuvslam::Log::setLogger()
*/

// Json::Value forward declaration
namespace Json {
class Value;
}

namespace cuvslam::log {

enum Levels { kTrace = 5, kDebug = 4, kInfo = 3, kWarn = 2, kErr = 1, kCritical = 0, kMaxlevel };

// interface for logger
class ILogger {
public:
  virtual void Message(Levels level, const char* text) = 0;
  virtual void Value(const char* name, const char* value) = 0;
  virtual void Json(const Json::Value& root) = 0;

  virtual void BeginScope(const char* name) = 0;
  virtual void EndScope() = 0;

  virtual ~ILogger() {}

public:
  static std::unique_ptr<ILogger> current_logger_;
};

// set current logger implementation.
void SetLogger(std::unique_ptr<ILogger>& logger);
// set null logger
void SetLogger();

// create ILogger Spdlog implementation
std::unique_ptr<ILogger> CreateSpdlogLogger(const char* filename);

}  // namespace cuvslam::log
