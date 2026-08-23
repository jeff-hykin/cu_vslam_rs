
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

#include <thread>

#include "log/logger_interface.h"

#ifdef CUVSLAM_LOG_ENABLE

#include "json/json.h"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/spdlog.h"

namespace cuvslam::log {

namespace {
void replace_all(std::string& str, const std::string& from, const std::string& to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
  }
}
}  // namespace

class SpdlogLogger : public ILogger {
public:
  SpdlogLogger(const char* filename) {
    this->main_thread_id_ = std::this_thread::get_id();

    this->builder_["indentation"] = "";

    const std::string name = "async_cuvslam_logger";
    this->async_file_ = spdlog::get(name);

    if (!this->async_file_) {
      bool truncate = true;
      this->async_file_ = spdlog::basic_logger_mt<spdlog::async_factory>(name, filename, truncate);
      this->async_file_->set_pattern("%v");
    }
  }
  virtual ~SpdlogLogger() {}

  virtual void Message(Levels /* level */, const char* text) override {
    if (!text) return;
    // TODO: remove \n from text
    std::string str = text;
    replace_all(str, "\n", "");

    // TODO: level
    auto spdlog_level = spdlog::level::info;
    this->async_file_->log(spdlog_level, "{}{{ \"level\": \"{}\", \"message\" : \"{}\" }}", ThreadName(), "info",
                           str.c_str());
    this->async_file_->flush();
  }
  virtual void Value(const char* name, const char* value) override {
    if (!name || !value) return;
    // TODO: level
    auto spdlog_level = spdlog::level::info;
    this->async_file_->log(spdlog_level, "{}{{ \"{}\": {} }}", ThreadName(), name, value);
    this->async_file_->flush();
  }
  virtual void Json(const Json::Value& root) override {
    std::string jsonStr = Json::writeString(builder_, root);
    replace_all(jsonStr, "\n", "");

    // TODO: level
    auto spdlog_level = spdlog::level::info;
    this->async_file_->log(spdlog_level, ThreadName() + jsonStr);
    this->async_file_->flush();
  }
  virtual void BeginScope(const char* name) override {
    auto spdlog_level = spdlog::level::info;
    this->async_file_->log(spdlog_level, "{}{{ \"scope\":1, \"name\": \"{}\" }}", ThreadName(), name);
    this->async_file_->flush();

    intend_ += 2;
  }
  virtual void EndScope() override {
    intend_ = std::max(0, intend_ - 2);

    auto spdlog_level = spdlog::level::info;
    this->async_file_->log(spdlog_level, "{}{{ \"scope\":0 }}", ThreadName());
    this->async_file_->flush();
  }

protected:
  std::shared_ptr<spdlog::logger> async_file_;
  int intend_ = 0;
  std::thread::id main_thread_id_;
  Json::StreamWriterBuilder builder_;

  std::string ThreadName() {
    std::string treadidstr;
    auto tid = std::this_thread::get_id();
    if (this->main_thread_id_ == tid) return "";

    std::ostringstream ss;
    ss << '[' << tid << ']';
    return ss.str();
  }
};

std::unique_ptr<ILogger> CreateSpdlogLogger(const char* filename) { return std::make_unique<SpdlogLogger>(filename); }

}  // namespace cuvslam::log

#else  // CUVSLAM_LOG_ENABLE

namespace cuvslam::log {

std::unique_ptr<ILogger> CreateSpdlogLogger(const char*) { return nullptr; }

}  // namespace cuvslam::log

#endif  // CUVSLAM_LOG_ENABLE
