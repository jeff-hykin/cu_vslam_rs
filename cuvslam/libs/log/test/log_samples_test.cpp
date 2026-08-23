
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

#include <chrono>
#include <thread>

#include "json/json.h"

#include "common/include_gtest.h"
#include "common/log.h"
#include "log/log_eigen.h"

namespace test::logger {
using namespace cuvslam;

using LogSamples = log::Enable<LogRoot>;

void Values() {
  bool v1 = false;
  int v2 = 1;
  uint32_t v3 = 2;
  uint64_t v4 = 3;
  float v5 = 4;
  double v6 = 5;
  log::Value<LogSamples>("bool", v1);
  log::Value<LogSamples>("int", v2);
  log::Value<LogSamples>("uint32", v3);
  log::Value<LogSamples>("uint64", v4);
  log::Value<LogSamples>("float", v5);
  log::Value<LogSamples>("double", v6);

  log::Value<LogSamples>("text", "just string");
  log::Value<LogSamples>("std::string", std::string("text2"));

  Isometry3T i1 = Isometry3T::Identity();
  cuvslam::storage::Vec3<float> m1(0, 1, 2);
  cuvslam::Vector2T m2(0, 2);
  cuvslam::Vector3T m3(0, 3, 4);
  cuvslam::Matrix6T m4 = cuvslam::Matrix6T::Zero();
  cuvslam::QuaternionT q1 = cuvslam::QuaternionT::Identity();
  log::Value<LogSamples>("Isometry3T", i1);
  log::Value<LogSamples>("cuvslam::storage::Vec3<float>", m1);
  log::Value<LogSamples>("cuvslam::Vector2T", m2);
  log::Value<LogSamples>("cuvslam::Vector3T", m3);
  log::Value<LogSamples>("cuvslam::Matrix6T", m4);
  log::Value<LogSamples>("cuvslam::QuaternionT", q1);
}

TEST(LogTest, TestAll) {
  // set current logger implementation. null is available and default
  auto logger = log::CreateSpdlogLogger("testall.log");
  log::SetLogger(logger);

  Values();

  // Multithreading
  auto thread_obj = std::thread([&] {
    Vector3T point(0, 1, 2);
    float radius = 10;
    log::Value<LogSamples>("pose_th", point);
    log::Value<LogSamples>("radius_th", radius);
  });
  thread_obj.join();

  std::this_thread::sleep_for(std::chrono::seconds(1));

  using critical = log::Enable<LogRoot>;
  using info = log::Enable<LogRoot>;

  // log1
  using log1 = log::Enable<info>;

  std::vector<float> array = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

  // messages
  std::string text = "ABC";
  log::Message<log1>(log::kCritical, "MESSAGE %s", text.c_str());

  log::Log<log1>([]() { log::Message<log1>(log::kWarn, "MESSAGE"); });
  log::Log<critical>([]() { log::Message<info>(log::kWarn, "MESSAGE"); });

  {
    size_t frame_id = 12;
    log::Scoped<log1> log_frame("frame");
    log::Value<log1>("frame_id", frame_id);
    Vector3T point(0, 1, 2);
    float radius = 10;
    uint32_t color = 0xFFFFFF;

    // json:
    {
      Json::Value waypoint;
      waypoint["pose"][0] = point[0];
      waypoint["pose"][1] = point[1];
      waypoint["pose"][2] = point[2];
      waypoint["radius"] = radius;
      waypoint["color"] = color;
      // 2 ways
      log::Json<log1>(waypoint);
    }

    // value
    {
      // or
      log::Value<log1>("pose", point);
      log::Value<log1>("radius", radius);
      log::Value<log1>("color", color);
      log::Value<log1>("array", array.begin(), array.end());
    }

    // custom:
    {
      log::Log<log1>([&]() {
        // here some calcumation
        auto x = pow(sin(radius), 10);
        log::Value<log1>("x", x);
        // string operations
        log::Message<log1>(log::kDebug, "failed");
      });
    }

    // sub.records
    for (int i = 0; i < 10; i++) {
      log::Scoped<log1> log_track("track2d");

      size_t id = 13;
      Vector2T uv(1, 8);
      log::Value<log1>("id", id);
      log::Value<log1>("uv", uv);
    }
    // sub.records
    for (int i = 0; i < 10; i++) {
      size_t id = 14;
      log::Scoped<log1> log_track("track3d");

      {
        Vector2T uv(2, 9);
        log::Value<log1>("id", id);
        log::Value<log1>("uv", uv);
      }
    }

    /*
    TODO(drobustov): desired behavior

    log1::Message(log::kCritical, "MESSAGE %s", text.c_str());
    critical::Log([]() { critical::Message(log::kWarn, "MESSAGE"); });
    log1::Json(waypoint);
    log1::Value("pose", point);
    log1::Value("radius", radius);
    log1::Value("color", color);
    log1::Message(log::kDebug, "failed");
    */
  }
}

}  // namespace test::logger
