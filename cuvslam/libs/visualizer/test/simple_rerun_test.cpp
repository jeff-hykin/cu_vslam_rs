

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

#include "common/include_gtest.h"
#include "rerun.hpp"

// Test basic rerun functionality without test fixtures
TEST(SimpleRerunTest, TestBasicLogging) {
  // Create a test recording stream
  auto recording = rerun::RecordingStream("simple_test");

  // Test logging text
  recording.log("test/text", rerun::TextLog("Simple test message"));

  // The test succeeds if no exceptions are thrown
  SUCCEED();
}

// Test 3D point logging
TEST(SimpleRerunTest, TestPointsLogging) {
  auto recording = rerun::RecordingStream("points_test");

  std::vector<rerun::Position3D> points = {{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, {2.0f, 0.0f, 2.0f}};

  recording.log("test/3d/points", rerun::Points3D(points));

  EXPECT_EQ(points.size(), 3u);
}

// Test 2D point logging
TEST(SimpleRerunTest, Test2DPointsLogging) {
  auto recording = rerun::RecordingStream("points2d_test");

  std::vector<rerun::Position2D> points_2d = {{100.0f, 50.0f}, {200.0f, 100.0f}, {300.0f, 150.0f}};

  recording.log("test/2d/points", rerun::Points2D(points_2d));

  EXPECT_EQ(points_2d.size(), 3u);
}

// Test color functionality
TEST(SimpleRerunTest, TestColors) {
  auto recording = rerun::RecordingStream("colors_test");

  std::vector<rerun::Position3D> points = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};

  std::vector<rerun::Color> colors = {
      {255, 0, 0},  // Red
      {0, 255, 0}   // Green
  };

  recording.log("test/colored_points", rerun::Points3D(points).with_colors(colors));

  EXPECT_EQ(points.size(), colors.size());
}

// Test image logging
TEST(SimpleRerunTest, TestImageLogging) {
  auto recording = rerun::RecordingStream("image_test");

  // Test grayscale image (8-bit)
  constexpr uint32_t width = 64;
  constexpr uint32_t height = 48;
  std::vector<uint8_t> grayscale_data(width * height);

  // Fill with a simple pattern
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      grayscale_data[y * width + x] = static_cast<uint8_t>((x + y) % 256);
    }
  }

  // Log grayscale image
  recording.log("test/images/grayscale",
                rerun::Image(grayscale_data.data(), {width, height}, rerun::datatypes::ColorModel::L));

  // Test RGB image
  constexpr uint32_t rgb_width = 32;
  constexpr uint32_t rgb_height = 24;
  std::vector<uint8_t> rgb_data(rgb_width * rgb_height * 3);

  // Fill with RGB pattern
  for (uint32_t y = 0; y < rgb_height; ++y) {
    for (uint32_t x = 0; x < rgb_width; ++x) {
      const uint32_t idx = (y * rgb_width + x) * 3;
      rgb_data[idx + 0] = static_cast<uint8_t>(x * 8);        // Red
      rgb_data[idx + 1] = static_cast<uint8_t>(y * 10);       // Green
      rgb_data[idx + 2] = static_cast<uint8_t>((x + y) * 4);  // Blue
    }
  }

  // Log RGB image
  recording.log("test/images/rgb",
                rerun::Image(rgb_data.data(), {rgb_width, rgb_height}, rerun::datatypes::ColorModel::RGB));

  EXPECT_EQ(grayscale_data.size(), width * height);
  EXPECT_EQ(rgb_data.size(), rgb_width * rgb_height * 3);
}

// Test clearing
TEST(SimpleRerunTest, TestClear) {
  auto recording = rerun::RecordingStream("clear_test");

  // Log something first
  recording.log("test/to_clear", rerun::TextLog("Will be cleared"));

  // Clear it
  recording.log("test/to_clear", rerun::Clear());

  SUCCEED();
}
