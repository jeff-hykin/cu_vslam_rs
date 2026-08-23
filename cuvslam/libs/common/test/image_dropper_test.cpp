
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

#include <iostream>
#include <random>

#include "common/image_dropper.h"
#include "common/include_gtest.h"
using namespace cuvslam;

TEST(ImageDropper, Demo) {
  const uint32_t num = 8;
  const uint32_t len = 79;
  const double drop_rate = 0.25;
  std::cout << "Target drop rate = " << drop_rate << std::endl;
  std::array<ImageDropperType, 3> types{ImageDropperType::Steady, ImageDropperType::Normal, ImageDropperType::Sticky};
  for (const auto& type : types) {
    size_t drops = 0;
    std::vector<std::string> sequences{num};
    auto dropper = CreateImageDropper(type, std::mt19937(::testing::UnitTest::GetInstance()->random_seed()));
    for (uint32_t rep = 0; rep < len; rep++) {
      auto dropped_images = dropper->GetDroppedImages(drop_rate, num);
      drops += dropped_images.size();
      for (uint32_t i = 0; i < num; i++) sequences[i] += dropped_images.find(i) == dropped_images.end() ? '.' : 'X';
    }
    for (uint32_t i = 0; i < num; i++) {
      std::cout << sequences[i] << std::endl;
    }
    std::cout << "Test drop rate = " << static_cast<double>(drops) / num / len << std::endl << std::endl;
  }
}

using DropperSuite = testing::TestWithParam<std::tuple<ImageDropperType, double>>;

INSTANTIATE_TEST_SUITE_P(ImageDropper, DropperSuite,
                         testing::Combine(testing::Values(ImageDropperType::Steady, ImageDropperType::Normal,
                                                          ImageDropperType::Sticky),
                                          testing::Values(0., 0.001, 0.01, 0.1, 0.25, 0.5)));

TEST_P(DropperSuite, KeepsAverageDropRate) {
  const auto [type, rate] = GetParam();
  const int num = 8;
  const int len = 100000;
  std::vector<int> drops(num, 0);
  int total_drops = 0;
  auto dropper = CreateImageDropper(type, std::mt19937(::testing::UnitTest::GetInstance()->random_seed()));
  for (int rep = 0; rep < len; rep++) {
    auto dropped = dropper->GetDroppedImages(rate, num);
    for (auto i : dropped) {
      drops[i]++;
    }
  }
  // sticky dropper has larger variation (due to stickiness)
  const auto type_mul = type == ImageDropperType::Sticky ? 5. : 1.;
  for (int i = 0; i < num; i++) {
    const auto max_deviation = std::sqrt(len * rate * (1 - rate)) * 6 * type_mul;  // "6 sigma"
    EXPECT_NEAR(rate * len, drops[i], max_deviation);
    total_drops += drops[i];
  }
  const auto max_deviation = std::sqrt(num * len * rate * (1 - rate)) * 6 * type_mul;  // "6 sigma"
  EXPECT_NEAR(rate * num * len, total_drops, max_deviation);
}

using RatesSuite = testing::TestWithParam<double>;

INSTANTIATE_TEST_SUITE_P(ImageDropper, RatesSuite, testing::Values(0., 0.001, 0.01, 0.1, 0.25, 0.5));

TEST_P(RatesSuite, SteadyDrops) {
  const auto rate = GetParam();
  const int num = 8;
  const int len = 1000;
  auto dropper =
      CreateImageDropper(ImageDropperType::Steady, std::mt19937(::testing::UnitTest::GetInstance()->random_seed()));
  for (int rep = 0; rep < len; rep++) {
    auto dropped = dropper->GetDroppedImages(rate, num);
    EXPECT_LT(std::fabs(rate * num - dropped.size()), 1.);
  }
}
