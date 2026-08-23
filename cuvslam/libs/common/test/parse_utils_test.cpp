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

#include <limits>
#include <stdexcept>
#include <string>

#include "common/include_gtest.h"
#include "common/parse_utils.h"

using namespace cuvslam;

TEST(ParseUtils, ParseBool) {
  EXPECT_TRUE(common::ParseBool("true"));
  EXPECT_TRUE(common::ParseBool("TRUE"));
  EXPECT_TRUE(common::ParseBool("TrUe"));
  EXPECT_TRUE(common::ParseBool("1"));
  EXPECT_FALSE(common::ParseBool("false"));
  EXPECT_FALSE(common::ParseBool("FALSE"));
  EXPECT_FALSE(common::ParseBool("0"));
  EXPECT_THROW(common::ParseBool("yes"), std::runtime_error);
}

TEST(ParseUtils, ParseFloat) {
  EXPECT_FLOAT_EQ(common::ParseFloat("1.25"), 1.25f);
  EXPECT_FLOAT_EQ(common::ParseFloat("-2"), -2.f);
  EXPECT_THROW(common::ParseFloat(""), std::runtime_error);
  EXPECT_THROW(common::ParseFloat("abc"), std::runtime_error);
  EXPECT_THROW(common::ParseFloat("1.25abc"), std::runtime_error);
}

TEST(ParseUtils, ParseInt32) {
  EXPECT_EQ(common::ParseInt32("42"), 42);
  EXPECT_EQ(common::ParseInt32("-42"), -42);
  EXPECT_THROW(common::ParseInt32(""), std::runtime_error);
  EXPECT_THROW(common::ParseInt32("42abc"), std::runtime_error);
  EXPECT_THROW(common::ParseInt32(std::to_string(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1)),
               std::runtime_error);
}

TEST(ParseUtils, ParseInt64) {
  EXPECT_EQ(common::ParseInt64("42"), 42);
  EXPECT_EQ(common::ParseInt64("-42"), -42);
  EXPECT_THROW(common::ParseInt64(""), std::runtime_error);
  EXPECT_THROW(common::ParseInt64("abc"), std::runtime_error);
  EXPECT_THROW(common::ParseInt64("42abc"), std::runtime_error);
  EXPECT_THROW(common::ParseInt64("9223372036854775808"), std::runtime_error);
  EXPECT_THROW(common::ParseInt64("-9223372036854775809"), std::runtime_error);
}
