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

#include "common/state_serial.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

namespace cuvslam::serial {

TEST(StateSerialTest, PodRoundTrip) {
  Writer w;
  w.write_pod<uint32_t>(0xDEADBEEF);
  w.write_pod<int64_t>(-1234567890123LL);
  w.write_pod<float>(3.25f);
  w.write_bool(true);
  w.write_bool(false);
  w.write_size(42);

  Reader r(w.buffer());
  EXPECT_EQ(r.read_pod<uint32_t>(), 0xDEADBEEFu);
  EXPECT_EQ(r.read_pod<int64_t>(), -1234567890123LL);
  EXPECT_EQ(r.read_pod<float>(), 3.25f);
  EXPECT_TRUE(r.read_bool());
  EXPECT_FALSE(r.read_bool());
  EXPECT_EQ(r.read_size(), 42u);
  EXPECT_TRUE(r.at_end());
}

TEST(StateSerialTest, StringVectorOptionalRoundTrip) {
  Writer w;
  w.write_string("hello state");
  w.write_pod_vector(std::vector<uint8_t>{1, 2, 3, 255});
  w.write_optional_pod(std::optional<int32_t>{-7});
  w.write_optional_pod(std::optional<int32_t>{});

  Reader r(w.buffer());
  EXPECT_EQ(r.read_string(), "hello state");
  EXPECT_EQ(r.read_pod_vector<uint8_t>(), (std::vector<uint8_t>{1, 2, 3, 255}));
  EXPECT_EQ(r.read_optional_pod<int32_t>(), std::optional<int32_t>{-7});
  EXPECT_EQ(r.read_optional_pod<int32_t>(), std::optional<int32_t>{});
  EXPECT_TRUE(r.at_end());
}

TEST(StateSerialTest, EigenAndIsometryRoundTripIsBitExact) {
  Matrix3T m;
  m << 1.5f, -2.25f, 3.0e-7f, 4.0f, 5.5f, -6.0f, 7.0f, 8.0f, 9.125f;
  Isometry3T iso = Isometry3T::Identity();
  iso.translate(Vector3T(0.1f, -0.2f, 0.3f));
  iso.rotate(Eigen::AngleAxisf(0.5f, Vector3T::UnitY()));

  Writer w;
  w.write_eigen(m);
  w.write_isometry(iso);

  Matrix3T m2 = Matrix3T::Zero();
  Isometry3T iso2 = Isometry3T::Identity();
  Reader r(w.buffer());
  r.read_eigen(m2);
  r.read_isometry(iso2);
  // Bit-exactness matters: checkpoint restore must reproduce state exactly.
  EXPECT_EQ(std::memcmp(m.data(), m2.data(), sizeof(float) * 9), 0);
  EXPECT_EQ(std::memcmp(iso.matrix().data(), iso2.matrix().data(), sizeof(float) * 16), 0);
  EXPECT_TRUE(r.at_end());
}

TEST(StateSerialTest, TagMismatchThrows) {
  Writer w;
  w.write_tag(0x11111111);
  Reader r(w.buffer());
  EXPECT_THROW(r.expect_tag(0x22222222, "test section"), std::runtime_error);
}

TEST(StateSerialTest, TruncatedBufferThrows) {
  Writer w;
  w.write_pod<uint64_t>(1);
  Reader r(w.buffer().data(), 4);  // truncated
  EXPECT_THROW(r.read_pod<uint64_t>(), std::runtime_error);
}

}  // namespace cuvslam::serial
