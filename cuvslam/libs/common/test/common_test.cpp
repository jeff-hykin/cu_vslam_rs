
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

#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "common/image.h"
#include "common/include_gtest.h"
#include "common/isometry_utils.h"
#include "common/statistic.h"
#include "common/vector_3t.h"

namespace test {
using namespace cuvslam;

// This test should be always disabled
// We check for compile-time errors but discard runtime verification
TEST(TestcuVSLAMTraceMacros, DISABLED_TraceMacros) {
  TraceError("Hello from Tracer!");
  TraceMessage("Hello from %s", "Format Tracer");
}

TEST(TestcuVSLAMTraceMacros, TraceMacroSpew) {
  TraceMessage("Ignore: This is TEST output from %s.\r\n", "TraceMessage Macro");
}

TEST(TestcuVSLAMLimits, TestBasicTrigLimits) {
  const float theta = epsilon();
  const float sin = std::sin(theta);
  const float cos = std::cos(theta);
  const float tan = std::tan(theta / 2);
  const float sin_theta = sin / theta;
  const float theta_tan = (theta / 2) / tan;
  const float cos_theta = (1 - cos) / theta;
  EXPECT_EQ(sin, theta);
  EXPECT_EQ(cos, 1.f);
  EXPECT_EQ(sin_theta, 1.f);
  EXPECT_EQ(theta_tan, 1.f);
  EXPECT_EQ(cos_theta, 0.f);
}

TEST(TestCameraIndexedContainers, SparseFrameSentinels) {
  constexpr size_t kNumCameras = 4;
  Sources sources(kNumCameras);
  Sources masks(kNumCameras);
  Metas metas(kNumCameras);
  DepthSources depth_sources(kNumCameras);

  uint8_t image0 = 1;
  uint8_t image2 = 2;
  uint8_t mask2 = 3;
  float depth0 = 1.0f;

  sources[0].data = &image0;
  sources[2].data = &image2;
  masks[2].data = &mask2;
  depth_sources[0].type = ImageSource::F32;
  depth_sources[0].data = &depth0;
  metas[0].camera_index = 0;
  metas[2].camera_index = 2;

  EXPECT_NE(sources[0].data, nullptr);
  EXPECT_EQ(sources[1].data, nullptr);
  EXPECT_NE(sources[2].data, nullptr);
  EXPECT_EQ(sources[3].data, nullptr);

  EXPECT_EQ(masks[0].data, nullptr);
  EXPECT_NE(masks[2].data, nullptr);

  EXPECT_NE(depth_sources[0].data, nullptr);
  EXPECT_EQ(depth_sources[2].data, nullptr);
  EXPECT_EQ(metas[2].camera_index, 2);
}

TEST(TestStatisticalVariable, SVTest) {
  NamedStatisticalVariable<float> statVar;
  std::vector<float> statVec;

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::normal_distribution<float> nd(0.5f, 2.5f);

  const size_t totalCount = 10000;

  for (size_t i = 0; i < totalCount; i++) {
    const float x = nd(gen);
    statVar(x);
    statVec.push_back(x);
  }

  const size_t nStats = statVec.size();
  assert(nStats == totalCount);
  const float mean = std::accumulate(statVec.cbegin(), statVec.cend(), float(0)) / float(nStats);
  const float var = std::accumulate(statVec.cbegin(), statVec.cend(), float(0),
                                    [=](float a, float b) -> float { return a + (b - mean) * (b - mean); }) /
                    float(nStats);
  std::stable_sort(statVec.begin(), statVec.end());

  EXPECT_EQ(statVar.count(), totalCount);
  EXPECT_EQ(statVar.max(), statVec.back());
  EXPECT_EQ(statVar.min(), statVec.front());
  EXPECT_TRUE(IsApprox(statVar.mean(), mean, epsilon() * 100));
  EXPECT_TRUE(IsApprox(statVar.variance(), var, epsilon() * 100));
}

class TestcuVSLAMExtentionEigenTypes : public testing::Test {};

const float SCALAR_EPSILON = epsilon();
const float SCALAR_EPSILON_2X = epsilon() * 2;
const float SCALAR_EPSILON_4X = epsilon() * 4;
const float SCALAR_EPSILON_8X = epsilon() * 8;
const float SCALAR_EPSILON_12X = epsilon() * 12;
const float SCALAR_EPSILON_400X = epsilon() * 400;
const float SCALAR_EPSILON_600X = epsilon() * 600;

const struct AnglePair {
  float radian;
  float degree;
  AnglePair(const float v) : radian(v), degree(v * float(180 / PI)) {}

} ANGLE_LIST[] = {float(0.2),    float(0.0001),   float(-0.52),    float(1.3332),   float(-1.675),
                  float(3.5677), float(-7.779),   float(PI / 3),   float(-PI / 4),  float(PI / 2),
                  float(-PI),    float(2.2 * PI), float(3.5 * PI), float(-3.6 * PI)};

TEST_F(TestcuVSLAMExtentionEigenTypes, ScalarAndAngleStaticCheck) {
  // static_assert((std::is_same<float, float>::value), "Static check that we use float as float type");
  // static_assert((std::is_same<AngleT::Scalar, float>::value), "Static check that we use float as AngleT type");
  static_assert(sizeof(AngleT) == sizeof(float),
                "Static check that memory size of AngleT type and float type is the same");
  static_assert(AngleUnits::Degree != AngleUnits::Radian, "Static check that AngleUnits::Degree != AngleUnits::Radian");
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleConstructAndGetValue) {
  const float epsilonRad = SCALAR_EPSILON_8X;
  const float epsilonDeg = SCALAR_EPSILON_600X;

  for (const auto& a : ANGLE_LIST) {
    AngleT aRad(a.radian, AngleUnits::Radian);
    AngleT aDeg(a.degree, AngleUnits::Degree);
    EXPECT_EQ(aRad.getValue(), a.radian);
    EXPECT_NEAR(aDeg.getValue(), a.radian, epsilonRad);
    const float degVal = aRad.getValue(AngleUnits::Degree);
    EXPECT_NEAR(degVal, a.degree, epsilonDeg);
    const float degVal2 = aDeg.getValue(AngleUnits::Degree);
    EXPECT_NEAR(degVal2, a.degree, epsilonDeg);
    EXPECT_EQ(aRad.getValue(AngleUnits::Radian), a.radian);
    EXPECT_NEAR(aDeg.getValue(AngleUnits::Radian), a.radian, epsilonRad);
    EXPECT_EQ(float(aRad), a.radian);
    EXPECT_NEAR(float(aDeg), a.radian, epsilonRad);
    AngleT aTest1(aRad);
    AngleT aTest2;
    aTest2 = aDeg;
    EXPECT_EQ(aTest1.getValue(), a.radian);
    EXPECT_NEAR(aTest2.getValue(), a.radian, epsilonRad);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleDefaultScalarOps) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.radian, AngleUnits::Radian);
    const AngleT delta = float(0.1);
    const float factor = float(0.967);

    AngleT aTest2 = aTest + delta;
    EXPECT_EQ(aTest2.getValue(), a.radian + delta.getValue());

    aTest2 = aTest - delta;
    EXPECT_EQ(aTest2.getValue(), a.radian - delta.getValue());

    aTest2 = aTest * factor;
    EXPECT_EQ(aTest2.getValue(), a.radian * factor);

    aTest2 = aTest / factor;
    EXPECT_EQ(aTest2.getValue(), a.radian / factor);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleOperatorAssignAdd) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.degree, AngleUnits::Degree);
    const AngleT delta(a.radian * float(0.7395) - float(0.2835));

    AngleT aTest2 = aTest - delta;
    aTest2 += delta;
    EXPECT_NEAR(aTest2.getValue(), a.radian, SCALAR_EPSILON_8X);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleOperatorAssignSub) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.degree, AngleUnits::Degree);
    const AngleT delta(a.radian * float(0.121) + float(0.1556));

    AngleT aTest2 = aTest + delta;
    aTest2 -= delta;
    EXPECT_NEAR(aTest2.getValue(), a.radian, SCALAR_EPSILON_12X);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleOperatorAssignMul) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.degree, AngleUnits::Degree);
    const float factor = float(1.3556);

    AngleT aTest2 = aTest / factor;
    aTest2 *= factor;
    EXPECT_NEAR(aTest2.getValue(), a.radian, SCALAR_EPSILON_12X);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleOperatorAssignDiv) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.degree, AngleUnits::Degree);
    const float factor = float(0.3556);

    AngleT aTest2 = aTest * factor;
    aTest2 /= factor;
    EXPECT_NEAR(aTest2.getValue(), a.radian, SCALAR_EPSILON_12X);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, AngleNormalize) {
  for (const auto& a : ANGLE_LIST) {
    const AngleT aTest(a.degree, AngleUnits::Degree);
    const AngleT aNormed = aTest.getNormalized();

    if (std::fabs(aTest) <= float(PI)) {
      EXPECT_NEAR(aNormed.getValue(), aTest.getValue(), SCALAR_EPSILON_2X);
    } else {
      EXPECT_LE(std::fabs(aNormed), float(PI));
      const float test = std::fmod(float(aTest) + float(PI * 5), float(PI * 2)) - float(PI);
      EXPECT_NEAR(float(aNormed), test, SCALAR_EPSILON_12X);
    }
  }
}

const Vector3T ROTATION_VECTOR_VALUES(-0.23f, 1.74f, -3.1f);
const Vector3T TRANSLATION_VECTOR_VALUES(1.678f, -3.742f, 5.125f);

const AngleVector3T ANGLE_VECTOR(AngleVector(ROTATION_VECTOR_VALUES, AngleUnits::Radian));
const AngleVector3T ANGLE_VECTOR_NORM(ANGLE_VECTOR.x().getNormalized(), ANGLE_VECTOR.y().getNormalized(),
                                      ANGLE_VECTOR.z().getNormalized());
const Rotation3T ROTATION(ANGLE_VECTOR);

const Translation3T TRANSLATION(TRANSLATION_VECTOR_VALUES);

const Isometry3T ISOMETRIC_TRANSFORMATION = TRANSLATION * ROTATION;

TEST_F(TestcuVSLAMExtentionEigenTypes, TypeTransformDecompositionTranslation) {
  const Vector3T vTrans = ISOMETRIC_TRANSFORMATION.translation();

  EXPECT_EQ(vTrans.x(), TRANSLATION_VECTOR_VALUES.x());
  EXPECT_EQ(vTrans.y(), TRANSLATION_VECTOR_VALUES.y());
  EXPECT_EQ(vTrans.z(), TRANSLATION_VECTOR_VALUES.z());
}

TEST_F(TestcuVSLAMExtentionEigenTypes, TransformDecompositionEulerRotation) {
  const AngleVector3T rot = getEulerRotation(ISOMETRIC_TRANSFORMATION);

  if (std::fabs(rot.x() - ANGLE_VECTOR_NORM.x()) < SCALAR_EPSILON_2X) {
    EXPECT_NEAR(rot.y(), ANGLE_VECTOR_NORM.y(), SCALAR_EPSILON_2X);
    EXPECT_NEAR(rot.z(), ANGLE_VECTOR_NORM.z(), SCALAR_EPSILON_2X);
  } else {
    EXPECT_NEAR(std::fabs(rot.x() - ANGLE_VECTOR_NORM.x()), float(PI), SCALAR_EPSILON_2X);
    EXPECT_NEAR(std::fabs(rot.y() + ANGLE_VECTOR_NORM.y()), float(PI), SCALAR_EPSILON_2X);
    EXPECT_NEAR(std::fabs(rot.z() - ANGLE_VECTOR_NORM.z()), float(PI), SCALAR_EPSILON_2X);
  }
}

TEST_F(TestcuVSLAMExtentionEigenTypes, TransformDecompositionEulerRotationStable) {
  const AngleVector3T nearRot(ANGLE_VECTOR_NORM + AngleVector(Vector3T(0.3f, 0.3f, 0.3f)));
  const AngleVector3T rot = getEulerRotationStable(ISOMETRIC_TRANSFORMATION, nearRot);

  EXPECT_NEAR(rot.x(), ANGLE_VECTOR_NORM.x(), SCALAR_EPSILON_2X);
  EXPECT_NEAR(rot.y(), ANGLE_VECTOR_NORM.y(), SCALAR_EPSILON_2X);
  EXPECT_NEAR(rot.z(), ANGLE_VECTOR_NORM.z(), SCALAR_EPSILON_2X);
}

}  // namespace test
