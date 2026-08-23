
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

#include "math/twist.h"
#include "common/include_gtest.h"
#include "common/rotation_utils.h"
#include "common/vector_2t.h"

namespace test::math {
using namespace cuvslam;

using TwistT = cuvslam::math::VectorTwist<6>;

TEST(TwistTests, RotTwist2RTAndBack) {
  Vector3T rand3(Vector3T::Random().normalized());
  Vector2T rand2(3 * Vector2T::Random());
  Vector6T vec = (Vector6T() << rand3 * rand2(0), Vector3T::Zero()).finished();
  TwistT tw(vec);
  Isometry3T rt = tw.transform();
  TwistT tw2(rt);
  Matrix3T m = common::CalculateRotationFromSVD(rt.matrix());
  Vector3T rvec = Vector3T(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));
  Vector3T twrot = tw.head(3);
  Vector3T tw2rot = tw2.head(3);

  EXPECT_TRUE(rt.translation().isZero());
  EXPECT_TRUE(Vector3T(tw2.tail(3)).isZero());
  EXPECT_NEAR(std::abs(rand3.dot(rvec)), float(rvec.norm()), 5 * epsilon());
  EXPECT_NEAR(tw2rot.norm(), std::abs(rand2(0)), 20 * epsilon());
  EXPECT_LT((tw2rot - twrot).norm(), 20 * epsilon());
}

TEST(TwistTests, RT2FullTwistAndBack) {
  Vector3T rand3(Vector3T::Random() * PI);
  QuaternionT qt(rand3, AngleUnits::Radian);
  // qt.normalize();
  assert(IsApprox(qt.norm(), float(1), 10 * epsilon()));
  Isometry3T t;
  t.linear() = qt.toRotationMatrix();
  t.translation() = Vector3T::Random() * 100;
  t.makeAffine();

  TwistT tw(t);
  Isometry3T t2 = tw.transform();

  EXPECT_TRUE(common::CalculateRotationFromSVD(t2.matrix()).isApprox(common::CalculateRotationFromSVD(t.matrix())));
  EXPECT_TRUE(t2.translation().isApprox(t.translation()));
}

TEST(TwistTests, SimplifiedCamPointDerivative) {
  Vector3T p3dWorld(Vector3T::Random());

  Vector3T rand3(Vector3T::Random() * PI);
  QuaternionT qt(rand3, AngleUnits::Radian);

  Isometry3T t;
  t.linear() = qt.toRotationMatrix();
  t.translation() = Vector3T::Random() * 100;
  t.makeAffine();

  cuvslam::math::TwistDerivativesT::CameraDerivs testCamDeriv, goldCamDeriv;
  cuvslam::math::TwistDerivativesT::PointDerivs testPtDeriv, goldPtDeriv;

  cuvslam::math::TwistDerivativesT::calcCamPointDerivs(p3dWorld, t, common::CalculateRotationFromSVD(t.matrix()),
                                                       testCamDeriv, testPtDeriv);

  cuvslam::math::TwistDerivativesOldT aux;
  aux.calcDerivs(p3dWorld, t, goldCamDeriv, goldPtDeriv);

  EXPECT_TRUE(testCamDeriv.isApprox(goldCamDeriv));
  EXPECT_TRUE(testPtDeriv.isApprox(goldPtDeriv));
}

TEST(TwistTests, SimplifiedPointDerivative) {
  Vector3T p3dWorld(Vector3T::Random());

  Vector3T rand3(Vector3T::Random() * PI);
  QuaternionT qt(rand3, AngleUnits::Radian);

  Isometry3T t;
  t.linear() = qt.toRotationMatrix();
  t.translation() = Vector3T::Random() * 100;
  t.makeAffine();

  cuvslam::math::TwistDerivativesT::PointDerivs testPtDeriv, goldPtDeriv;
  cuvslam::math::TwistDerivativesT::CameraDerivs goldCamDeriv;

  cuvslam::math::TwistDerivativesT::calcPointDerivs(p3dWorld, t, common::CalculateRotationFromSVD(t.matrix()),
                                                    testPtDeriv);

  cuvslam::math::TwistDerivativesOldT aux;
  aux.calcDerivs(p3dWorld, t, goldCamDeriv, goldPtDeriv);

  EXPECT_TRUE(testPtDeriv.isApprox(goldPtDeriv));
}

TEST(TwistTests, SimplifiedCamDerivative) {
  Vector3T p3dWorld(Vector3T::Random());

  Vector3T rand3(Vector3T::Random() * PI);
  QuaternionT qt(rand3, AngleUnits::Radian);

  Isometry3T t;
  t.linear() = qt.toRotationMatrix();
  t.translation() = Vector3T::Random() * 100;
  t.makeAffine();

  cuvslam::math::TwistDerivativesT::CameraDerivs testCamDeriv, goldCamDeriv;

  cuvslam::math::TwistDerivativesT::calcCamDerivs(p3dWorld, t, testCamDeriv);

  cuvslam::math::TwistDerivativesOldT aux;
  aux.calcDerivs(p3dWorld, t, goldCamDeriv);

  EXPECT_TRUE(testCamDeriv.isApprox(goldCamDeriv));
}

}  // namespace test::math
