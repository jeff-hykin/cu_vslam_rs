
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

#include "camera/camera.h"
#include "common/include_gtest.h"
#include "common/unaligned_types.h"

namespace Test {
using namespace cuvslam;

TEST(TestCamera, TestNormalizeFishEye) {
  // tesseract gt_trial3 camera
  const Vector2T resolution(1280, 800);
  const Vector2T focal(613.52020263671875f, 613.0762939453125f);
  const Vector2T principal(555.54913330078125f, 338.2757568359375f);
  const float k1 = -0.02501552551984787f;
  const float k2 = -0.0008984454907476902f;
  const float k3 = 0.f;
  const float k4 = 0.f;
  const float max_normalized_uv_radius2 = 1.5f;
  const float max_xy_radius = 5.f;

  camera::FisheyeCameraModel cam(resolution, focal, principal, k1, k2, k3, k4, max_normalized_uv_radius2,
                                 max_xy_radius);

  const int n_tests = 100000;
  for (int i = 0; i < n_tests; ++i) {
    const Vector2T xy(Vector2T::Random() / sqrt(2.f) * max_xy_radius);

    Vector2T uv, xy_renormalized;
    ASSERT_TRUE(cam.denormalizePoint(xy, uv));
    ASSERT_TRUE(cam.normalizePoint(uv, xy_renormalized));
    ASSERT_TRUE(xy_renormalized.isApprox(xy, 0.0001f))
        << "xy_renormalized: " << xy_renormalized.transpose() << " xy: " << xy.transpose();
  }
}

TEST(TestCamera, TestNormalizeRadial) {
  const Vector2T resolution(1280, 800);
  const Vector2T focal(613.52020263671875f, 613.0762939453125f);
  const Vector2T principal(555.54913330078125f, 338.2757568359375f);
  const float k1 = -0.28340811f;
  const float k2 = 0.07395907f;
  const float k3 = 0.f;
  const float p1 = 0.00019359f;
  const float p2 = 1.76187114e-05f;
  const float max_normalized_uv_radius2 = 1.5f;
  const float max_xy_radius = 1.f;

  camera::Brown5KCameraModel cam(resolution, focal, principal, k1, k2, k3, p1, p2, max_normalized_uv_radius2,
                                 max_xy_radius);

  const int n_tests = 100000;
  for (int i = 0; i < n_tests; ++i) {
    const Vector2T xy(Vector2T::Random() / sqrt(2.f) * max_xy_radius);

    Vector2T uv, xy_renormalized;
    ASSERT_TRUE(cam.denormalizePoint(xy, uv));
    ASSERT_TRUE(cam.normalizePoint(uv, xy_renormalized));
    ASSERT_TRUE(xy_renormalized.isApprox(xy, 0.001f))
        << "xy_renormalized: " << xy_renormalized.transpose() << " xy: " << xy.transpose();
  }
}

TEST(TestCamera, TestNormalizePolynomial) {
  const Vector2T resolution(1920, 1200);
  const Vector2T focal(959.4535522460938, 957.6038818359375);
  const Vector2T principal(956.630126953125, 594.9000854492188);
  const float k1 = 4.43821640e+01;
  const float k2 = 3.06905212e+01;
  const float k3 = 1.90606833e+00;
  const float k4 = 4.47033691e+01;
  const float k5 = 4.71669884e+01;
  const float k6 = 8.89805698e+00;
  const float p1 = -1.83029522e-04;
  const float p2 = -2.14764557e-04;
  const float max_normalized_uv_radius2 = 1.5f;
  const float max_xy_radius = 1.f;

  camera::PolynomialCameraModel cam(resolution, focal, principal, k1, k2, k3, k4, k5, k6, p1, p2,
                                    max_normalized_uv_radius2, max_xy_radius);

  const int n_tests = 100000;
  for (int i = 0; i < n_tests; ++i) {
    const Vector2T xy(Vector2T::Random() / sqrt(2.f) * max_xy_radius);

    Vector2T uv, xy_renormalized;
    ASSERT_TRUE(cam.denormalizePoint(xy, uv));
    ASSERT_TRUE(cam.normalizePoint(uv, xy_renormalized));
    ASSERT_TRUE(xy_renormalized.isApprox(xy, 0.001f))
        << "xy_renormalized: " << xy_renormalized.transpose() << " xy: " << xy.transpose();
  }
}

}  // namespace Test
