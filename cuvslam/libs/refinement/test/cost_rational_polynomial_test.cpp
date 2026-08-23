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

#include "refinement/cost_rational_polynomial.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <random>
#include "ceres/ceres.h"

using namespace cuvslam::refinement::rational_polynomial;

class RationalPolynomialCostTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up common test parameters with realistic values
    // Camera intrinsics
    intrinsics[0] = 613.52020263671875;
    intrinsics[1] = 613.0762939453125;
    intrinsics[2] = 555.54913330078125;
    intrinsics[3] = 338.2757568359375;

    // OpenCV camera matrix
    camera_matrix =
        (cv::Mat_<double>(3, 3) << intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);

    // Distortion coefficients (k1,k2,k3,p1,p2,k4,k5,k6)
    // Note: OpenCV uses different order than our implementation
    k1_k2_k3_k4_k5_k6[0] = -0.28340811;  // k1
    k1_k2_k3_k4_k5_k6[1] = 0.07395907;   // k2
    k1_k2_k3_k4_k5_k6[2] = 0.0;          // k3
    k1_k2_k3_k4_k5_k6[3] = 0.0;          // k4
    k1_k2_k3_k4_k5_k6[4] = 0.0;          // k5
    k1_k2_k3_k4_k5_k6[5] = 0.0;          // k6
    p1_p2[0] = 0.00019359;               // p1
    p1_p2[1] = 1.76187114e-05;           // p2

    // OpenCV distortion coefficients (k1,k2,p1,p2,k3,k4,k5,k6)
    dist_coeffs = (cv::Mat_<double>(8, 1) << k1_k2_k3_k4_k5_k6[0],  // k1
                   k1_k2_k3_k4_k5_k6[1],                            // k2
                   p1_p2[0],                                        // p1
                   p1_p2[1],                                        // p2
                   k1_k2_k3_k4_k5_k6[2],                            // k3
                   k1_k2_k3_k4_k5_k6[3],                            // k4
                   k1_k2_k3_k4_k5_k6[4],                            // k5
                   k1_k2_k3_k4_k5_k6[5]                             // k6
    );

    // Camera pose (identity rotation and zero translation)
    for (int i = 0; i < 3; ++i) {
      T_camera_world_angleaxis[i] = 0.0;
      T_camera_world_translation[i] = 0.0;
    }

    // OpenCV rotation vector and translation vector
    rvec = cv::Mat::zeros(3, 1, CV_64F);
    tvec = cv::Mat::zeros(3, 1, CV_64F);

    camera_from_rig = Eigen::Matrix<double, 4, 4>::Identity();
  }

  // Test helper to project a 3D point using our implementation
  bool project3DPoint(const double point[3], double* predicted_u, double* predicted_v) {
    return calculatePredictedObservation<double>(&T_camera_world_angleaxis[0], &T_camera_world_translation[0],
                                                 &point[0], &intrinsics[0], dist_coeffs.ptr<double>(), camera_from_rig,
                                                 predicted_u, predicted_v);
  }

  // Test helper to project a 3D point using OpenCV
  bool projectPointOpenCV(const double point[3], cv::Point2d& projected) {
    std::vector<cv::Point3d> object_points = {cv::Point3d(point[0], point[1], point[2])};
    std::vector<cv::Point2d> image_points;

    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, image_points);
    projected = image_points[0];
    return true;
  }

  std::array<double, 4> intrinsics;
  double k1_k2_k3_k4_k5_k6[6];
  double p1_p2[2];
  double T_camera_world_angleaxis[3];
  double T_camera_world_translation[3];

  Eigen::Matrix<double, 4, 4> camera_from_rig;
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  cv::Mat rvec;
  cv::Mat tvec;
};

TEST_F(RationalPolynomialCostTest, TestAgainstOpenCV) {
  // Test a grid of 3D points and compare with OpenCV's projection
  const int n_points = 5;
  const double z = 2.0;
  const double max_xy = 1.0;

  for (int ix = -n_points; ix <= n_points; ++ix) {
    for (int iy = -n_points; iy <= n_points; ++iy) {
      double x = ix * max_xy / n_points;
      double y = iy * max_xy / n_points;
      double point[3] = {x, y, z};

      // Project using our implementation
      double predicted_u, predicted_v;
      ASSERT_TRUE(project3DPoint(point, &predicted_u, &predicted_v));

      // Project using OpenCV
      cv::Point2d opencv_projected;
      ASSERT_TRUE(projectPointOpenCV(point, opencv_projected));

      // Compare results
      EXPECT_NEAR(predicted_u, opencv_projected.x, 1e-4) << "Point: (" << x << ", " << y << ", " << z << ")";
      EXPECT_NEAR(predicted_v, opencv_projected.y, 1e-4) << "Point: (" << x << ", " << y << ", " << z << ")";
    }
  }
}

TEST_F(RationalPolynomialCostTest, TestRotatedCameraAgainstOpenCV) {
  // Test with various camera rotations
  const std::vector<double> angles = {M_PI / 6, M_PI / 4, M_PI / 3};  // 30, 45, 60 degrees
  const std::vector<int> axes = {0, 1, 2};                            // rotation around x, y, z axes

  double point[3] = {0.5, 0.3, 2.0};

  for (double angle : angles) {
    for (int axis : axes) {
      // Set rotation for our implementation
      std::fill(T_camera_world_angleaxis, T_camera_world_angleaxis + 3, 0.0);
      T_camera_world_angleaxis[axis] = angle;

      // Set rotation for OpenCV
      cv::Mat new_rvec = cv::Mat::zeros(3, 1, CV_64F);
      new_rvec.at<double>(axis) = angle;
      rvec = new_rvec;

      // Project using our implementation
      double predicted_u, predicted_v;
      ASSERT_TRUE(project3DPoint(point, &predicted_u, &predicted_v));

      // Project using OpenCV
      cv::Point2d opencv_projected;
      ASSERT_TRUE(projectPointOpenCV(point, opencv_projected));

      // Compare results
      EXPECT_NEAR(predicted_u, opencv_projected.x, 1e-4) << "Rotation " << angle << " around axis " << axis;
      EXPECT_NEAR(predicted_v, opencv_projected.y, 1e-4) << "Rotation " << angle << " around axis " << axis;
    }
  }
}

TEST_F(RationalPolynomialCostTest, TestDistortionCoefficientsAgainstOpenCV) {
  // Test different combinations of distortion coefficients
  const double point[3] = {0.5, 0.3, 2.0};
  const std::vector<std::vector<double>> test_coeffs = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},   // No distortion
      {-0.2, 0.0, 0.0, 0.0, 0.0, 0.0},  // Only k1
      {0.0, 0.1, 0.0, 0.0, 0.0, 0.0},   // Only k2
      {-0.2, 0.1, 0.0, 0.0, 0.0, 0.0},  // k1 and k2
      {0.0, 0.0, 0.1, 0.0, 0.0, 0.0},   // Only k3
      {-0.2, 0.1, 0.05, 0.0, 0.0, 0.0}  // k1, k2, k3
  };

  const std::vector<std::vector<double>> test_tangential = {
      {0.0, 0.0},     // No tangential
      {0.001, 0.0},   // Only p1
      {0.0, 0.001},   // Only p2
      {0.001, 0.001}  // Both p1 and p2
  };

  for (const auto& radial_coeffs : test_coeffs) {
    for (const auto& tangential_coeffs : test_tangential) {
      // Set coefficients for our implementation
      std::copy(radial_coeffs.begin(), radial_coeffs.end(), k1_k2_k3_k4_k5_k6);
      std::copy(tangential_coeffs.begin(), tangential_coeffs.end(), p1_p2);

      // Set coefficients for OpenCV
      dist_coeffs.at<double>(0) = radial_coeffs[0];      // k1
      dist_coeffs.at<double>(1) = radial_coeffs[1];      // k2
      dist_coeffs.at<double>(2) = tangential_coeffs[0];  // p1
      dist_coeffs.at<double>(3) = tangential_coeffs[1];  // p2
      dist_coeffs.at<double>(4) = radial_coeffs[2];      // k3
      dist_coeffs.at<double>(5) = radial_coeffs[3];      // k4
      dist_coeffs.at<double>(6) = radial_coeffs[4];      // k5
      dist_coeffs.at<double>(7) = radial_coeffs[5];      // k6

      // Project using our implementation
      double predicted_u, predicted_v;
      ASSERT_TRUE(project3DPoint(point, &predicted_u, &predicted_v));

      // Project using OpenCV
      cv::Point2d opencv_projected;
      ASSERT_TRUE(projectPointOpenCV(point, opencv_projected));

      // Compare results
      EXPECT_NEAR(predicted_u, opencv_projected.x, 1e-4)
          << "Failed with radial coeffs: " << radial_coeffs[0] << ", " << radial_coeffs[1] << ", " << radial_coeffs[2]
          << " and tangential coeffs: " << tangential_coeffs[0] << ", " << tangential_coeffs[1];
      EXPECT_NEAR(predicted_v, opencv_projected.y, 1e-4)
          << "Failed with radial coeffs: " << radial_coeffs[0] << ", " << radial_coeffs[1] << ", " << radial_coeffs[2]
          << " and tangential coeffs: " << tangential_coeffs[0] << ", " << tangential_coeffs[1];
    }
  }
}

TEST_F(RationalPolynomialCostTest, TestReprojectionErrorAgainstOpenCV) {
  // Test that reprojection errors match OpenCV's projection
  double point[3] = {0.5, 0.3, 2.0};

  // Get OpenCV's projection as our "observed" point
  cv::Point2d opencv_projected;
  ASSERT_TRUE(projectPointOpenCV(point, opencv_projected));

  // Create cost function with OpenCV's projection as observation
  ReprojectionError cost_function(&camera_from_rig, opencv_projected.x, opencv_projected.y);
  double residuals[2];

  // Compute residuals
  ASSERT_TRUE(cost_function(T_camera_world_angleaxis, T_camera_world_translation, point, &intrinsics[0],
                            dist_coeffs.ptr<double>(), residuals));

  // Residuals should be near zero since we're using OpenCV's projection as observation
  EXPECT_NEAR(residuals[0], 0.0, 1e-4);
  EXPECT_NEAR(residuals[1], 0.0, 1e-4);
}
