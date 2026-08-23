
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

#include <random>

#include "Eigen/Core"

#include "common/include_gtest.h"
#include "math/transform_between_pointclouds.h"
namespace test::slam {
using namespace cuvslam;
using Eigen::MatrixXf;
using Eigen::VectorXf;

TEST(SlamTest, rigid_transform_3d) {
  std::vector<Vector3T> A(128);

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_real_distribution<> d_translate(-1, 1);

  for (size_t i = 0; i < A.size(); i++) {
    Vector3T t(d_translate(gen), d_translate(gen), d_translate(gen));
    A[i] = t;
  }
  Vector3T rand3(Vector3T::Random() * PI);
  QuaternionT qt(rand3, AngleUnits::Radian);

  Isometry3T t;
  t.linear() = qt.toRotationMatrix();
  t.translation() = Vector3T::Random() * 1;
  t.makeAffine();

  std::vector<Vector3T> B(A.size());
  std::normal_distribution<> nd_translate(0, 0.1);
  for (size_t i = 0; i < A.size(); i++) {
    B[i] = t * A[i];
    B[i] += Vector3T(nd_translate(gen), nd_translate(gen), nd_translate(gen));
  }

  Isometry3T r;
  math::TransformBetweenPointclouds(A, B, r);

  std::cout << "r:\n" << r.matrix() << std::endl;
  std::cout << "t:\n" << t.matrix() << std::endl;

  Isometry3T E = r * t.inverse();
  std::cout << "E:\n" << E.matrix() << std::endl;

  /*/
  Isometry3T E = r * t.inverse();
  std::cout << "E:\n" << E.matrix() << std::endl;

  Vector3T A_mean = std::accumulate(A.begin(), A.end(), Vector3T(0, 0, 0)) / A.size();
  Vector3T B_mean = std::accumulate(B.begin(), B.end(), Vector3T(0, 0, 0)) / B.size();
  std::cout << "mean(A)" << A_mean << "\n";
  std::cout << "mean(B)" << B_mean << "\n";

  MatrixXf Am(3, A.size());
  MatrixXf Bm(3, B.size());
  for (size_t i = 0; i < A.size(); i++) {
      Vector3T a = A[i] - A_mean;
      for (int x = 0; x < 3; x++) {
          Am(x, i) = a(x);
      }
      Vector3T b = B[i] - B_mean;
      for (int x = 0; x < 3; x++) {
          Bm(x, i) = b(x);
      }
  }
//    std::cout << "Am = " << Am << "\n\n";
//    std::cout << "Bm = " << Bm << "\n\n";

  MatrixXf m = Am * Bm.transpose();
  std::cout << "Am * Bm.transpose(): " << "\n";
  std::cout << m << "\n";

  Eigen::Matrix3f AB = m;

  {
      Eigen::JacobiSVD<MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
      std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
      std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() <<
std::endl; std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV()
<< std::endl; Eigen::Vector3f rhs(1, 0, 0); std::cout << "Now consider this rhs vector:" << std::endl << rhs <<
std::endl; std::cout << "A least-squares solution of m*x = rhs is:" << std::endl << svd.solve(rhs) << std::endl;

      std::cout << "Rotation:\n";
      auto R = svd.matrixV() * svd.matrixU().transpose();
      std::cout << "R:\n" << R << std::endl;
      std::cout << "t:\n" << t.matrix() << std::endl;

      Isometry3T r = Isometry3T::Identity();
      r.linear() = R;
      r.translation() = - (R * A_mean) + B_mean;
      r.makeAffine();
      std::cout << "r:\n" << r.matrix() << std::endl;

      Isometry3T E = r * t.inverse();
      std::cout << "E:\n" << E.matrix() << std::endl;
  }
  /*/

  printf("d");
}

}  // namespace test::slam
