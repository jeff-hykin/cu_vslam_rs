
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
#include <vector>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "math/pgo.h"
#include "math/twist.h"
namespace test::math {
using namespace cuvslam;
using namespace cuvslam::math;

namespace {

void random_shift(std::mt19937& gen, Isometry3T& pose, const Matrix6T& noise_cov) {
  std::normal_distribution<float> d(0, 1);

  Vector6T x;  // 0 mean, I covariance
  x << d(gen), d(gen), d(gen), d(gen), d(gen), d(gen);

  x = noise_cov.llt().matrixL() * x;

  Isometry3T pose_shift;
  Exp(pose_shift, x);

  pose = pose * pose_shift;
}

Isometry3T random_pose(std::mt19937& gen, const Isometry3T& mean, const Matrix6T& cov) {
  Isometry3T out = mean;
  random_shift(gen, out, cov);
  return out;
}
}  // namespace

TEST(PGO, PGOTest) {
  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());

  const size_t num_poses = 300;
  const size_t num_lcs = 20;
  const float dt = 1.f / 20.f;
  const Matrix6T cov = Matrix6T::Identity() * 1e-1;
  const Matrix6T cov_shift = Matrix6T::Identity() * 1e-2;
  const float velocity = 2;

  std::vector<Isometry3T> original_poses;
  std::vector<Isometry3T> noise_poses;
  float t = 0;
  for (size_t i = 0; i < num_poses; i++) {
    float x = velocity * t;
    t += dt;

    Isometry3T mean = Isometry3T::Identity();
    mean.translation() << x, 0, 0;

    Isometry3T pose = random_pose(gen, mean, cov);
    original_poses.push_back(pose);
    random_shift(gen, pose, cov_shift);
    noise_poses.push_back(pose);
  }

  std::vector<PGOInput::PoseDelta> deltas;
  for (size_t i = 0; i < num_poses - 1; i++) {
    const Isometry3T w_from_p1 = original_poses[i];
    const Isometry3T w_from_p2 = original_poses[i + 1];

    PGOInput::PoseDelta delta = {static_cast<int>(i), static_cast<int>(i + 1), w_from_p1.inverse() * w_from_p2,
                                 Matrix6T::Identity()};
    deltas.push_back(delta);
  }

  for (size_t i = 0; i < num_lcs; i++) {
    std::uniform_int_distribution<int> distr_int(0, original_poses.size() - 1);

    int id1 = distr_int(gen);
    int id2 = id1;
    while (id2 == id1) {
      id2 = distr_int(gen);
    }

    const Isometry3T w_from_p1 = original_poses[id1];
    const Isometry3T w_from_p2 = original_poses[id2];

    PGOInput::PoseDelta delta = {id1, id2, w_from_p1.inverse() * w_from_p2, Matrix6T::Identity()};
    deltas.push_back(delta);
  }

  PGOInput inputs;
  inputs.constrained_pose_ids = {0};
  inputs.poses = noise_poses;
  inputs.deltas = deltas;

  ASSERT_TRUE(PGO{}.run(inputs));
}

namespace {
Vector4T generate_plane(std::mt19937& gen) {
  std::uniform_real_distribution<float> d(-5, 5);

  Vector3T plane_norm = {d(gen), d(gen), d(gen)};
  plane_norm = plane_norm / plane_norm.norm();

  Vector4T plane;
  plane.segment<3>(0) = plane_norm;
  plane[3] = d(gen);
  return plane;
}

Vector3T velocity_on_the_plane(const Vector3T& plane_norm) {
  Vector3T vel = plane_norm;

  if (std::abs(vel.z()) < std::numeric_limits<float>::epsilon()) {
    vel.z() = 0;
    return vel;
  }

  vel.z() = -(vel.x() * vel.x() + vel.y() * vel.y()) / vel.z();
  return vel;
}
}  // namespace

TEST(PGO, DISABLED_PGOTestPlanar) {
  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());

  const size_t num_poses = 300;
  const size_t num_lcs = 20;
  const float dt = 1.f / 20.f;
  const Matrix6T cov_shift = Matrix6T::Identity() * 1e-3;

  for (int k = 0; k < 5; k++) {
    Vector4T plane = generate_plane(gen);
    if (k == 0) {
      plane = {0, 1, 0, 0};
    }

    const Vector3T velocity = velocity_on_the_plane(plane.segment<3>(0));

    std::vector<Isometry3T> original_poses;
    std::vector<Isometry3T> noise_poses;
    float t = 0;
    for (size_t i = 0; i < num_poses; i++) {
      Vector3T x = velocity * t;
      t += dt;

      Isometry3T mean = Isometry3T::Identity();
      mean.translation() = x;

      Isometry3T pose = mean;
      original_poses.push_back(pose);
      if (i != 0) {
        random_shift(gen, pose, cov_shift);
      }
      noise_poses.push_back(pose);
    }

    std::vector<PGOInput::PoseDelta> deltas;
    for (size_t i = 0; i < num_poses - 1; i++) {
      const Isometry3T w_from_p1 = original_poses[i];
      const Isometry3T w_from_p2 = original_poses[i + 1];

      PGOInput::PoseDelta delta = {static_cast<int>(i), static_cast<int>(i + 1), w_from_p1.inverse() * w_from_p2,
                                   Matrix6T::Identity()};
      deltas.push_back(delta);
    }

    for (size_t i = 0; i < num_lcs; i++) {
      std::uniform_int_distribution<int> distr_int(0, original_poses.size() - 1);

      int id1 = distr_int(gen);
      int id2 = id1;
      while (id2 == id1) {
        id2 = distr_int(gen);
      }

      const Isometry3T w_from_p1 = original_poses[id1];
      const Isometry3T w_from_p2 = original_poses[id2];

      PGOInput::PoseDelta delta = {id1, id2, w_from_p1.inverse() * w_from_p2, Matrix6T::Identity()};
      deltas.push_back(delta);
    }

    PGOInput inputs;
    inputs.constrained_pose_ids = {0};
    inputs.poses = noise_poses;
    inputs.deltas = deltas;
    inputs.use_planar_constraint = true;

    inputs.plane_normal = plane;
    inputs.planar_weight = 10;

    ASSERT_TRUE(PGO{}.run(inputs));

    float thresh = 1e-1;

    Eigen::Matrix<float, 1, 3> n_T = inputs.plane_normal.segment<3>(0).transpose();

    for (size_t i = 0; i < inputs.poses.size(); i++) {
      if (inputs.constrained_pose_ids.find(i) != inputs.constrained_pose_ids.end()) {
        continue;
      }

      const Vector3T& T = inputs.poses[i].translation();

      float e = n_T * T + plane[3];

      if (std::abs(e) >= thresh) {
        std::cout << "std::abs(e) = " << std::abs(e) << std::endl;
      }
      ASSERT_TRUE(std::abs(e) < thresh);
    }
  }
}

}  // namespace test::math
