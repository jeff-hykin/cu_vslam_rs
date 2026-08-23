
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

#include "pnp/mono_pnp.h"

#include "common/rotation_utils.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "math/robust_cost_function.h"
#include "math/twist.h"

// #define OPTIMIZER_DEBUG
#ifdef OPTIMIZER_DEBUG
#define DEBUG_ONLY(x) x
#include <iostream>
#else  // not a debug build
#define DEBUG_ONLY(x)
#endif

namespace cuvslam::pnp {

// Compute value of the cost function
// for a pose refinement problem (6-DOF or 3-DOF):
// cost = | observation - project(T * p) |^2 -> min
float ComputeCost(const PoseRefinementInput& input, const Isometry3T& camera_from_world) {
  float cost{0};

  // some points might get rejected because they are too close,
  // so we count "good" points
  int32_t count{0};

  for (int32_t i{0}; i < input.n; ++i) {
    Vector3T point = camera_from_world * vec3(input.xyzs + i * 3);

    auto observation = vec2(input.uvs + i * 2);
    float* info_ptr = input.infos + i * 4;
    if (input.defaultInfo != nullptr) {
      info_ptr = (float*)input.defaultInfo->data();
    }
    auto info = mat2(info_ptr);

    // Caller should have removed all the points which
    // are too close to the camera.
    // During optimization points can move closer.
    // We protect optimizer from excessively large entries of the hessian
    // by skipping such points.
    // This is not a perfect solution: it reduces the total cost.
    // We compute average cost, which is not very sensitive to a few missing points.
    if (point.z() <= sqrt_epsilon()) continue;

    ++count;

    auto one_over_z = 1.f / point.z();
    Vector2T xy = point.topRows(2) * one_over_z;

    Vector2T r = observation - xy;
    float r_squared_norm = r.dot(info * r);

    cost += math::ComputeHuberLoss(r_squared_norm, input.huber_delta);
  }

  // This is a pathological situtation and should never happen.
  // However, we have to be robust to such cases.
  if (0 == count) {
    assert(false);
    count = 1;
  }

  auto one_over_count = 1.f / static_cast<float>(count);

  return cost * one_over_count;
}

struct ComputeCostStereo {
  storage::Isometry3<float> right_from_left;

  float operator()(const PoseRefinementInput* input, const Isometry3T& left_camera_from_world) {
    float costR(0.f);

    const float costL = ComputeCost(input[0], left_camera_from_world);
    if (input[1].n) {
      // input[0].n currently can not be 0 as otherwise we'll bail on
      // points[0].size() < 3 test in resectioning but for a future development, so
      const float weight = (input[0].n) ? (float)input[1].n / input[0].n : 1.0f;

      costR = ComputeCost(input[1], right_from_left * left_camera_from_world) * weight;
    }

    return costL + costR;
  }
};

// Compute value of the cost function, its gradient and hessian at update = 0
// for 6-DOF pose refinement problem:
// cost = | observation - project(exp_se3(update) * T * p) |^2 -> min
float ComputeCostSE3(const PoseRefinementInput& input, const Isometry3T& camera_from_world, Matrix6T& hessian,
                     Vector6T& negative_gradient) {
  float cost{0};

  hessian.setZero();
  negative_gradient.setZero();

  // some points might get rejected because they are too close,
  // so we count "good" points
  int32_t count{0};

  for (int32_t i{0}; i < input.n; ++i) {
    Vector3T p = camera_from_world * vec3(input.xyzs + i * 3);

    // Caller should have removed all the points which
    // are too close to the camera.
    // During optimization points can move closer.
    // We protect optimizer from excessively large entries of the hessian
    // by skipping such points.
    // This is not a perfect solution: it reduces the total cost.
    // We compute average cost, which is not very sensitive to a few missing points.
    if (p.z() <= sqrt_epsilon()) continue;

    ++count;

    auto observation = vec2(input.uvs + i * 2);
    float* info_ptr = input.infos + i * 4;
    if (input.defaultInfo != nullptr) {
      info_ptr = (float*)input.defaultInfo->data();
    }
    auto info = mat2(info_ptr);

    auto one_over_z = 1.f / p.z();
    Vector2T xy = p.topRows(2) * one_over_z;

    Vector2T r = observation - xy;
    float r_squared_norm = r.dot(info * r);

    cost += math::ComputeHuberLoss(r_squared_norm, input.huber_delta);

    MatrixMN<float, 2, 3> dproject;
    dproject << one_over_z, 0.f, -xy(0) * one_over_z, 0.f, one_over_z, -xy(1) * one_over_z;

    // jacobian for the rotational component,
    // translational block is identity
    MatrixMN<float, 3, 3> dtransform;
    dtransform.block<3, 3>(0, 0) << 0.f, p.z(), -p.y(), -p.z(), 0.f, p.x(), p.y(), -p.x(), 0.f;

    // d  r   |
    // -------|
    // d omega| omega = 0
    MatrixMN<float, 2, 6> j;
    j.block<2, 3>(0, 0) = dproject * dtransform;
    j.block<2, 3>(0, 3) = dproject;

    // weight for robust least squares approximation
    auto w = math::ComputeDHuberLoss(r_squared_norm, input.huber_delta);
    MatrixMN<float, 6, 2> jsw = j.transpose() * (info * w);

    hessian += jsw * j;

    // There should be an extra '-' in the formulas above,
    // but we "defer" it for simplicity:
    // Gauss-Newton step is defined as a solution to
    // J^T J p = -J^T e
    // Our residual has the form e = z - f(x),
    // so J must be d/dx (-f(x)), but we compute d/dx f(x) instead.
    // In other words, our J is the negative of the text book J,
    // effectively making our gradient a negative gradient.
    negative_gradient += jsw * r;
  }

  auto one_over_count = 1.f / static_cast<float>(count);

  negative_gradient *= one_over_count;
  hessian *= one_over_count;

  return cost * one_over_count;
}

struct ComputeCostSE3Stereo {
  storage::Isometry3<float> right_from_left;

  float operator()(const PoseRefinementInput* input, const Isometry3T& left_camera_from_world, Matrix6T& hessian,
                   Vector6T& negative_gradient) {
    using Mat = MatrixMN<float, 6, 6>;
    using Vec = MatrixMN<float, 6, 1>;
    float cost = ComputeCostSE3(input[0], left_camera_from_world, hessian, negative_gradient);

    if (input[1].n) {
      // J^TJ approximation for Right camera
      Mat hessianR;
      Vec negative_gradientR;
      Eigen::Matrix<float, 6, 6> Adj;

      // input[0].n currently can not be 0 as otherwise we'll bail on
      // points[0].size() < 3 test in resectioning but for a future development, so
      const float weight = (input[0].n) ? (float)input[1].n / input[0].n : 1.0f;

      cost += ComputeCostSE3(input[1], right_from_left * left_camera_from_world, hessianR, negative_gradientR) * weight;

      math::Adjoint(Adj, right_from_left);
      auto AdjT = Adj.transpose();
      hessian = hessian + AdjT * hessianR * Adj * weight;
      negative_gradient = negative_gradient + AdjT * negative_gradientR * weight;
    }

    return cost;
  }
};

// Compute value of the cost function
// for a 3-DOF pose refinement problem:
// cost = | observation - project(exp_so3(update) * T * p) |^2 -> min
float ComputeCostSO3(const PoseRefinementInput* inputPtr, const Isometry3T& camera_from_world, Matrix3T& hessian,
                     Vector3T& negative_gradient) {
  float cost{0};

  hessian.setZero();
  negative_gradient.setZero();

  const PoseRefinementInput& input = inputPtr[0];

  // some points might get rejected because they are too close,
  // so we count "good" points
  int32_t count{0};

  for (int32_t i{0}; i < input.n; ++i) {
    Vector3T p = camera_from_world * vec3(input.xyzs + i * 3);

    // Caller should have removed all the points which
    // are too close to the camera.
    // During optimization points can move closer.
    // We protect optimizer from excessively large entries of the hessian
    // by skipping such points.
    // This is not a perfect solution: it reduces the total cost.
    // We compute average cost, which is not very sensitive to a few missing points.
    if (p.z() <= sqrt_epsilon()) continue;

    ++count;

    auto observation = vec2(input.uvs + i * 2);
    float* info_ptr = input.infos + i * 4;
    if (input.defaultInfo != nullptr) {
      info_ptr = (float*)input.defaultInfo->data();
    }
    auto info = mat2(info_ptr);

    auto one_over_z = 1.f / p.z();
    Vector2T xy = p.topRows(2) * one_over_z;

    Vector2T r = observation - xy;
    float r_squared_norm = r.dot(info * r);

    cost += math::ComputeHuberLoss(r_squared_norm, input.huber_delta);

    MatrixMN<float, 2, 3> dproject;
    dproject << one_over_z, 0.f, -xy(0) * one_over_z, 0.f, one_over_z, -xy(1) * one_over_z;

    // jacobian for the rotational component,
    // translational block is identity
    MatrixMN<float, 3, 3> dtransform;
    dtransform.block<3, 3>(0, 0) << 0.f, p.z(), -p.y(), -p.z(), 0.f, p.x(), p.y(), -p.x(), 0.f;

    // d  r   |
    // -------|
    // d omega| omega = 0
    MatrixMN<float, 2, 3> j;
    j.block<2, 3>(0, 0) = dproject * dtransform;

    // weight for robust least squares approximation
    auto w = math::ComputeDHuberLoss(r_squared_norm, input.huber_delta);
    MatrixMN<float, 3, 2> jsw = j.transpose() * (info * w);

    hessian += jsw * j;

    // There should be an extra '-' in the formulas above,
    // but we "defer" it for simplicity.
    // See ComputeCostSE3 for more details.
    negative_gradient += jsw * r;
  }

  auto one_over_count = 1.f / static_cast<float>(count);

  negative_gradient *= one_over_count;
  hessian *= one_over_count;

  return cost * one_over_count;
}

// Minimizes reprojection error by iteratively minimizing:
// sum loss(|observation - project(guess * point)|^2)
//
// with guess computed by local_update(guess, step, current_estimate)
template <int32_t Dim, class CostFunction, class LocalUpdate>
bool MinimizeReprojectionError(Isometry3T& left_camera_from_world, MatrixMN<float, Dim, Dim>& precision,
                               const PoseRefinementInput* input, CostFunction cost_function, LocalUpdate local_update,
                               const Isometry3T& right_from_left_camera_transform) {
  using Mat = MatrixMN<float, Dim, Dim>;
  using Vec = MatrixMN<float, Dim, 1>;

  // J^TJ approximation
  Mat hessian;

  Vec negative_gradient;

  auto initial_cost = cost_function(input, left_camera_from_world, hessian, negative_gradient);
  if (initial_cost < 5e-3f) {
    // Nothing to minimize. We have a "pendulum" effect on still frames otherwise.
    precision = Mat::Identity();
    return true;
  }
  auto current_cost = initial_cost;

  // Diagonal matrix `scaling` controls the shape of the trust region.
  Vec scaling = hessian.diagonal();

  DEBUG_ONLY(std::cout << "scaling: " << scaling.transpose() << "\n");

  float lambda = 1.f;

  int32_t num_iterations = 0;

  ComputeCostStereo compCost;
  compCost.right_from_left = right_from_left_camera_transform;

  const int max_iterations = 13;
  do {
    ++num_iterations;

    Mat augmented_system = hessian + (lambda * scaling).asDiagonal().toDenseMatrix();

    // TODO: (msmirnov) try modified Cholesky
    auto decomposition = augmented_system.ldlt();
    if (!decomposition.isPositive()) {
      // We don't expect hessian to have negative eigen values
      // because our J^TJ approximation is always positive semi-definite.
      // Here we use Gershgorin circle theorem to modify lambda to ensure
      // positive definiteness of the hessian.
      lambda = (hessian - hessian.diagonal().asDiagonal().toDenseMatrix()).colwise().sum().maxCoeff();

      DEBUG_ONLY(printf("Indefinite JTJ + DTD, resetting lambda\n"));
      continue;
    }

    Vec step = decomposition.solve(negative_gradient);

    Isometry3T guess;
    local_update(guess, step, left_camera_from_world);

    auto cost = compCost(input, guess);
    auto predicted_relative_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(scaling.asDiagonal() * step) / current_cost;

    auto step_norm = step.template lpNorm<1>();
    if ((predicted_relative_reduction < sqrt_epsilon()) && (step_norm < sqrt_epsilon())) {
      DEBUG_ONLY(printf("converged\n"));
      current_cost = cost;
      break;
    }

    // This value tells us how well the nonlinear problem is approximated
    // with the linear one.
    // Values below zero indicate increase in cost,
    // values above 0.75 indicate that the linear model is a good approximation.
    //
    // For more details see:
    // More, J.J. 1977. "Levenberg--Marquardt algorithm: implementation and theory". United States. doi:.
    // https://www.osti.gov/servlets/purl/7256021.
    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    auto step_residual = (augmented_system * step - negative_gradient).template lpNorm<1>();
    (void)step_residual;

    DEBUG_ONLY(printf("%3d    % 12.8f    % 12.8f    % 12.8f    % 12.8f    % 12.8f    % 12.8f    % 12.8f\n",
                      num_iterations, current_cost - cost, cost, negative_gradient.template lpNorm<1>(),
                      step.template lpNorm<1>(), lambda, rho, step_residual););

    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step
      left_camera_from_world = guess;

      // our model is good
      if (rho > 0.75f) {
        // limit lambda from below to stabilize the process
        auto new_lambda = lambda / 5.f;
        if (new_lambda > 0.f) {
          lambda = new_lambda;
        }
      }

      current_cost = cost;

      // update hessian and negative_gradient
      cost_function(input, left_camera_from_world, hessian, negative_gradient);

      scaling = scaling.cwiseMax(hessian.diagonal());
    } else {
      lambda *= 2.f;
    }
  } while (num_iterations < max_iterations);

  precision = hessian;

  const float max_success_cost = 0.6f;
  return current_cost < max_success_cost;
}

void LeftUpdateSE3(Isometry3T& ab, const Vector6T& a, const Isometry3T& b) {
  Isometry3T update;
  math::Exp(update, a);
  ab = update * b;
  // TODO: (msmirnov) use modified Gram-Schmidt to save on SVD
  // orthogonalize: Eigen does SVD to compute closest orthonormal matrix
  ab.matrix().block<3, 3>(0, 0) = common::CalculateRotationFromSVD(ab.matrix());
}

void LeftUpdateSO3(Isometry3T& ab, const Vector3T& a, const Isometry3T& b) {
  Isometry3T update;
  math::Exp(update, a);
  ab = update * b;
  // TODO: (msmirnov) use modified Gram-Schmidt to save on SVD
  // orthogonalize: Eigen does SVD to compute closest orthonormal matrix
  ab.matrix().block<3, 3>(0, 0) = common::CalculateRotationFromSVD(ab.matrix());
}

}  // namespace cuvslam::pnp

namespace cuvslam::pnp {

bool RefinePose(Isometry3T& left_camera_from_world, Matrix6T& precision, const PoseRefinementInput* input,
                const PosePrior&, const Isometry3T& right_from_left_camera_transform) {
  if (input[0].n < 3) {
    DEBUG_ONLY(printf("Not enough points\n"));
    return false;
  }

  ComputeCostSE3Stereo costFunction;
  costFunction.right_from_left = right_from_left_camera_transform;
  // Explicit template parameters to help VC++,
  // otherwise it crashes.

  // we will need to project points into camera
  Isometry3T left_camera_from_world_copy = left_camera_from_world;

  if (!MinimizeReprojectionError<6>(left_camera_from_world_copy, precision, input, costFunction, LeftUpdateSE3,
                                    right_from_left_camera_transform)) {
    return false;
  }
  left_camera_from_world = left_camera_from_world_copy;

  return true;
}

bool RefineRotation(Isometry3T& left_camera_from_world, Matrix3T& precision, const PoseRefinementInput* input,
                    const RotationPrior&, const Isometry3T& right_from_left_camera_transform) {
  if (input[0].n < 3) {
    DEBUG_ONLY(printf("Not enough points\n"));
    return false;
  }

  Isometry3T left_camera_from_world_copy = left_camera_from_world;
  if (!MinimizeReprojectionError<3>(left_camera_from_world_copy, precision, input, ComputeCostSO3, LeftUpdateSO3,
                                    right_from_left_camera_transform)) {
    return false;
  }
  left_camera_from_world = left_camera_from_world_copy;

  return true;
}

}  // namespace cuvslam::pnp
