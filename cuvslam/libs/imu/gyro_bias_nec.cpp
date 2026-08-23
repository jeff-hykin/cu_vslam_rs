
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

#include "imu/gyro_bias_nec.h"

#include <cmath>
#include <unordered_map>

#include "common/log.h"
#include "math/twist.h"

namespace cuvslam::sba_imu {

namespace {

using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;

// Pre-computed summation matrices for one keyframe pair (Stereo-NEC style).
struct NECSummation {
  Matrix3d xxF = Matrix3d::Zero();
  Matrix3d yyF = Matrix3d::Zero();
  Matrix3d zzF = Matrix3d::Zero();
  Matrix3d xyF = Matrix3d::Zero();
  Matrix3d yzF = Matrix3d::Zero();
  Matrix3d zxF = Matrix3d::Zero();
};

// All data needed for one keyframe pair's NEC constraint.
struct NECEdge {
  NECSummation summation;
  Matrix3d JRg;      // Jacobian of preintegrated rotation w.r.t. gyro bias
  Matrix3d delta_R;  // Preintegrated rotation at original bias
  Matrix3d Rbc;      // Rotation from camera to body/IMU frame
};

// Compute summation matrices from matched bearing vector pairs.
// Ported from Stereo-NEC's Optimizer::summation().
NECSummation ComputeNecSummation(const std::vector<Vector3d>& bearing_i, const std::vector<Vector3d>& bearing_j) {
  NECSummation s;
  for (size_t k = 0; k < bearing_i.size(); k++) {
    const Vector3d& f1 = bearing_i[k];
    const Vector3d& f2 = bearing_j[k];
    Matrix3d F = f2 * f2.transpose();
    s.xxF += f1[0] * f1[0] * F;
    s.yyF += f1[1] * f1[1] * F;
    s.zzF += f1[2] * f1[2] * F;
    s.xyF += f1[0] * f1[1] * F;
    s.yzF += f1[1] * f1[2] * F;
    s.zxF += f1[2] * f1[0] * F;
  }
  return s;
}

// Compute NEC residual (smallest eigenvalue of constraint matrix M) for a single edge.
// Ported from Stereo-NEC's GyroScopeBiasEdgeFactor.
double ComputeResidual(const Vector3d& bg, const NECEdge& edge) {
  // Bias-corrected rotation: Rij_update = delta_R * Exp(JRg * bg)
  Matrix3T exp_result;
  math::Exp(exp_result, (edge.JRg * bg).cast<float>());
  Matrix3d Rij_update = edge.delta_R * exp_result.cast<double>();

  // Transform to camera frame
  Matrix3d R = edge.Rbc.transpose() * Rij_update * edge.Rbc;

  const NECSummation& s = edge.summation;

  // Build 3x3 symmetric matrix M
  Matrix3d M;

  // M(0,0)
  M(0, 0) = (R.row(2) * s.yyF * R.row(2).transpose())(0);
  M(0, 0) += -2.0 * (R.row(2) * s.yzF * R.row(1).transpose())(0);
  M(0, 0) += (R.row(1) * s.zzF * R.row(1).transpose())(0);

  // M(0,1)
  M(0, 1) = (R.row(2) * s.yzF * R.row(0).transpose())(0);
  M(0, 1) += -(R.row(2) * s.xyF * R.row(2).transpose())(0);
  M(0, 1) += -(R.row(1) * s.zzF * R.row(0).transpose())(0);
  M(0, 1) += (R.row(1) * s.zxF * R.row(2).transpose())(0);

  // M(0,2)
  M(0, 2) = (R.row(2) * s.xyF * R.row(1).transpose())(0);
  M(0, 2) += -(R.row(2) * s.yyF * R.row(0).transpose())(0);
  M(0, 2) += -(R.row(1) * s.zxF * R.row(1).transpose())(0);
  M(0, 2) += (R.row(1) * s.yzF * R.row(0).transpose())(0);

  // M(1,1)
  M(1, 1) = (R.row(0) * s.zzF * R.row(0).transpose())(0);
  M(1, 1) += -2.0 * (R.row(0) * s.zxF * R.row(2).transpose())(0);
  M(1, 1) += (R.row(2) * s.xxF * R.row(2).transpose())(0);

  // M(1,2)
  M(1, 2) = (R.row(0) * s.zxF * R.row(1).transpose())(0);
  M(1, 2) += -(R.row(0) * s.yzF * R.row(0).transpose())(0);
  M(1, 2) += -(R.row(2) * s.xxF * R.row(1).transpose())(0);
  M(1, 2) += (R.row(2) * s.xyF * R.row(0).transpose())(0);

  // M(2,2)
  M(2, 2) = (R.row(1) * s.xxF * R.row(1).transpose())(0);
  M(2, 2) += -2.0 * (R.row(0) * s.xyF * R.row(1).transpose())(0);
  M(2, 2) += (R.row(0) * s.yyF * R.row(0).transpose())(0);

  // Symmetric entries
  M(1, 0) = M(0, 1);
  M(2, 0) = M(0, 2);
  M(2, 1) = M(1, 2);

  // Smallest eigenvalue via closed-form cubic solution (Cardano's formula).
  // double precision is required: (b^2 - 3c)^3 and the acos(ratio) clamp are
  // prone to catastrophic cancellation near repeated roots in float.
  double b = -M(0, 0) - M(1, 1) - M(2, 2);
  double c = -std::pow(M(0, 2), 2) - std::pow(M(1, 2), 2) - std::pow(M(0, 1), 2) + M(0, 0) * M(1, 1) +
             M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
  double d = M(1, 1) * std::pow(M(0, 2), 2) + M(0, 0) * std::pow(M(1, 2), 2) + M(2, 2) * std::pow(M(0, 1), 2) -
             M(0, 0) * M(1, 1) * M(2, 2) - 2.0 * M(0, 1) * M(1, 2) * M(0, 2);

  double ss = 2.0 * std::pow(b, 3) - 9.0 * b * c + 27.0 * d;
  double t = 4.0 * std::pow(std::pow(b, 2) - 3.0 * c, 3);

  if (t <= 0.0) return 0.0;

  double ratio = ss / std::sqrt(t);
  ratio = std::max(-1.0, std::min(1.0, ratio));

  double alpha = std::acos(ratio);
  double beta = alpha / 3.0;
  double y = std::cos(beta);

  double r = 0.5 * std::sqrt(t);
  double w = std::cbrt(r);

  double k = w * y;
  return (-b - 2.0 * k) / 3.0;
}

constexpr int kMinCorrespondences = 15;
constexpr float kMaxPreintegrationTime = 1.0f;
constexpr int kMaxIterations = 50;
constexpr double kConvergenceThreshold = 1e-8;
constexpr double kNumericalDiffStep = 1e-6;

// Compute sum of squared residuals for all edges
double ComputeTotalCost(const Vector3d& bg, const std::vector<NECEdge>& edges) {
  double cost = 0.0;
  for (const auto& edge : edges) {
    double r = ComputeResidual(bg, edge);
    cost += r * r;
  }
  return cost;
}

}  // namespace

void MatchObservations(const std::vector<camera::Observation>& obs_i, const std::vector<camera::Observation>& obs_j,
                       std::vector<Eigen::Vector3d>& bearing_i, std::vector<Eigen::Vector3d>& bearing_j) {
  std::unordered_map<TrackId, Vector2T> map_j;
  for (const auto& obs : obs_j) {
    if (obs.cam_id == 0) {
      map_j[obs.id] = obs.xy;
    }
  }

  for (const auto& obs : obs_i) {
    if (obs.cam_id != 0) continue;
    auto it = map_j.find(obs.id);
    if (it != map_j.end()) {
      Eigen::Vector3d bi(obs.xy.x(), obs.xy.y(), 1.0);
      Eigen::Vector3d bj(it->second.x(), it->second.y(), 1.0);
      bearing_i.push_back(bi.normalized());
      bearing_j.push_back(bj.normalized());
    }
  }
}

Vector3T SolveGyroBiasNec(const std::vector<Pose>& poses,
                          const std::vector<std::vector<camera::Observation>>& observations_per_kf,
                          const Eigen::Matrix3d& Rbc) {
  if (poses.size() < 2 || observations_per_kf.size() < poses.size()) {
    TraceDebug("SolveGyroBiasNec: insufficient data (poses=%zu, obs=%zu)", poses.size(), observations_per_kf.size());
    return Vector3T::Zero();
  }

  // Build NEC edges for each consecutive keyframe pair
  std::vector<NECEdge> edges;
  for (size_t i = 0; i + 1 < poses.size(); i++) {
    const auto& preint = poses[i].preintegration;
    if (preint.empty()) continue;
    if (preint.dT_s > kMaxPreintegrationTime) continue;
    std::vector<Vector3d> bearing_i, bearing_j;
    MatchObservations(observations_per_kf[i], observations_per_kf[i + 1], bearing_i, bearing_j);

    if (bearing_i.size() < kMinCorrespondences) continue;

    NECEdge edge;
    edge.summation = ComputeNecSummation(bearing_i, bearing_j);
    edge.JRg = preint.JRg.cast<double>();
    edge.delta_R = preint.dR.cast<double>();
    edge.Rbc = Rbc;
    edges.push_back(std::move(edge));
  }

  if (edges.size() < 2) {
    TraceDebug("SolveGyroBiasNec: insufficient valid edges (%zu)", edges.size());
    return Vector3T::Zero();
  }

  const int num_edges = static_cast<int>(edges.size());

  // Levenberg-Marquardt optimization with numerical Jacobian
  Vector3d bg = Vector3d::Zero();
  double lambda = 1e-3;
  double current_cost = ComputeTotalCost(bg, edges);

  for (int iter = 0; iter < kMaxIterations; iter++) {
    // Compute residuals and numerical Jacobian
    Eigen::VectorXd residuals(num_edges);
    Eigen::MatrixXd jacobian(num_edges, 3);

    for (int e = 0; e < num_edges; e++) {
      residuals(e) = ComputeResidual(bg, edges[e]);

      // Central differences for each bg component
      for (int j = 0; j < 3; j++) {
        Vector3d bg_plus = bg;
        Vector3d bg_minus = bg;
        bg_plus(j) += kNumericalDiffStep;
        bg_minus(j) -= kNumericalDiffStep;
        double r_plus = ComputeResidual(bg_plus, edges[e]);
        double r_minus = ComputeResidual(bg_minus, edges[e]);
        jacobian(e, j) = (r_plus - r_minus) / (2.0 * kNumericalDiffStep);
      }
    }

    // LM normal equations: (J^T J + lambda * diag(J^T J)) delta = -J^T r
    Eigen::Matrix3d JtJ = jacobian.transpose() * jacobian;
    Vector3d Jtr = jacobian.transpose() * residuals;

    Eigen::Matrix3d augmented = JtJ;
    augmented.diagonal() += lambda * JtJ.diagonal().cwiseMax(1e-6);

    Vector3d delta = augmented.ldlt().solve(-Jtr);

    // Only accept step if cost decreases
    Vector3d bg_new = bg + delta;
    double new_cost = ComputeTotalCost(bg_new, edges);

    if (new_cost < current_cost) {
      bg = bg_new;
      current_cost = new_cost;
      lambda = std::max(lambda * 0.1, 1e-10);

      if (delta.norm() < kConvergenceThreshold) {
        break;
      }
    } else {
      lambda = std::min(lambda * 10.0, 1e6);
    }
  }

  return Vector3T(static_cast<float>(bg[0]), static_cast<float>(bg[1]), static_cast<float>(bg[2]));
}

}  // namespace cuvslam::sba_imu
