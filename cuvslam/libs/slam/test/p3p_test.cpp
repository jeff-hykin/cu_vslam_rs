
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

#include <iostream>

#include "Eigen/Core"
#include "Eigen/Eigenvalues"

namespace {
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Remove leading terms with zero coefficients.
VectorXd RemoveLeadingZeros(const VectorXd& polynomial_in) {
  int i = 0;
  while (i < (polynomial_in.size() - 1) && polynomial_in(i) == 0) {
    ++i;
  }
  return polynomial_in.tail(polynomial_in.size() - i);
}
void FindLinearPolynomialRoots(const VectorXd& polynomial, VectorXd* real, VectorXd* imaginary) {
  // CHECK_EQ(polynomial.size(), 2);
  if (real != NULL) {
    real->resize(1);
    (*real)(0) = -polynomial(1) / polynomial(0);
  }

  if (imaginary != NULL) {
    imaginary->setZero(1);
  }
}

void FindQuadraticPolynomialRoots(const VectorXd& polynomial, VectorXd* real, VectorXd* imaginary) {
  // CHECK_EQ(polynomial.size(), 3);
  const double a = polynomial(0);
  const double b = polynomial(1);
  const double c = polynomial(2);
  const double D = b * b - 4 * a * c;
  const double sqrt_D = sqrt(fabs(D));
  if (real != NULL) {
    real->setZero(2);
  }
  if (imaginary != NULL) {
    imaginary->setZero(2);
  }

  // Real roots.
  if (D >= 0) {
    if (real != NULL) {
      // Stable quadratic roots according to BKP Horn.
      // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
      if (b >= 0) {
        (*real)(0) = (-b - sqrt_D) / (2.0 * a);
        (*real)(1) = (2.0 * c) / (-b - sqrt_D);
      } else {
        (*real)(0) = (2.0 * c) / (-b + sqrt_D);
        (*real)(1) = (-b + sqrt_D) / (2.0 * a);
      }
    }
    return;
  }

  // Use the normal quadratic formula for the complex case.
  if (real != NULL) {
    (*real)(0) = -b / (2.0 * a);
    (*real)(1) = -b / (2.0 * a);
  }
  if (imaginary != NULL) {
    (*imaginary)(0) = sqrt_D / (2.0 * a);
    (*imaginary)(1) = -sqrt_D / (2.0 * a);
  }
}

void BuildCompanionMatrix(const VectorXd& polynomial, MatrixXd* companion_matrix_ptr) {
  // CHECK_NOTNULL(companion_matrix_ptr);
  MatrixXd& companion_matrix = *companion_matrix_ptr;

  const int degree = polynomial.size() - 1;

  companion_matrix.resize(degree, degree);
  companion_matrix.setZero();
  companion_matrix.diagonal(-1).setOnes();
  companion_matrix.col(degree - 1) = -polynomial.reverse().head(degree);
}

void BalanceCompanionMatrix(MatrixXd* companion_matrix_ptr) {
  // CHECK_NOTNULL(companion_matrix_ptr);
  MatrixXd& companion_matrix = *companion_matrix_ptr;
  MatrixXd companion_matrix_offdiagonal = companion_matrix;
  companion_matrix_offdiagonal.diagonal().setZero();

  const int degree = companion_matrix.rows();

  // gamma <= 1 controls how much a change in the scaling has to
  // lower the 1-norm of the companion matrix to be accepted.
  //
  // gamma = 1 seems to lead to cycles (numerical issues?), so
  // we set it slightly lower.
  const double gamma = 0.9;

  // Greedily scale row/column pairs until there is no change.
  bool scaling_has_changed;
  do {
    scaling_has_changed = false;

    for (int i = 0; i < degree; ++i) {
      const double row_norm = companion_matrix_offdiagonal.row(i).lpNorm<1>();
      const double col_norm = companion_matrix_offdiagonal.col(i).lpNorm<1>();

      // Decompose row_norm/col_norm into mantissa * 2^exponent,
      // where 0.5 <= mantissa < 1. Discard mantissa (return value
      // of frexp), as only the exponent is needed.
      int exponent = 0;
      std::frexp(row_norm / col_norm, &exponent);
      exponent /= 2;

      if (exponent != 0) {
        const double scaled_col_norm = std::ldexp(col_norm, exponent);
        const double scaled_row_norm = std::ldexp(row_norm, -exponent);
        if (scaled_col_norm + scaled_row_norm < gamma * (col_norm + row_norm)) {
          // Accept the new scaling. (Multiplication by powers of 2 should not
          // introduce rounding errors (ignoring non-normalized numbers and
          // over- or underflow))
          scaling_has_changed = true;
          companion_matrix_offdiagonal.row(i) *= std::ldexp(1.0, -exponent);
          companion_matrix_offdiagonal.col(i) *= std::ldexp(1.0, exponent);
        }
      }
    }
  } while (scaling_has_changed);

  companion_matrix_offdiagonal.diagonal() = companion_matrix.diagonal();
  companion_matrix = companion_matrix_offdiagonal;
}

bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& polynomial_in, Eigen::VectorXd* real,
                                        Eigen::VectorXd* imaginary) {
  if (polynomial_in.size() == 0) {
    std::cout << "Invalid polynomial of size 0 passed to FindPolynomialRoots";
    return false;
  }

  VectorXd polynomial = RemoveLeadingZeros(polynomial_in);
  const int degree = polynomial.size() - 1;

  // Is the polynomial constant?
  if (degree == 0) {
    std::cout << "Trying to extract roots from a constant "
              << "polynomial in FindPolynomialRootsCompanionMatrix";
    // We return true with no roots, not false, as if the polynomial is constant
    // it is correct that there are no roots. It is not the case that they were
    // there, but that we have failed to extract them.
    return true;
  }

  // Linear
  if (degree == 1) {
    FindLinearPolynomialRoots(polynomial, real, imaginary);
    return true;
  }

  // Quadratic
  if (degree == 2) {
    FindQuadraticPolynomialRoots(polynomial, real, imaginary);
    return true;
  }

  // The degree is now known to be at least 3. For cubic or higher
  // roots we use the method of companion matrices.

  // Divide by leading term
  const double leading_term = polynomial(0);
  polynomial /= leading_term;

  // Build and balance the companion matrix to the polynomial.
  MatrixXd companion_matrix(degree, degree);
  BuildCompanionMatrix(polynomial, &companion_matrix);
  BalanceCompanionMatrix(&companion_matrix);

  // Find its (complex) eigenvalues.
  Eigen::EigenSolver<MatrixXd> solver(companion_matrix, false);
  if (solver.info() != Eigen::Success) {
    std::cout << "Failed to extract eigenvalues from companion matrix.";
    return false;
  }

  // Output roots
  if (real != NULL) {
    *real = solver.eigenvalues().real();
  } else {
    std::cout << "NULL pointer passed as real argument to "
              << "FindPolynomialRoots. Real parts of the roots will not "
              << "be returned.";
  }
  if (imaginary != NULL) {
    *imaginary = solver.eigenvalues().imag();
  }
  return true;
}

bool FindPolynomialRoots(const VectorXd& polynomial, VectorXd* real, VectorXd* imaginary) {
  return FindPolynomialRootsCompanionMatrix(polynomial, real, imaginary);
}

// Solves for cos(theta) that will describe the rotation of the plane from
// intermediate world frame to intermediate camera frame. The method returns the
// roots of a quartic (i.e. solutions to cos(alpha) ) and several factors that
// are needed for back-substitution.
int SolvePlaneRotation(const Vector3d normalized_image_points[3], const Vector3d& intermediate_image_point,
                       const Vector3d& intermediate_world_point, const double d_12, double cos_theta[4],
                       double cot_alphas[4], double* b) {
  // Calculate these parameters ahead of time for reuse and
  // readability. Notation for these variables is consistent with the notation
  // from the paper.
  const double f_1 = intermediate_image_point[0] / intermediate_image_point[2];
  const double f_2 = intermediate_image_point[1] / intermediate_image_point[2];
  const double p_1 = intermediate_world_point[0];
  const double p_2 = intermediate_world_point[1];
  const double cos_beta = normalized_image_points[0].dot(normalized_image_points[1]);
  *b = 1.0 / (1.0 - cos_beta * cos_beta) - 1.0;

  if (cos_beta < 0) {
    *b = -sqrt(*b);
  } else {
    *b = sqrt(*b);
  }

  // Definition of temporary variables for readability in the coefficients
  // calculation.
  const double f_1_pw2 = f_1 * f_1;
  const double f_2_pw2 = f_2 * f_2;
  const double p_1_pw2 = p_1 * p_1;
  const double p_1_pw3 = p_1_pw2 * p_1;
  const double p_1_pw4 = p_1_pw3 * p_1;
  const double p_2_pw2 = p_2 * p_2;
  const double p_2_pw3 = p_2_pw2 * p_2;
  const double p_2_pw4 = p_2_pw3 * p_2;
  const double d_12_pw2 = d_12 * d_12;
  const double b_pw2 = (*b) * (*b);

  // Computation of coefficients of 4th degree polynomial.
  Eigen::VectorXd coefficients(5);
  coefficients(0) = -f_2_pw2 * p_2_pw4 - p_2_pw4 * f_1_pw2 - p_2_pw4;
  coefficients(1) =
      2.0 * p_2_pw3 * d_12 * (*b) + 2.0 * f_2_pw2 * p_2_pw3 * d_12 * (*b) - 2.0 * f_2 * p_2_pw3 * f_1 * d_12;
  coefficients(2) = -f_2_pw2 * p_2_pw2 * p_1_pw2 - f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2 - f_2_pw2 * p_2_pw2 * d_12_pw2 +
                    f_2_pw2 * p_2_pw4 + p_2_pw4 * f_1_pw2 + 2.0 * p_1 * p_2_pw2 * d_12 +
                    2.0 * f_1 * f_2 * p_1 * p_2_pw2 * d_12 * (*b) - p_2_pw2 * p_1_pw2 * f_1_pw2 +
                    2.0 * p_1 * p_2_pw2 * f_2_pw2 * d_12 - p_2_pw2 * d_12_pw2 * b_pw2 - 2.0 * p_1_pw2 * p_2_pw2;
  coefficients(3) = 2.0 * p_1_pw2 * p_2 * d_12 * (*b) + 2.0 * f_2 * p_2_pw3 * f_1 * d_12 -
                    2.0 * f_2_pw2 * p_2_pw3 * d_12 * (*b) - 2.0 * p_1 * p_2 * d_12_pw2 * (*b);
  coefficients(4) = -2 * f_2 * p_2_pw2 * f_1 * p_1 * d_12 * (*b) + f_2_pw2 * p_2_pw2 * d_12_pw2 + 2.0 * p_1_pw3 * d_12 -
                    p_1_pw2 * d_12_pw2 + f_2_pw2 * p_2_pw2 * p_1_pw2 - p_1_pw4 - 2.0 * f_2_pw2 * p_2_pw2 * p_1 * d_12 +
                    p_2_pw2 * f_1_pw2 * p_1_pw2 + f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2;

  // Computation of roots.
  Eigen::VectorXd roots;
  FindPolynomialRoots(coefficients, &roots, NULL);

  // Calculate cot(alpha) needed for back-substitution.
  for (int i = 0; i < roots.size(); i++) {
    cos_theta[i] = roots(i);
    cot_alphas[i] =
        (-f_1 * p_1 / f_2 - cos_theta[i] * p_2 + d_12 * (*b)) / (-f_1 * cos_theta[i] * p_2 / f_2 + p_1 - d_12);
  }

  return static_cast<int>(roots.size());
}

// Given the complete transformation between intermediate world and camera
// frames (parameterized by cos_theta and cot_alpha), back-substitute the
// solution and get an absolute camera pose.
void Backsubstitute(const Matrix3d& intermediate_world_frame, const Matrix3d& intermediate_camera_frame,
                    const Vector3d& world_point_0, const double cos_theta, const double cot_alpha, const double d_12,
                    const double b, Vector3d* translation, Matrix3d* rotation) {
  const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  const double sin_alpha = sqrt(1.0 / (cot_alpha * cot_alpha + 1.0));
  double cos_alpha = sqrt(1.0 - sin_alpha * sin_alpha);

  if (cot_alpha < 0) {
    cos_alpha = -cos_alpha;
  }

  // Get the camera position in the intermediate world frame
  // coordinates. (Eq. 5 from the paper).
  const Vector3d c_nu(d_12 * cos_alpha * (sin_alpha * b + cos_alpha),
                      cos_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
                      sin_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha));

  // Transform c_nu into world coordinates. Use a Map to put the solution
  // directly into the output.
  *translation = world_point_0 + intermediate_world_frame.transpose() * c_nu;

  // Construct the transformation from the intermediate world frame to the
  // intermediate camera frame.
  Matrix3d intermediate_world_to_camera_rotation;
  intermediate_world_to_camera_rotation << -cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta, sin_alpha,
      -cos_alpha * cos_theta, -cos_alpha * sin_theta, 0, -sin_theta, cos_theta;

  // Construct the rotation matrix.
  *rotation = (intermediate_world_frame.transpose() * intermediate_world_to_camera_rotation.transpose() *
               intermediate_camera_frame)
                  .transpose();

  // Adjust translation to account for rotation.
  *translation = -(*rotation) * (*translation);
}

}  // namespace

namespace test::slam {
using Eigen::Vector2d;
using Eigen::VectorXcd;

bool PoseFromThreePoints(const Vector2d feature_point[3], const Vector3d points_3d[3],
                         std::vector<Matrix3d>* solution_rotations, std::vector<Vector3d>* solution_translations);

bool PoseFromThreePoints(const Vector2d feature_point[3], const Vector3d points_3d[3],
                         std::vector<Matrix3d>* solution_rotations, std::vector<Vector3d>* solution_translations) {
  Vector3d normalized_image_points[3];
  // Store points_3d in world_points for ease of use. NOTE: we cannot use a
  // const ref or a Map because the world_points entries may be swapped later.
  Vector3d world_points[3];
  for (int i = 0; i < 3; ++i) {
    normalized_image_points[i] = feature_point[i].homogeneous().normalized();
    world_points[i] = points_3d[i];
  }

  // If the points are collinear, there are no possible solutions.
  double kTolerance = 1e-6;
  Vector3d world_1_0 = world_points[1] - world_points[0];
  Vector3d world_2_0 = world_points[2] - world_points[0];
  if (world_1_0.cross(world_2_0).squaredNorm() < kTolerance) {
    std::cout << "The 3 world points are collinear! No solution for absolute "
                 "pose exits.";
    return false;
  }

  // Create intermediate camera frame such that the x axis is in the direction
  // of one of the normalized image points, and the origin is the same as the
  // absolute camera frame. This is a rotation defined as the transformation:
  // T = [tx, ty, tz] where tx = f0, tz = (f0 x f1) / ||f0 x f1||, and
  // ty = tx x tz and f0, f1, f2 are the normalized image points.
  Matrix3d intermediate_camera_frame;
  intermediate_camera_frame.row(0) = normalized_image_points[0];
  intermediate_camera_frame.row(2) = normalized_image_points[0].cross(normalized_image_points[1]).normalized();
  intermediate_camera_frame.row(1) = intermediate_camera_frame.row(2).cross(intermediate_camera_frame.row(0));

  // Project the third world point into the intermediate camera frame.
  Vector3d intermediate_image_point = intermediate_camera_frame * normalized_image_points[2];

  // Enforce that the intermediate_image_point is in front of the intermediate
  // camera frame. If the point is behind the camera frame, recalculate the
  // intermediate camera frame by swapping which feature we align the x axis to.
  if (intermediate_image_point[2] > 0) {
    std::swap(normalized_image_points[0], normalized_image_points[1]);

    intermediate_camera_frame.row(0) = normalized_image_points[0];
    intermediate_camera_frame.row(2) = normalized_image_points[0].cross(normalized_image_points[1]).normalized();
    intermediate_camera_frame.row(1) = intermediate_camera_frame.row(2).cross(intermediate_camera_frame.row(0));

    intermediate_image_point = intermediate_camera_frame * normalized_image_points[2];

    std::swap(world_points[0], world_points[1]);
    world_1_0 = world_points[1] - world_points[0];
    world_2_0 = world_points[2] - world_points[0];
  }

  // Create the intermediate world frame transformation that has the
  // origin at world_points[0] and the x-axis in the direction of
  // world_points[1]. This is defined by the transformation: N = [nx, ny, nz]
  // where nx = (p1 - p0) / ||p1 - p0||
  // nz = nx x (p2 - p0) / || nx x (p2 -p0) || and ny = nz x nx
  // Where p0, p1, p2 are the world points.
  Matrix3d intermediate_world_frame;
  intermediate_world_frame.row(0) = world_1_0.normalized();
  intermediate_world_frame.row(2) = intermediate_world_frame.row(0).cross(world_2_0).normalized();
  intermediate_world_frame.row(1) = intermediate_world_frame.row(2).cross(intermediate_world_frame.row(0));

  // Transform world_point[2] to the intermediate world frame coordinates.
  Vector3d intermediate_world_point = intermediate_world_frame * world_2_0;

  // Distance from world_points[1] to the intermediate world frame origin.
  double d_12 = world_1_0.norm();

  // Solve for the cos(theta) that will give us the transformation from
  // intermediate world frame to intermediate camera frame. We also get the
  // cot(alpha) for each solution necessary for back-substitution.
  double cos_theta[4];
  double cot_alphas[4];
  double b;
  const int num_solutions = SolvePlaneRotation(normalized_image_points, intermediate_image_point,
                                               intermediate_world_point, d_12, cos_theta, cot_alphas, &b);

  // Backsubstitution of each solution
  solution_translations->resize(num_solutions);
  solution_rotations->resize(num_solutions);
  for (int i = 0; i < num_solutions; i++) {
    Backsubstitute(intermediate_world_frame, intermediate_camera_frame, world_points[0], cos_theta[i], cot_alphas[i],
                   d_12, b, &solution_translations->at(i), &solution_rotations->at(i));
  }

  return num_solutions > 0;
}

}  // namespace test::slam
