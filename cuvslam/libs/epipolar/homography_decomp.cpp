
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

#include "common/types.h"

#include "epipolar/homography.h"

namespace cuvslam::epipolar {

const float ComputeHomography::ROTATIONAL_THRESHOLD = 0.0022f;

inline float clampIfZeroNeg(const float value) {
  assert(value >= 0 || abs(value) < 10 * epsilon());
  return (std::max)(float(0), value);
}

inline float oppositeOfMinor(const Matrix3T& m, const size_t row, const size_t col) {
  assert(row < 3 && col < 3);

  const size_t x1 = (col == 0) ? 1 : 0;
  const size_t x2 = (col == 2) ? 1 : 2;
  const size_t y1 = (row == 0) ? 1 : 0;
  const size_t y2 = (row == 2) ? 1 : 2;

  return m(y1, x2) * m(y2, x1) - m(y1, x1) * m(y2, x2);
}

// computes R = H( I - (2/v)*te_star*ne_t )
inline Matrix3T findRmatFrom_tstar_n(const Matrix3T& h, const Vector3T& tstar, const Vector3T& n, const float v) {
  assert(std::abs(v) > epsilon());
  return h * (Matrix3T::Identity() - (2 / v) * tstar * n.transpose());
}

bool ComputeHomography::decomposeHomography(const Matrix3T& hNorm, CameraMatrixNormalVector& camMotions) {
  if (IsRotationalMatrix(hNorm, ROTATIONAL_THRESHOLD)) {
    assert(false && "We should never be doing decompose for Rotational case");
    return false;
  }

  // S = H'H - I
  Matrix3T s = hNorm.transpose() * hNorm - Matrix3T::Identity();

  // Compute nvectors
  const float m00 = clampIfZeroNeg(oppositeOfMinor(s, 0, 0));
  const float m11 = clampIfZeroNeg(oppositeOfMinor(s, 1, 1));
  const float m22 = clampIfZeroNeg(oppositeOfMinor(s, 2, 2));

  const float rtM00 = std::sqrt(m00);
  const float rtM11 = std::sqrt(m11);
  const float rtM22 = std::sqrt(m22);

  const float m01 = oppositeOfMinor(s, 0, 1);
  const float m12 = oppositeOfMinor(s, 1, 2);
  const float m02 = oppositeOfMinor(s, 0, 2);

  const float n0 = std::abs(s(0, 0));
  const float n1 = std::abs(s(1, 1));
  const float n2 = std::abs(s(2, 2));

  // find max index of |Sii|, i = 0, 1, 2
  const size_t indx = (n0 < n1) ? ((n1 < n2) ? 2 : 1) : ((n0 < n2) ? 2 : 0);

  Vector3T vbase, vdelta;

  switch (indx) {
    case 0:
      vbase = {s(0, 0), s(0, 1), s(0, 2)};
      vdelta = {0, rtM22, std::copysign(rtM11, m12)};
      break;

    case 1:
      vbase = {s(0, 1), s(1, 1), s(1, 2)};
      vdelta = {rtM22, 0, -std::copysign(rtM00, m02)};
      break;

    case 2:
      vbase = {s(0, 2), s(1, 2), s(2, 2)};
      vdelta = {std::copysign(rtM11, m01), rtM00, 0};
      break;

    default:
      assert(false);
      break;
  }

  const Vector3T npa(vbase + vdelta);
  const Vector3T npb(vbase - vdelta);

  const float traceS = s.trace();
  const float c = 1 + traceS - (m00 + m11 + m22);
  assert(c >= 0);
  const float v = 2 * std::sqrt(c);

  const float r_2 = 2 + traceS + v;
  const float nt_2 = 2 + traceS - v;

  assert(r_2 >= 0 && nt_2 >= 0);

  const float r = std::sqrt(r_2);
  const float n_t = std::sqrt(nt_2);

  const Vector3T na = npa.normalized();
  const Vector3T nb = npb.normalized();

  const float half_nt = 0.5f * n_t;
  const float esii_t_r = std::copysign(r, s(indx, indx));

  const Vector3T ta_star = half_nt * (esii_t_r * nb - n_t * na);
  const Vector3T tb_star = half_nt * (esii_t_r * na - n_t * nb);

  // Ra, ta, na
  const Isometry3T ra(findRmatFrom_tstar_n(hNorm, ta_star, na, v));
  const Translation3T ta(ra * ta_star);

  camMotions[0] = CameraMatrixNormal(ta * ra, na);
  camMotions[1] = CameraMatrixNormal(ta.inverse() * ra, -na);

  // Rb, tb, nb
  const Isometry3T rb(findRmatFrom_tstar_n(hNorm, tb_star, nb, v));
  const Translation3T tb(rb * tb_star);

  camMotions[2] = CameraMatrixNormal(tb * rb, nb);
  camMotions[3] = CameraMatrixNormal(tb.inverse() * rb, -nb);

  return true;
}

}  // namespace cuvslam::epipolar
