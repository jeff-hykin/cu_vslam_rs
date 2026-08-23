
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

#pragma once

namespace cuvslam::cuda {

template <typename Scalar, int NROWS_, int NCOLS_>
struct Mat {
  using Matrix = Mat<Scalar, NROWS_, NCOLS_>;
  static const int NROWS = NROWS_;
  static const int NCOLS = NCOLS_;
  Scalar d_[NROWS_][NCOLS_];
  Scalar *operator[](int i) { return d_[i]; }
  const Scalar *operator[](int i) const { return d_[i]; }
};

using Matf11 = Mat<float, 1, 3>;
using Matf13 = Mat<float, 1, 3>;
using Matf22 = Mat<float, 2, 2>;
using Matf23 = Mat<float, 2, 3>;
using Matf16 = Mat<float, 1, 6>;
using Matf26 = Mat<float, 2, 6>;
using Matf32 = Mat<float, 3, 2>;
using Matf33 = Mat<float, 3, 3>;
using Matf36 = Mat<float, 3, 6>;
using Matf66 = Mat<float, 6, 6>;
using Matf62 = Mat<float, 6, 2>;
using Matf99 = Mat<float, 9, 9>;
using Matf93 = Mat<float, 9, 3>;
using Pose = Mat<float, 4, 4>;
// using Matf26 = Mat<float, 2, 6>;
using Matd33 = Mat<double, 3, 3>;

template <typename Scalar, int DIM_>
struct Vec {
  static const int DIM = DIM_;
  Scalar d_[DIM_];
  Scalar &operator[](int i) { return d_[i]; }
  const Scalar operator[](int i) const { return d_[i]; }
};

using Vecf1 = Vec<float, 1>;
using Vecf2 = Vec<float, 2>;
using Vecf3 = Vec<float, 3>;
using Vecf6 = Vec<float, 6>;
using Vecf9 = Vec<float, 9>;

using Vecd3 = Vec<double, 3>;

}  // namespace cuvslam::cuda
