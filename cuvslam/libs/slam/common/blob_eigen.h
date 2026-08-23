
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

#include "common/include_eigen.h"

#include "blob.h"

namespace cuvslam::slam {

template <typename _Scalar, int _Rows, int _Cols, int _Options>
void write(BlobWriter& blob_writer, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      blob_writer.write(v(i, j));
    }
  }
}

template <typename _Scalar, int _Dim, int _Mode, int _Options>
void write(BlobWriter& blob_writer, const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      blob_writer.write(v(i, j));
    }
  }
}

template <typename _Scalar, int _Rows, int _Cols, int _Options>
bool read(const BlobReader& blob_reader, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      if (!blob_reader.read(v(i, j))) {
        return false;
      }
    }
  }
  return true;
}
template <typename _Scalar, int _Dim, int _Mode, int _Options>
bool read(const BlobReader& blob_reader, Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& v) {
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      if (!blob_reader.read(v(i, j))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace cuvslam::slam
