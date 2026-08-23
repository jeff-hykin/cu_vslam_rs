
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

#include <limits>
#include <sstream>

#include "common/isometry.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

#include "log/log.h"

namespace cuvslam::log {

// Eigen::Transform
template <typename _Scalar, int _Dim, int _Mode, int _Options>
inline std::string transform_to_string(const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<_Scalar>::max_digits10);
  oss << "[";
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      if (i != 0 || j != 0) oss << ", ";
      auto value = v(i, j);
      if (std::isinf(value)) {
        if (value > 0) {
          value = std::numeric_limits<_Scalar>::max();
        } else {
          value = -std::numeric_limits<_Scalar>::max();
        }
      }
      oss << value;
    }
  }
  oss << "]";
  return oss.str();
}

template <class ENABLE = ENABLE_default, typename _Scalar, int _Dim, int _Mode, int _Options,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const Eigen::Transform<_Scalar, _Dim, _Mode, _Options>& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), transform_to_string(value).c_str());
}

// Eigen::Matrix
template <typename _Scalar, int _Rows, int _Cols, int _Options>
std::string matrix_to_string(const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<_Scalar>::max_digits10);
  oss << "[";
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      if (i != 0 || j != 0) oss << ", ";
      auto value = v(i, j);
      if (std::isinf(value)) {
        if (value > 0) {
          value = std::numeric_limits<_Scalar>::max();
        } else {
          value = -std::numeric_limits<_Scalar>::max();
        }
      }
      oss << value;
    }
  }
  oss << "]";
  return oss.str();
}

template <class ENABLE = ENABLE_default, class T,
          std::enable_if_t<std::is_same<T, cuvslam::storage::Vec3<float>>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Vector2T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Vector3T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Vector6T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Matrix6T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T,
          std::enable_if_t<std::is_same<T, cuvslam::storage::Mat6<float>>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Matrix7T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Matrix4T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Matrix3T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}
template <class ENABLE = ENABLE_default, class T, std::enable_if_t<std::is_same<T, cuvslam::Matrix2T>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), matrix_to_string(value).c_str());
}

// Eigen::Quaternion
template <typename _Scalar, int _Options>
std::string quaternion_to_string(const Eigen::Quaternion<_Scalar, _Options>& v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<_Scalar>::max_digits10);
  oss << "[" << v.x() << "," << v.y() << "," << v.z() << "," << v.w() << "]";
  return oss.str();
}

template <class ENABLE = ENABLE_default, class T,
          std::enable_if_t<std::is_same<T, cuvslam::QuaternionT>::value, int> = 0,
          std::enable_if_t<ENABLE::enable_, int> = 0>
void Value(const std::string& name, const T& value) {
  if (ILogger::current_logger_) ILogger::current_logger_->Value(name.c_str(), quaternion_to_string(value).c_str());
}

}  // namespace cuvslam::log
