
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

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace cuvslam {

// This namespace contains "plain" types which are suitable for unaligned storage.
// Key part is Eigen::DontAlign flag.
// The main intent to have this namespace is to simplify memory management and
// GPU interop.
namespace storage {
constexpr auto kMatrixStorageOptions = Eigen::DontAlign | Eigen::ColMajor;
constexpr auto kVectorStorageOptions = Eigen::DontAlign;

template <class T>
using Pose = Eigen::Transform<T, 3, Eigen::Isometry, kMatrixStorageOptions>;
template <class T>
using Mat2 = Eigen::Matrix<T, 2, 2, kMatrixStorageOptions>;
template <class T>
using Mat3 = Eigen::Matrix<T, 3, 3, kMatrixStorageOptions>;
template <class T>
using Mat6 = Eigen::Matrix<T, 6, 6, kMatrixStorageOptions>;
template <class T>
using Vec2 = Eigen::Matrix<T, 2, 1, kVectorStorageOptions>;
template <class T>
using Vec3 = Eigen::Matrix<T, 3, 1, kVectorStorageOptions>;
template <class T>
using Vec4 = Eigen::Matrix<T, 4, 1, kVectorStorageOptions>;
template <class T>
using Vec6 = Eigen::Matrix<T, 6, 1, kVectorStorageOptions>;

template <class T, int N>
using Vec = Eigen::Matrix<T, N, 1, kVectorStorageOptions>;
template <class T, int M, int N>
using Mat = Eigen::Matrix<T, M, N, kVectorStorageOptions>;

template <class T>
using Isometry3 = Eigen::Transform<T, 3, Eigen::Isometry, kMatrixStorageOptions>;
template <class T>
using Affine2 = Eigen::Transform<T, 2, Eigen::Affine, kMatrixStorageOptions>;
}  // namespace storage

// overloads for const-size containers (e.g. std::array), requirements: data(), constexpr size(), value_type
template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec(T&& arr) {
  using ContainerType = std::remove_reference_t<T>;
  using MatrixType =
      std::conditional_t<std::is_const_v<ContainerType>, const storage::Vec<typename ContainerType::value_type, N>,
                         storage::Vec<typename ContainerType::value_type, N>>;
  static_assert(N == ContainerType{}.size());
  return Eigen::Map<MatrixType, alignment>(arr.data());
}

template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned,
          typename = std::enable_if_t<!std::is_array_v<std::remove_reference_t<T>>>>
inline auto mat(T&& arr) {
  using ContainerType = std::remove_reference_t<T>;
  using MatrixType =
      std::conditional_t<std::is_const_v<ContainerType>, const storage::Mat<typename ContainerType::value_type, N, N>,
                         storage::Mat<typename ContainerType::value_type, N, N>>;
  static_assert(N * N == ContainerType{}.size());
  return Eigen::Map<MatrixType, alignment>(arr.data());
}

// array overloads
template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec(const T (&x)[N]) {
  return Eigen::Map<const cuvslam::storage::Vec<T, N>, alignment>(x);
}

template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec(T (&x)[N]) {
  return Eigen::Map<cuvslam::storage::Vec<T, N>, alignment>(x);
}

template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned,
          typename = std::enable_if_t<std::is_floating_point_v<T>>, int N2 = N* N>
inline auto mat(const T (&x)[N2]) {
  return Eigen::Map<const cuvslam::storage::Mat<T, N, N>, alignment>(x);
}

template <int N, class T, Eigen::AlignmentType alignment = Eigen::Unaligned,
          typename = std::enable_if_t<std::is_floating_point_v<T>>, int N2 = N* N>
inline auto mat(T (&x)[N2]) {
  return Eigen::Map<cuvslam::storage::Mat<T, N, N>, alignment>(x);
}

// ponter overloads, AVOID
template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec2(const T* x) {
  return Eigen::Map<const cuvslam::storage::Vec2<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec3(const T* x) {
  return Eigen::Map<const cuvslam::storage::Vec3<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto mat2(const T* x) {
  return Eigen::Map<const cuvslam::storage::Mat2<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec2(T* x) {
  return Eigen::Map<cuvslam::storage::Vec2<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto vec3(T* x) {
  return Eigen::Map<cuvslam::storage::Vec3<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto mat2(T* x) {
  return Eigen::Map<cuvslam::storage::Mat2<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto mat3(T* x) {
  return Eigen::Map<cuvslam::storage::Mat3<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto mat3(const T* x) {
  return Eigen::Map<const cuvslam::storage::Mat3<T>, alignment>(x);
}

template <class T, Eigen::AlignmentType alignment = Eigen::Unaligned>
inline auto mat6(T* x) {
  return Eigen::Map<cuvslam::storage::Mat6<T>, alignment>(x);
}
}  // namespace cuvslam
