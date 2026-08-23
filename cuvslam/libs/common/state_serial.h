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

#include <cstdint>
#include <cstring>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "common/include_eigen.h"

namespace cuvslam::serial {

// Binary state (checkpoint) writer/reader for Odometry state serialization.
//
// Format: little-endian byte stream of tagged, fixed-layout values. Every
// aggregate written by a save_state() method is preceded by a 32-bit tag so
// that a mismatched read fails fast with a clear error instead of decoding
// garbage. The overall container versioning (magic + version) is owned by the
// top-level Odometry serializer, not by this layer.

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "state serialization assumes a little-endian host");

class Writer {
public:
  void write_bytes(const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buffer_.insert(buffer_.end(), p, p + size);
  }

  template <typename T>
  void write_pod(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>, "write_pod requires a trivially copyable type");
    write_bytes(&value, sizeof(T));
  }

  void write_bool(bool value) { write_pod<uint8_t>(value ? 1 : 0); }

  void write_size(size_t value) { write_pod<uint64_t>(static_cast<uint64_t>(value)); }

  void write_tag(uint32_t tag) { write_pod<uint32_t>(tag); }

  void write_string(const std::string& s) {
    write_size(s.size());
    write_bytes(s.data(), s.size());
  }

  template <typename T>
  void write_pod_vector(const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>, "write_pod_vector requires a trivially copyable type");
    write_size(v.size());
    if (!v.empty()) {
      write_bytes(v.data(), v.size() * sizeof(T));
    }
  }

  // Serializes any fixed-size Eigen matrix (or the matrix of an Eigen::Transform) element by element in
  // row-major order, independent of the in-memory storage order.
  template <typename Derived>
  void write_eigen(const Eigen::MatrixBase<Derived>& m) {
    using Scalar = typename Derived::Scalar;
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
      for (Eigen::Index c = 0; c < m.cols(); ++c) {
        write_pod<Scalar>(m(r, c));
      }
    }
  }

  template <typename Scalar, int Dim, int Mode>
  void write_isometry(const Eigen::Transform<Scalar, Dim, Mode>& t) {
    write_eigen(t.matrix());
  }

  template <typename T>
  void write_optional_pod(const std::optional<T>& v) {
    write_bool(v.has_value());
    if (v.has_value()) {
      write_pod<T>(*v);
    }
  }

  const std::vector<uint8_t>& buffer() const { return buffer_; }
  std::vector<uint8_t> take_buffer() { return std::move(buffer_); }

private:
  std::vector<uint8_t> buffer_;
};

class Reader {
public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  explicit Reader(const std::vector<uint8_t>& buffer) : Reader(buffer.data(), buffer.size()) {}

  void read_bytes(void* out, size_t size) {
    if (pos_ + size > size_) {
      throw std::runtime_error("cuVSLAM state deserialization: unexpected end of data");
    }
    std::memcpy(out, data_ + pos_, size);
    pos_ += size;
  }

  template <typename T>
  T read_pod() {
    static_assert(std::is_trivially_copyable_v<T>, "read_pod requires a trivially copyable type");
    T value;
    read_bytes(&value, sizeof(T));
    return value;
  }

  bool read_bool() { return read_pod<uint8_t>() != 0; }

  size_t read_size() { return static_cast<size_t>(read_pod<uint64_t>()); }

  void expect_tag(uint32_t expected, const char* what) {
    const uint32_t got = read_pod<uint32_t>();
    if (got != expected) {
      throw std::runtime_error(std::string("cuVSLAM state deserialization: bad section tag for ") + what);
    }
  }

  std::string read_string() {
    const size_t size = read_size();
    std::string s(size, '\0');
    read_bytes(s.data(), size);
    return s;
  }

  template <typename T>
  std::vector<T> read_pod_vector() {
    static_assert(std::is_trivially_copyable_v<T>, "read_pod_vector requires a trivially copyable type");
    const size_t size = read_size();
    std::vector<T> v(size);
    if (size != 0) {
      read_bytes(v.data(), size * sizeof(T));
    }
    return v;
  }

  template <typename Derived>
  void read_eigen(Eigen::MatrixBase<Derived>& m) {
    using Scalar = typename Derived::Scalar;
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
      for (Eigen::Index c = 0; c < m.cols(); ++c) {
        m(r, c) = read_pod<Scalar>();
      }
    }
  }

  template <typename Scalar, int Dim, int Mode>
  void read_isometry(Eigen::Transform<Scalar, Dim, Mode>& t) {
    read_eigen(t.matrix());
  }

  template <typename T>
  std::optional<T> read_optional_pod() {
    if (!read_bool()) {
      return std::nullopt;
    }
    return read_pod<T>();
  }

  size_t remaining() const { return size_ - pos_; }
  bool at_end() const { return pos_ == size_; }

private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
};

}  // namespace cuvslam::serial
