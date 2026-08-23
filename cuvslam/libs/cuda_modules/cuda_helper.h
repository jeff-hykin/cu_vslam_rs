
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
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "common/image_matrix.h"
#include "common/log.h"
#include "cuvslam/cuvslam_gpu.h"

#if !defined(NDEBUG)
#define ERROR_MESSAGE(...) TraceError("%s:%d: %s", __FILE__, __LINE__, __VA_ARGS__)
#else
#define ERROR_MESSAGE(...) TraceError("%s", __VA_ARGS__)
#endif

#define CUDA_CHECK_IMPL(status, prefix, to_string, is_throw)               \
  do {                                                                     \
    auto ret = (status);                                                   \
    if (ret != 0) {                                                        \
      std::stringstream msg;                                               \
      msg << prefix << " error " << (to_string)(ret) << "(" << ret << ")"; \
      ERROR_MESSAGE(msg.str().c_str());                                    \
      if constexpr (is_throw) {                                            \
        throw std::runtime_error(msg.str());                               \
      }                                                                    \
    }                                                                      \
  } while (0)

#define CUDA_CHECK(status) CUDA_CHECK_IMPL(status, "[CUDA]", cudaGetErrorString, true)
#define CUDA_CHECK_NOTHROW(status) CUDA_CHECK_IMPL(status, "[CUDA]", cudaGetErrorString, false)

namespace cuvslam::cuda {

enum GPUCopyDirection { ToGPU, ToCPU };

// CUDA allocation and copy APIs take byte counts. Most callers deal in element
// counts derived from images, tracks, or optimization problem sizes, so keep the
// overflow check at the conversion point instead of relying on every caller to
// prove its input range first.
inline size_t CheckedMul(size_t lhs, size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw std::bad_array_new_length();
  }
  return lhs * rhs;
}

template <typename T>
size_t CheckedBytes(size_t elements) {
  return CheckedMul(elements, sizeof(T));
}

template <typename T>
class GPUArray {
public:
  explicit GPUArray(size_t size) : size_(size) {
    const size_t array_size = CheckedBytes<T>(size_);
    cpu_elements.resize(size);
    CUDA_CHECK(cudaMalloc((void**)&data_, array_size));
  }

  T& operator[](size_t i) {
    assert(i < size_);
    return cpu_elements[i];
  }

  void copy(GPUCopyDirection direction, cudaStream_t s) {
    const size_t array_size = CheckedBytes<T>(size_);
    if (direction == ToGPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)data_, (void*)cpu_elements.data(), array_size, cudaMemcpyHostToDevice, s));
    } else if (direction == ToCPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)cpu_elements.data(), (void*)data_, array_size, cudaMemcpyDeviceToHost, s));
    } else {
      TraceError("Unsupported copy direction");
    }
  }

  void copy_top_n(GPUCopyDirection direction, size_t top_n, cudaStream_t s) {
    if (top_n == 0) {
      return;
    }
    // This is a runtime boundary, not just a debug invariant. In release builds
    // an oversized top_n would otherwise wrap the byte count and copy past the
    // allocation that CUDA accepted.
    if (top_n > size_) {
      throw std::out_of_range("copy_top_n exceeds GPUArray size");
    }
    const size_t array_size = CheckedBytes<T>(top_n);
    if (direction == ToGPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)data_, (void*)cpu_elements.data(), array_size, cudaMemcpyHostToDevice, s));
    } else if (direction == ToCPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)cpu_elements.data(), (void*)data_, array_size, cudaMemcpyDeviceToHost, s));
    } else {
      TraceError("Unsupported copy direction");
    }
  }

  ~GPUArray() { CUDA_CHECK_NOTHROW(cudaFree(data_)); }

  T* ptr() const { return data_; }

  size_t size() const { return size_; }

private:
  T* data_ = nullptr;
  size_t size_ = 0;
  std::vector<T> cpu_elements;
};

template <typename T>
class GPUArrayPinned {
public:
  explicit GPUArrayPinned(size_t size) { alloc(size); }

  T& operator[](size_t i) {
    assert(i < size_);
    return host_data_[i];
  }

  const T& operator[](size_t i) const {
    assert(i < size_);
    return host_data_[i];
  }

  void copy(GPUCopyDirection direction, cudaStream_t s) const {
    const size_t array_size = CheckedBytes<T>(size_);
    if (direction == ToGPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)data_, (void*)host_data_, array_size, cudaMemcpyHostToDevice, s));
    } else if (direction == ToCPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)host_data_, (void*)data_, array_size, cudaMemcpyDeviceToHost, s));
    } else {
      TraceError("Unsupported copy direction");
    }
  }

  void copy_top_n(GPUCopyDirection direction, size_t top_n, cudaStream_t s) const {
    if (top_n == 0) {
      return;
    }
    // Pinned buffers are shared by host and device copies. Reject the bad range
    // before computing bytes so both sides keep the same allocation contract.
    if (top_n > size_) {
      throw std::out_of_range("copy_top_n exceeds GPUArrayPinned size");
    }
    const size_t array_size = CheckedBytes<T>(top_n);
    if (direction == ToGPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)data_, (void*)host_data_, array_size, cudaMemcpyHostToDevice, s));
    } else if (direction == ToCPU) {
      CUDA_CHECK(cudaMemcpyAsync((void*)host_data_, (void*)data_, array_size, cudaMemcpyDeviceToHost, s));
    } else {
      TraceError("Unsupported copy direction");
    }
  }

  ~GPUArrayPinned() { free(); }

  T* ptr() const { return data_; }

  size_t size() const { return size_; }

  void resize(size_t size) {
    if (size_ == size) {
      return;
    }
    if (data_ && host_data_) {
      free();
    }
    alloc(size);
  }

private:
  void free() {
    assert(data_ && host_data_);
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFreeHost(host_data_));
    data_ = host_data_ = nullptr;
    size_ = 0;
  }

  void alloc(size_t size) {
    assert(!data_ && !host_data_);
    const size_t array_size = CheckedBytes<T>(size);
    CUDA_CHECK(cudaHostAlloc((void**)&host_data_, array_size, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc((void**)&data_, array_size));
    size_ = size;
  }

  T* data_ = nullptr;
  size_t size_ = 0;
  T* host_data_ = nullptr;
};

template <typename T>
class GPUOnlyArray {
public:
  GPUOnlyArray() = default;

  explicit GPUOnlyArray(size_t size) : size_(size) {
    const size_t array_size = CheckedBytes<T>(size_);
    CUDA_CHECK(cudaMalloc((void**)&data_, array_size));
  }

  GPUOnlyArray(const GPUOnlyArray&) = delete;

  GPUOnlyArray(GPUOnlyArray&& other) noexcept {
    if (data_) {
      CUDA_CHECK_NOTHROW(cudaFree(data_));
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  };

  ~GPUOnlyArray() {
    if (data_) {
      CUDA_CHECK_NOTHROW(cudaFree(data_));
    }
  }

  T* ptr() const { return data_; }

  size_t size() const { return size_; }

private:
  T* data_ = nullptr;
  size_t size_ = 0;
};

template <typename T>
class GPUImage {
private:
  T* data_ = nullptr;
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t pitch_ = 0;

  cudaTextureObject_t texture_filter_point = 0;
  cudaTextureObject_t texture_filter_linear = 0;

public:
  GPUImage() = default;

  GPUImage& operator=(GPUImage&& other) noexcept = default;

  explicit GPUImage(const cuvslam::ImageMatrix<T>& image) {
    rows_ = image.rows();
    cols_ = image.cols();

    // Pitch allocations are row-oriented: CUDA needs bytes per row and rows
    // separately. Checking the row byte count here prevents a wrapped pitch from
    // describing less memory than the image copy will later address.
    const size_t row_size = CheckedBytes<T>(cols_);
    CUDA_CHECK(cudaMallocPitch((void**)&data_, &pitch_, row_size, rows_));
    CUDA_CHECK(
        cudaMemcpy2D((void*)data_, pitch_, (void*)image.data(), row_size, row_size, rows_, cudaMemcpyHostToDevice));
    prepare_texture();
  }

  GPUImage(const GPUImage& image) {
    rows_ = image.rows();
    cols_ = image.cols();
    pitch_ = image.pitch();

    const size_t row_size = CheckedBytes<T>(cols_);
    CUDA_CHECK(cudaMallocPitch(&data_, &pitch_, row_size, rows_));
    CUDA_CHECK(
        cudaMemcpy2D((void*)data_, pitch_, (void*)image.data_, pitch_, row_size, rows_, cudaMemcpyDeviceToDevice));
    prepare_texture();
  }

  GPUImage(size_t cols, size_t rows) : rows_(rows), cols_(cols) {
    CUDA_CHECK(cudaMallocPitch((void**)&data_, &pitch_, CheckedBytes<T>(cols_), rows_));
    prepare_texture();
  }

  void init(size_t cols, size_t rows) {
    if (data_ == nullptr) {
      rows_ = rows;
      cols_ = cols;
      CUDA_CHECK(cudaMallocPitch((void**)&data_, &pitch_, CheckedBytes<T>(cols_), rows_));
      prepare_texture();
    }
  }

  void copy(GPUCopyDirection direction, const T* image_ptr, cudaStream_t s) const {
    assert(data_ != nullptr);
    const size_t pitch = CheckedBytes<T>(cols_);
    if (direction == ToGPU) {
      CUDA_CHECK(
          cudaMemcpy2DAsync((void*)data_, pitch_, (void*)image_ptr, pitch, pitch, rows_, cudaMemcpyHostToDevice, s));
    } else if (direction == ToCPU) {
      CUDA_CHECK(
          cudaMemcpy2DAsync((void*)image_ptr, pitch, (void*)data_, pitch_, pitch, rows_, cudaMemcpyDeviceToHost, s));
    } else {
      TraceError("Unsupported copy direction");
    }
  }

  GPUImage& operator=(const GPUImage& image) {
    if (this == &image) {
      return *this;
    }

    assert(rows_ == image.rows_);
    assert(cols_ == image.cols_);
    assert(pitch_ == image.pitch_);
    assert(data_ != nullptr);
    assert(image.data_ != nullptr);

    CUDA_CHECK(cudaMemcpy2D((void*)data_, pitch_, image.data_, pitch_, CheckedBytes<T>(cols_), rows_,
                            cudaMemcpyDeviceToDevice));
    return *this;
  }

  GPUImage& operator=(const cuvslam::ImageMatrix<T>& image) {
    assert(static_cast<size_t>(image.cols()) == cols_);
    assert(static_cast<size_t>(image.rows()) == rows_);
    assert(data_ != nullptr);

    const size_t pitch = CheckedBytes<T>(cols_);
    CUDA_CHECK(cudaMemcpy2D((void*)data_, pitch_, (void*)image.data(), pitch, pitch, rows_, cudaMemcpyHostToDevice));
    return *this;
  }

  ~GPUImage() {
    if (texture_filter_point != 0) {
      CUDA_CHECK_NOTHROW(cudaDestroyTextureObject(texture_filter_point));
    }
    if (texture_filter_linear != 0) {
      CUDA_CHECK_NOTHROW(cudaDestroyTextureObject(texture_filter_linear));
    }
    if (data_ != nullptr) {
      CUDA_CHECK_NOTHROW(cudaFree(data_));
    }
  }

  T* ptr() const {
    assert(data_ != nullptr);
    return data_;
  }

  size_t cols() const {
    assert(cols_ != 0);
    return cols_;
  }

  size_t rows() const {
    assert(rows_ != 0);
    return rows_;
  }

  size_t pitch() const {
    assert(pitch_ != 0);
    return pitch_;
  }

  size_t size() const {
    assert(rows_ != 0);
    assert(cols_ != 0);
    // Preserve the same overflow contract for callers that size follow-on
    // buffers from an image rather than allocating the image directly.
    return CheckedMul(rows_, cols_);
  }

  cudaTextureObject_t get_texture_filter_point() const {
    assert(texture_filter_point != 0);
    return texture_filter_point;
  }

  cudaTextureObject_t get_texture_filter_linear() const {
    assert(texture_filter_linear != 0);
    return texture_filter_linear;
  }

private:
  void prepare_texture() {
    cudaResourceDesc resDesc;
    cudaChannelFormatDesc channelDesc;
    cudaTextureDesc texDesc;

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;

    channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    resDesc.res.pitch2D.devPtr = (void*)data_;
    resDesc.res.pitch2D.width = cols_;
    resDesc.res.pitch2D.height = rows_;
    resDesc.res.pitch2D.pitchInBytes = pitch_;
    resDesc.res.pitch2D.desc = channelDesc;

    // Specify texture object parameters
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeMirror;
    texDesc.addressMode[1] = cudaAddressModeMirror;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&texture_filter_point, &resDesc, &texDesc, NULL));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    CUDA_CHECK(cudaCreateTextureObject(&texture_filter_linear, &resDesc, &texDesc, NULL));
  }
};

using GPUImageT = GPUImage<float>;
using GPUImage8 = GPUImage<uint8_t>;
using GPUImage16 = GPUImage<uint16_t>;

template <class T>
struct HostAllocator {
  using value_type [[maybe_unused]] = T;

  HostAllocator() = default;

  template <class U>
  constexpr HostAllocator(const HostAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    const size_t array_size = CheckedBytes<T>(n);

    void* p;
    if (cudaSuccess == cudaHostAlloc(&p, array_size, cudaHostAllocDefault)) {
      return static_cast<T*>(p);
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
    TraceErrorIf(cudaSuccess != cudaFreeHost(p), "cudaFreeHost failed");
  }
};

class Stream {
public:
  Stream(bool sync_on_destroy = false, ICudaStreamProvider::StreamType type = ICudaStreamProvider::StreamType::Odom);
  ~Stream();

  cudaStream_t& get_stream();

private:
  cudaStream_t stream;
  ICudaStreamProvider::StreamType type_;
  bool sync_on_destroy_;
};

class GPUGraph {
public:
  explicit GPUGraph(cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal);
  void launch(const std::function<void(cudaStream_t s)>& lambda, cudaStream_t s);

private:
  bool is_initialized = false;
  cudaStreamCaptureMode mode_;
  cudaGraph_t graph_;
  cudaGraphExec_t instance_ = nullptr;
};

void RegisterCudaStreamProvider(ICudaStreamProvider* provider);

bool CheckCompatibility(std::string& message);

void WarmUpGpu();

bool IsGpuPointer(const void* ptr);

}  // namespace cuvslam::cuda
