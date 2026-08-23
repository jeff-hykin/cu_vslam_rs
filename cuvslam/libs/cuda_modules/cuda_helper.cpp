
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

#include "cuda_modules/cuda_helper.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_modules/culib_helper.h"

#include "cuda_modules/cuda_graph_helper.h"

namespace cuvslam::cuda {

class DefaultStreamProvider : public ICudaStreamProvider {
public:
  virtual ~DefaultStreamProvider() = default;

  cudaStream_t acquire(ICudaStreamProvider::StreamType /*type*/) final {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return stream;
  }

  void release(cudaStream_t stream, ICudaStreamProvider::StreamType /*type*/) final {
    CUDA_CHECK_NOTHROW(cudaStreamDestroy(stream));
  }
} default_cuda_stream_provider;

namespace {
ICudaStreamProvider* cuda_stream_provider = &default_cuda_stream_provider;
}  // namespace

Stream::Stream(bool sync_on_destroy, ICudaStreamProvider::StreamType type)
    : type_(type), sync_on_destroy_(sync_on_destroy) {
  stream = cuda_stream_provider->acquire(type);
}

Stream::~Stream() {
  if (sync_on_destroy_) {
    CUDA_CHECK_NOTHROW(cudaStreamSynchronize(stream));
  }
  cuda_stream_provider->release(stream, type_);
}

cudaStream_t& Stream::get_stream() { return stream; }

GPUGraph::GPUGraph(cudaStreamCaptureMode mode) : mode_(mode) {}

void GPUGraph::launch(const std::function<void(cudaStream_t s)>& lambda, cudaStream_t s) {
  if (!is_initialized) {
    CUDA_CHECK(cudaStreamBeginCapture(s, mode_));
    lambda(s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, 0));
    is_initialized = true;
  } else {
    cudaGraphExecUpdateResultInfo updateResultInfo;
    CUDA_CHECK(cudaStreamBeginCapture(s, mode_));
    lambda(s);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph_));

    CUDA_CHECK(cudaGraphExecUpdate(instance_, graph_, &updateResultInfo));
    assert(updateResultInfo.result == cudaGraphExecUpdateSuccess);
  }
  CUDA_CHECK(cudaGraphLaunch(instance_, s));
}

void RegisterCudaStreamProvider(ICudaStreamProvider* provider) {
  if (provider == nullptr) {
    cuda_stream_provider = &default_cuda_stream_provider;
  } else {
    cuda_stream_provider = provider;
  }
}

bool CheckCompatibility(std::string& message) {
#ifdef USE_CUDA
  int latest_CUDA_version_supported_by_driver;
  cudaError error = cudaDriverGetVersion(&latest_CUDA_version_supported_by_driver);
  if (error != cudaSuccess) {
    message = std::string{"Failed to get CUDA version: "} + cudaGetErrorString(error);
    return false;
  }
  if (latest_CUDA_version_supported_by_driver < CUDART_VERSION) {
    auto version_to_string = [](int version) {
      return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
    };
    message = "Your NVIDIA driver supports CUDA " + version_to_string(latest_CUDA_version_supported_by_driver) +
              ", but cuVSLAM requires at least CUDA " + version_to_string(CUDART_VERSION) +
              ". Please update the driver to the most recent version.";
    return false;
  }
#endif
  return true;
}

void WarmUpGpu() {
#ifdef USE_CUDA
  cudaFree(0);
  cusolverDnHandle_t handle = nullptr;
  cublasHandle_t cublasHandle = nullptr;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUBLAS_CHECK(cublasCreate(&cublasHandle));

  if (handle) {
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
  }
  if (cublasHandle) {
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
  }
#endif
}

bool IsGpuPointer(const void* ptr) {
  cudaPointerAttributes attributes;
  return cudaSuccess == cudaPointerGetAttributes(&attributes, ptr) &&
         (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged);
}

}  // namespace cuvslam::cuda
