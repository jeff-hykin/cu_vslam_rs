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

#include <cuda_runtime_api.h>

#include <cuvslam/cuvslam2.h>

namespace cuvslam {

/**
 * Interface to provide a custom cudaStream_t provider
 */
class ICudaStreamProvider {
public:
  enum class StreamType {
    Odom,
    Slam,
  };
  virtual ~ICudaStreamProvider() = default;
  virtual cudaStream_t acquire(StreamType) = 0;
  virtual void release(cudaStream_t, StreamType) = 0;
};

/**
 * Register the cuda stream provider. It must implement the class above, and it should stay alive
 * as long as the cuvslam API is in use.
 * You can reset to the default provider by passing a nullptr.
 *
 * @warning INITIALIZATION REQUIREMENT: This function must be called during application
 *          initialization, before creating any Odometry or Slam instances. The function
 *          is not thread-safe and should be called only once in a single-threaded context.
 *
 * @note Calling this function after Odometry or Slam instances have been created may
 *       result in undefined behavior.
 */

CUVSLAM_API
void RegisterCudaStreamProvider(ICudaStreamProvider* provider);

}  // namespace cuvslam
