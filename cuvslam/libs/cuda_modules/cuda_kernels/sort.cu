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

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

#if CUDART_VERSION >= 11060
#include <cub/cub.cuh>
#else
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace cuvslam::cuda {

struct KeypointGreater {
  __host__ __device__ bool operator()(const Keypoint& lhs, const Keypoint& rhs) { return lhs.strength > rhs.strength; }
};

void sort_keypoints(Keypoint* kp, size_t size, void* temp_buffer, size_t temp_buffer_size, cudaStream_t s) {
#if CUDART_VERSION >= 11060
  cub::DeviceMergeSort::SortKeys(temp_buffer, temp_buffer_size, kp, size, KeypointGreater(), s);
#else
  thrust::device_ptr<Keypoint> t_devPtr(kp);
  thrust::stable_sort(t_devPtr, t_devPtr + size, KeypointGreater());
#endif
}

size_t sort_keypoints_get_temp_buffer_size(size_t size) {
#if CUDART_VERSION >= 11060
  std::size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, (Keypoint*)nullptr, size, KeypointGreater());
  return temp_storage_bytes;
#else
  return 1;
#endif
}

}  // namespace cuvslam::cuda
