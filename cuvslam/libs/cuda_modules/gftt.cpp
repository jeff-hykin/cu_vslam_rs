
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

#include "cuda_modules/gftt.h"

#include <assert.h>

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

GFTT::GFTT() { CUDA_CHECK(init_gauss_square_kernel()); }

cudaError_t GFTT::compute(const GPUImageT& gradX, const GPUImageT& gradY, GPUImageT& value, cudaStream_t& stream) {
  assert(gradX.cols() == gradY.cols());
  assert(gradX.cols() == value.cols());
  assert(gradX.rows() == gradY.rows());
  assert(gradX.rows() == value.rows());

  cudaError_t error =
      gftt_values(gradX.get_texture_filter_point(), gradY.get_texture_filter_point(), value.ptr(), value.pitch(),
                  {static_cast<unsigned int>(gradX.cols()), static_cast<unsigned int>(gradX.rows())}, stream);

  CUDA_CHECK(error);

  return error;
}

}  // namespace cuvslam::cuda
