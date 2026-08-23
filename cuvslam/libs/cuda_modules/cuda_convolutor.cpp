
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

#include "cuda_modules/cuda_convolutor.h"

#include "common/log.h"

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

CudaConvolutor::CudaConvolutor() { CUDA_CHECK(init_conv_kernels()); }

cudaError_t CudaConvolutor::convKernelGradDerivX(const GPUImageT& in, GPUImageT& out, cudaStream_t& stream) {
  const unsigned int& h = in.rows();
  const unsigned int& w = in.cols();

  cudaError_t error = conv_grad_x(in.get_texture_filter_point(), {w, h}, out.ptr(), out.pitch(), stream);

  return error;
}

cudaError_t CudaConvolutor::convKernelGradDerivY(const GPUImageT& in, GPUImageT& out, cudaStream_t& stream) {
  const unsigned int& h = in.rows();
  const unsigned int& w = in.cols();

  cudaError_t error = conv_grad_y(in.get_texture_filter_point(), {w, h}, out.ptr(), out.pitch(), stream);

  return error;
}

}  // namespace cuvslam::cuda
