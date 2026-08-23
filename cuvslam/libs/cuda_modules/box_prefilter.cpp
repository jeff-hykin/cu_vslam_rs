
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

#include "cuda_modules/box_prefilter.h"

#include "common/log.h"

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

CudaBoxPrefilter::CudaBoxPrefilter() { CUDA_CHECK(init_box_prefilter_kernels()); }

void CudaBoxPrefilter::prefilter(const GPUImageT &in, GPUImageT &out, cudaStream_t &stream) {
  uint2 src_size = {static_cast<unsigned int>(in.cols()), static_cast<unsigned int>(in.rows())};

  if (!buffer_ || buffer_->rows() < src_size.y || buffer_->cols() < src_size.x) {
    buffer_ = std::make_unique<GPUImageT>(src_size.x, src_size.y);
  }

  CUDA_CHECK(box_blur_x(in.get_texture_filter_point(), src_size, buffer_->ptr(), buffer_->pitch(), stream));

  CUDA_CHECK(box_blur_y(buffer_->get_texture_filter_point(), src_size, out.ptr(), out.pitch(), stream));

  CUDA_CHECK(copy_border(in.get_texture_filter_point(), src_size, out.ptr(), out.pitch(), 1, stream));
}

}  // namespace cuvslam::cuda
