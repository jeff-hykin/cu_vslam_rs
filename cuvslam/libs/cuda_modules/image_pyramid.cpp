
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

#include "cuda_modules/image_pyramid.h"

#include <iostream>

namespace cuvslam::cuda {

size_t ScaleDownDim(size_t dim) { return (dim + 1) / 2; }

bool ImageGaussianScaler::operator()(const GPUImageT& inputImage, GPUImageT& outputImage, cudaStream_t& stream) {
  if (!initialized) {
    init_scaler();
  }
  cudaError_t error =
      gaussian_scaling(inputImage.get_texture_filter_point(),
                       {static_cast<unsigned>(inputImage.cols()), static_cast<unsigned>(inputImage.rows())},
                       outputImage.ptr(), outputImage.pitch(),
                       {static_cast<unsigned>(outputImage.cols()), static_cast<unsigned>(outputImage.rows())}, stream);

  CUDA_CHECK(error);

  return (error == cudaSuccess);
}

}  // namespace cuvslam::cuda
