
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

#include "cuda_modules/image_cast.h"

#include "cuda_modules/cuda_kernels/cuda_kernels.h"

namespace cuvslam::cuda {

GPUImageT& ImageCast::operator()(void* data, const ImageEncoding& encoding, const ImageShape& image_shape,
                                 cudaStream_t s) {
  if (gpu_image_float == nullptr) {
    gpu_image_float = std::make_unique<GPUImageT>(image_shape.width, image_shape.height);
  }
  cast(data, encoding, image_shape, *gpu_image_float, s);
  return *gpu_image_float;
}

bool ImageCast::cast(void* data, const ImageEncoding& encoding, const ImageShape& image_shape, GPUImageT& gpu_image,
                     cudaStream_t s) {
  cudaError_t error = cudaGetLastError();

  if (encoding == ImageEncoding::MONO8) {
    TRACE_EVENT ev = profiler_domain_.trace_event("Cast", profiler_color_);

    if (gpu_image_uint8 == nullptr) {
      gpu_image_uint8 = std::make_unique<GPUImage8>(image_shape.width, image_shape.height);
    }
    const unsigned char* data_ptr = static_cast<const unsigned char*>(data);
    gpu_image_uint8->copy(GPUCopyDirection::ToGPU, data_ptr, s);
    uint2 size = {(unsigned)image_shape.width, (unsigned)image_shape.height};
    error = cast_image(gpu_image_uint8->ptr(), gpu_image_uint8->pitch(), gpu_image.ptr(), gpu_image.pitch(), size, s);
  } else if (encoding == ImageEncoding::RGB8) {
    TRACE_EVENT ev = profiler_domain_.trace_event("CastRGB2GrayScale", profiler_color_);

    // RGB input expands each pixel into three bytes. Keep both multiplications
    // checked so a malicious or corrupted shape cannot shrink the staging
    // allocation while the copy still uses the full image dimensions.
    const size_t rgb_pixels =
        CheckedMul(static_cast<size_t>(image_shape.width), static_cast<size_t>(image_shape.height));
    const size_t rgb_elements = CheckedMul(rgb_pixels, 3);
    if (gpu_array_rgb == nullptr) {
      gpu_array_rgb = std::make_unique<GPUOnlyArray<uint8_t>>(rgb_elements);
    }

    CUDA_CHECK(cudaMemcpyAsync((void*)gpu_array_rgb->ptr(), (void*)data, rgb_elements, cudaMemcpyHostToDevice, s));
    uint2 size = {(unsigned)image_shape.width, (unsigned)image_shape.height};
    error = cast_image_rgb(gpu_array_rgb->ptr(), size.x * 3, gpu_image.ptr(), gpu_image.pitch(), size, s);
  }

  CUDA_CHECK(error);
  return error == cudaSuccess;
}

bool ImageCast::cast_depth(uint16_t* data, float scale, const ImageShape& image_shape, GPUImageT& gpu_image,
                           cudaStream_t s) {
  cudaError_t error = cudaGetLastError();

  TRACE_EVENT ev = profiler_domain_.trace_event("Cast depth", profiler_color_);
  if (gpu_image_uint16 == nullptr) {
    gpu_image_uint16 = std::make_unique<GPUImage16>(image_shape.width, image_shape.height);
  }
  gpu_image_uint16->copy(GPUCopyDirection::ToGPU, data, s);
  uint2 size = {(unsigned)image_shape.width, (unsigned)image_shape.height};
  error = cast_depth_u16(gpu_image_uint16->ptr(), gpu_image_uint16->pitch(), scale, gpu_image.ptr(), gpu_image.pitch(),
                         size, s);

  CUDA_CHECK(error);
  return error == cudaSuccess;
}

bool ImageCast::burn_mask_depth(uint8_t* cpu_mask, const ImageShape& image_shape, GPUImageT& gpu_depth,
                                cudaStream_t s) {
  cudaError_t error = cudaGetLastError();

  TRACE_EVENT ev = profiler_domain_.trace_event("burn_mask_depth", profiler_color_);
  if (gpu_image_uint8 == nullptr) {
    gpu_image_uint8 = std::make_unique<GPUImage8>(image_shape.width, image_shape.height);
  }
  gpu_image_uint8->copy(GPUCopyDirection::ToGPU, cpu_mask, s);
  uint2 size = {(unsigned)image_shape.width, (unsigned)image_shape.height};
  error =
      burn_depth_mask(gpu_depth.ptr(), gpu_depth.pitch(), gpu_image_uint8->ptr(), gpu_image_uint8->pitch(), size, s);

  CUDA_CHECK(error);
  return error == cudaSuccess;
}

}  // namespace cuvslam::cuda
