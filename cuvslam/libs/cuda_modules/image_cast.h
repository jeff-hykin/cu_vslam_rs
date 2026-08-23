
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

#include "common/image.h"
#include "common/types.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "cuda_modules/cuda_helper.h"

namespace cuvslam::cuda {

class ImageCast {
public:
  GPUImageT& operator()(void* data, const ImageEncoding& encoding, const ImageShape& image_shape, cudaStream_t s);

  bool cast(void* data, const ImageEncoding& encoding, const ImageShape& image_shape, GPUImageT& gpu_image,
            cudaStream_t s);
  bool cast_depth(uint16_t* data, float scale, const ImageShape& image_shape, GPUImageT& gpu_image, cudaStream_t s);
  bool burn_mask_depth(uint8_t* cpu_mask, const ImageShape& image_shape, GPUImageT& gpu_depth, cudaStream_t s);

private:
  std::unique_ptr<GPUImageT> gpu_image_float = nullptr;
  std::unique_ptr<GPUImage8> gpu_image_uint8 = nullptr;
  std::unique_ptr<GPUImage16> gpu_image_uint16 = nullptr;
  std::unique_ptr<GPUOnlyArray<uint8_t> > gpu_array_rgb = nullptr;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0xFF0000;
};

}  // namespace cuvslam::cuda
