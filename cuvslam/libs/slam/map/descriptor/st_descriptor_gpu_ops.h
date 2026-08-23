
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

#ifdef USE_CUDA
#include "cuda_modules/cuda_helper.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/image_cast.h"
#include "cuda_modules/st_tracker.h"
#endif

#include "slam/map/descriptor/st_descriptor_ops.h"

namespace cuvslam::slam {

// Shi-Tomasi descriptor operations

static constexpr int st_max_points = 2000;  // It's enough for LC detection

class STDescriptorGpuOps : public STDescriptorOps {
public:
  STDescriptorGpuOps(uint32_t n_shift_only_iterations, uint32_t n_full_mapping_iterations);
  ~STDescriptorGpuOps() override;

public:
  // methods for process batch of tracks
  void CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                         std::vector<FeatureDescriptor>& output) const override;
  void MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descriptors,
                   const std::vector<IFeatureDescriptorOps::Track2dInput>& input,
                   std::vector<IFeatureDescriptorOps::Track2dOutput>& output) const override;

#ifdef USE_CUDA
private:
  std::unique_ptr<cuda::GPUSTFeatureTrackerWithCache> st_tracker_;
  mutable cuda::GPUArray<cuda::TrackData> track_data_{st_max_points};
  mutable cuda::GPUArray<cuda::PointCacheDataT> cache_data_{st_max_points};

  mutable cuda::Stream stream_;
#endif
};

}  // namespace cuvslam::slam
