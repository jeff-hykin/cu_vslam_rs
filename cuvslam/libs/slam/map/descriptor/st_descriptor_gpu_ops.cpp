
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

#include <unordered_map>

#include "slam/common/slam_common.h"
#include "slam/map/descriptor/st_descriptor_gpu_ops.h"
#include "slam/map/descriptor/st_descriptor_merge_details.h"

using namespace cuvslam;
using namespace cuvslam::slam;
using namespace cuvslam::slam::merge_details;

#ifdef USE_CUDA
using namespace cuvslam::cuda;
STDescriptorGpuOps::STDescriptorGpuOps(uint32_t n_shift_only_iterations, uint32_t n_full_mapping_iterations)
    : STDescriptorOps(n_shift_only_iterations, n_full_mapping_iterations) {
  st_tracker_ = std::make_unique<GPUSTFeatureTrackerWithCache>(n_shift_only_iterations, n_full_mapping_iterations);
}
STDescriptorGpuOps::~STDescriptorGpuOps() {}

void FillCacheFromDescriptor(const STDescriptor& descriptor, PointCacheDataT& cache) {
  assert(descriptor.image_patches_size <= PYRAMID_LEVELS);
  for (size_t i = 0; i < descriptor.image_patches_size; i++) {
    memcpy(&cache.patch[PATCH_DIM * PATCH_DIM * i], descriptor.image_patches[i].data(),
           PATCH_DIM * PATCH_DIM * sizeof(float));

    cache.patch_sums[i] = descriptor.image_patches[i].sum();
  }
  cache.level_mask = descriptor.levels_mask;
}

std::shared_ptr<IFeatureDescriptor> DescriptorFromCache(const PointCacheDataT& cache) {
  size_t num_levels = 0;
  size_t temp = cache.level_mask;
  while (temp != 0) {
    temp >>= 1;
    num_levels++;
  }
  assert(num_levels <= PYRAMID_LEVELS);

  const size_t size_bytes = sizeof(STDescriptor) + sizeof(sof::STFeaturePatch) * num_levels;
  auto shared_ptr = std::shared_ptr<STDescriptor>(reinterpret_cast<STDescriptor*>(new char[size_bytes]), [](void* ptr) {
    if (ptr) {
      delete[] static_cast<char*>(ptr);
    }
  });
  STDescriptor* dst = shared_ptr.get();

  for (size_t i = 0; i < num_levels; i++) {
    memcpy(dst->image_patches[i].data(), &cache.patch[PATCH_DIM * PATCH_DIM * i],
           PATCH_DIM * PATCH_DIM * sizeof(float));
  }
  dst->image_patches_size = num_levels;
  dst->levels_mask = cache.level_mask;

  return shared_ptr;
}

// methods for process batch of tracks
void STDescriptorGpuOps::CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                                           std::vector<FeatureDescriptor>& output) const {
  output.clear();
  output.resize(input.size());

  assert(input.size() <= st_max_points);

  size_t num_tracks = input.size();
  for (size_t k = 0; k < num_tracks; ++k) {
    TrackData& data = track_data_[k];
    const Vector2T& position = input[k];

    data.track = {position.x(), position.y()};
    data.cache_index = -1;
  }

  track_data_.copy_top_n(GPUCopyDirection::ToGPU, num_tracks, stream_.get_stream());
  bool res = st_tracker_->build_points_cache(image->gpu_image_pyramid(), track_data_, cache_data_, num_tracks,
                                             stream_.get_stream());
  assert(res);
  if (!res) {
    TraceError("build_points_cache failed!");
  }
  // Workaround for blocking memcpy of pageable memory; now it will block only for a time of copy itself
  cudaStreamSynchronize(stream_.get_stream());
  cache_data_.copy_top_n(GPUCopyDirection::ToCPU, num_tracks, stream_.get_stream());
  track_data_.copy_top_n(GPUCopyDirection::ToCPU, num_tracks, stream_.get_stream());
  cudaStreamSynchronize(stream_.get_stream());

  for (size_t k = 0; k < num_tracks; ++k) {
    TrackData& data = track_data_[k];

    if (data.cache_index != -1) {
      output[k].pull_memory = DescriptorFromCache(cache_data_[k]);
    } else {
      output[k] = FeatureDescriptorEmpty;
    }
  }
}

void STDescriptorGpuOps::MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descriptors,
                                     const std::vector<IFeatureDescriptorOps::Track2dInput>& input,
                                     std::vector<IFeatureDescriptorOps::Track2dOutput>& output) const {
  if (input.empty()) {
    return;
  }
  // Input image here: image.source_image

  output.clear();

  size_t num_tracks = std::min(static_cast<size_t>(st_max_points), input.size());

  for (size_t k = 0; k < num_tracks; ++k) {
    TrackData& data = track_data_[k];
    data.cache_index = -1;
    data.ncc_threshold = 0.f;

    const IFeatureDescriptorOps::Track2dInput& track_input = input[k];
    auto st_fd = reinterpret_cast<const STDescriptor*>(descriptors[track_input.fd_index].get());
    if (st_fd) {
      FillCacheFromDescriptor(*st_fd, cache_data_[k]);
      data.cache_index = k;
    } else {
      SlamStderr("No valid Shi-Tomasi descriptor found.\n");
      return;
    }
  }

  for (size_t k = 0; k < num_tracks; ++k) {
    const IFeatureDescriptorOps::Track2dInput& track_input = input[k];

    Vector2T previous_uv(0, 0);
    if (track_input.has_predict_uv) {
      previous_uv = track_input.predict_uv;
    }

    TrackData& data = track_data_[k];

    data.track = {previous_uv.x(), previous_uv.y()};
    data.offset = {0, 0};
    data.track_status = false;
    data.search_radius_px = track_input.search_radius_px;
  }

  track_data_.copy_top_n(GPUCopyDirection::ToGPU, num_tracks, stream_.get_stream());
  cache_data_.copy_top_n(GPUCopyDirection::ToGPU, num_tracks, stream_.get_stream());
  bool res = st_tracker_->track_points_with_cache(image->gpu_gradient_pyramid(), image->gpu_image_pyramid(),
                                                  track_data_, cache_data_, num_tracks, stream_.get_stream());
  if (!res) {
    SlamStderr("Failed to track using Shi-Tomasi with cache on GPU.\n");
    return;
  }
  // Workaround for blocking memcpy of pageable memory; now it will block only for a time of copy itself
  cudaStreamSynchronize(stream_.get_stream());
  track_data_.copy_top_n(GPUCopyDirection::ToCPU, num_tracks, stream_.get_stream());
  cudaStreamSynchronize(stream_.get_stream());

  output.resize(input.size());

  for (auto& x : output) {
    x.successed = false;
  }

  for (size_t id = 0; id < num_tracks; id++) {
    IFeatureDescriptorOps::Track2dOutput& track_out = output[id];

    TrackData& data = track_data_[id];
    if (data.track_status) {
      track_out.successed = true;
      track_out.ncc = data.ncc;
      track_out.info[0] = data.info[0];
      track_out.info[1] = data.info[1];
      track_out.info[2] = data.info[2];
      track_out.info[3] = data.info[3];
      track_out.uv = {data.track.x, data.track.y};
    }
  }
}
#else
STDescriptorGpuOps::STDescriptorGpuOps(uint32_t n_shift_only_iterations, uint32_t n_full_mapping_iterations)
    : STDescriptorOps(n_shift_only_iterations, n_full_mapping_iterations) {}
STDescriptorGpuOps::~STDescriptorGpuOps() {}

// methods for process batch of tracks
void STDescriptorGpuOps::CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                                           std::vector<FeatureDescriptor>& output) const {
  (void)image;
  (void)input;
  (void)output;
}

void STDescriptorGpuOps::MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descroptors,
                                     const std::vector<IFeatureDescriptorOps::Track2dInput>& input,
                                     std::vector<IFeatureDescriptorOps::Track2dOutput>& output) const {
  (void)image;
  (void)descroptors;
  (void)input;
  (void)output;
}
#endif
