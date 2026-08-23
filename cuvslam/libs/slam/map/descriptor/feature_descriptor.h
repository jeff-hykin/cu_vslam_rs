
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

#include <stdint.h>

#include <memory>

#include "common/image.h"
#include "common/vector_2t.h"

#include "slam/common/blob.h"
#include "slam/common/slam_input_image.h"

namespace cuvslam::slam {

// IFeatureDescriptor, FeatureDescriptor, FeatureDescriptorRef
class IFeatureDescriptor {};
struct FeatureDescriptorStruct {
  const IFeatureDescriptor* const_memory_ptr = nullptr;
  std::shared_ptr<IFeatureDescriptor> pull_memory;

  operator bool() const { return const_memory_ptr || pull_memory; }
  bool operator!() const { return !const_memory_ptr && !pull_memory; }
  bool is_const_memory() const { return const_memory_ptr; }
  const IFeatureDescriptor* get() const {
    if (const_memory_ptr) {
      return const_memory_ptr;
    }
    return pull_memory.get();
  };
};
using FeatureDescriptor = FeatureDescriptorStruct;
using FeatureDescriptorRef = const FeatureDescriptorStruct&;
const FeatureDescriptorStruct FeatureDescriptorEmpty = FeatureDescriptorStruct({nullptr, nullptr});

// interface for operations over descriptors
class IFeatureDescriptorOps {
public:
  virtual ~IFeatureDescriptorOps() = default;
  virtual float Match(const FeatureDescriptorRef fd0, const FeatureDescriptorRef fd1) const = 0;
  virtual bool ToBlob(const FeatureDescriptorRef fd, BlobWriter& blob) const = 0;
  virtual FeatureDescriptor CreateFromBlob(const BlobReader& blob) const = 0;
  virtual FeatureDescriptor Copy(const FeatureDescriptor& source) const = 0;

  virtual int Compare(const FeatureDescriptorRef fd0, const FeatureDescriptorRef fd1) const = 0;

  // methods for process batch of tracks
  virtual void CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                                 std::vector<FeatureDescriptor>& output) const = 0;

  struct Track2dInput {
    bool has_predict_uv;
    Vector2T predict_uv;
    float search_radius_px;
    uint32_t fd_index;
    Track2dInput(uint32_t fd_index) : has_predict_uv(false), search_radius_px(2048.f), fd_index(fd_index){};
  };
  struct Track2dOutput {
    bool successed;
    Vector2T uv;
    float ncc = 0;
    float info[4];  // information matrix
  };
  virtual void MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descriptors,
                           const std::vector<Track2dInput>& input, std::vector<Track2dOutput>& output) const = 0;
};

}  // namespace cuvslam::slam
