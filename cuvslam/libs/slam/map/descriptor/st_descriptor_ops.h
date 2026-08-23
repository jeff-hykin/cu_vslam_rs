
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

#include "sof/st_tracker.h"

#include "slam/map/descriptor/feature_descriptor.h"

namespace cuvslam::slam {

struct STDescriptor : public IFeatureDescriptor {
  // image patches
  uint32_t levels_mask = 0;
  uint32_t image_patches_size = 0;
  sof::STFeaturePatch image_patches[];
};

// Shi-Tomasi descriptor operations

class STDescriptorOps : public IFeatureDescriptorOps {
public:
  STDescriptorOps(uint32_t n_shift_only_iterations, uint32_t n_full_mapping_iterations);
  ~STDescriptorOps() override;

  virtual float Match(const FeatureDescriptorRef fd0, const FeatureDescriptorRef fd1) const override;
  virtual bool ToBlob(const FeatureDescriptorRef fd, BlobWriter& blob) const override;
  virtual FeatureDescriptor CreateFromBlob(const BlobReader& blob_reader) const override;
  virtual FeatureDescriptor Copy(const FeatureDescriptor& source) const override;
  virtual int Compare(const FeatureDescriptorRef fd0, const FeatureDescriptorRef fd1) const override;

  // methods for process batch of tracks
  void CreateDescriptors(const ImageContextPtr& image, const std::vector<Vector2T>& input,
                         std::vector<FeatureDescriptor>& output) const override;
  void MapsToImage(const ImageContextPtr& image, const std::vector<FeatureDescriptor>& descriptors,
                   const std::vector<IFeatureDescriptorOps::Track2dInput>& input,
                   std::vector<IFeatureDescriptorOps::Track2dOutput>& output) const override;

private:
  uint32_t n_shift_only_iterations;
  uint32_t n_full_mapping_iterations;
  mutable sof::STTracker st_tracker;

public:
  struct Track2dInfo {
    float ncc = 0;
    float info[4];  // information matrix
  };
  FeatureDescriptor CreateDescriptor(const sof::ImagePyramidT& image, const sof::GradientPyramidT& image_gradient,
                                     const Vector2T& UV) const;
  bool MapToImage(const FeatureDescriptorRef fd, const sof::ImagePyramidT& image,
                  const sof::GradientPyramidT& image_gradient,
                  const Vector2T* predicted_UV,  // null if no predition
                  Vector2T& UV, Track2dInfo* info = nullptr, float search_radius_px = 2048.f) const;
};

}  // namespace cuvslam::slam
