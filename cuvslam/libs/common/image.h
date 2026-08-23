
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

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "common/camera_id.h"
#include "common/frame_id.h"
#include "common/image_matrix.h"
#include "common/vector_2t.h"

namespace cuvslam {

enum ImageEncoding { MONO8, RGB8 };

struct ImageShape {
  int width = 0;
  int height = 0;
};

struct ImageMeta {
  ImageShape shape;
  ImageShape mask_shape;
  // optional image annotation
  FrameId frame_id = 0;
  int64_t timestamp = 0;  // ns
  int frame_number = 0;   // number from image filename
  int camera_index = 0;   // number of the camera 0, 1, 2, etc.
  std::string filename;
  std::string filename_mask;

  std::optional<float> pixel_scale_factor;  // needed for depth
};

// This structure is just an image description. It doesn't own image memory.
struct ImageSource {
  enum Type { U8, F32, U16 };
  enum MemType { Host, Device };

  Type type = U8;
  MemType memory_type = Host;
  void* data = nullptr;  // somebody else owns this memory
  int pitch = 0;         // must be valid for GPU images, ignored for CPU images
  ImageEncoding image_encoding = ImageEncoding::MONO8;

  template <class T>
  struct TypeAsEnum;
  template <Type T>
  struct EnumAsType;

  template <class T>
  Eigen::Map<const ImageMatrix<T>> as(const ImageShape& shape) const {
    assert(TypeAsEnum<T>::value == type);
    return Eigen::Map<const ImageMatrix<T>>(static_cast<const T*>(data), shape.height, shape.width);
  }

  template <class T>
  Eigen::Map<ImageMatrix<T>> as(const ImageShape& shape) {
    assert(TypeAsEnum<T>::value == type);
    return Eigen::Map<ImageMatrix<T>>(static_cast<T*>(data), shape.height, shape.width);
  }
};

template <>
struct ImageSource::TypeAsEnum<float> {
  static constexpr auto value = ImageSource::F32;
};
template <>
struct ImageSource::TypeAsEnum<uint8_t> {
  static constexpr auto value = ImageSource::U8;
};
template <>
struct ImageSource::TypeAsEnum<uint16_t> {
  static constexpr auto value = ImageSource::U16;
};

template <>
struct ImageSource::EnumAsType<ImageSource::U8> {
  using type = uint8_t;
};
template <>
struct ImageSource::EnumAsType<ImageSource::F32> {
  using type = float;
};
template <>
struct ImageSource::EnumAsType<ImageSource::U16> {
  using type = uint16_t;
};

// Camera-indexed vectors are sized to rig.num_cameras before crossing internal APIs.
// Missing RGB/depth/mask data is represented by ImageSource::data == nullptr.
using Sources = std::vector<ImageSource>;

// Metadata slots are meaningful only when the matching image source/context slot is present.
using Metas = std::vector<ImageMeta>;

// Missing per-frame depth is represented by ImageSource::data == nullptr.
using DepthSources = std::vector<ImageSource>;
}  // namespace cuvslam
