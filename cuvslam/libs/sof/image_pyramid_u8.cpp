
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

#include "sof/image_pyramid_u8.h"

#include "sof/box_blur.h"

namespace cuvslam::sof {

void ImagePyramidU8::build(const ImageSource& image, const ImageShape& shape, bool blur_filter) {
  clear();
  add_level(shape.width, shape.height);

  const ImageMatrix<uint8_t>& input_matrix = image.as<uint8_t>(shape);
  ImageSource& base_level_image = levels_[0].first;
  ImageShape& base_level_shape = levels_[0].second;

  // allocate memory and copy
  base_level_image.as<uint8_t>(base_level_shape) = input_matrix;

  if (blur_filter) {
    BoxBlur3(input_matrix, static_cast<uint8_t*>(base_level_image.data));
  }

  int w = shape.width;
  int h = shape.height;

  scaler_.compute_new_size(w, h);

  constexpr int kMinSize = 15;
  while ((w >= kMinSize) && (h >= kMinSize)) {
    add_level(w, h);

    ImageSource& dst_source = levels_[num_levels_ - 1].first;
    ImageShape& dst_shape = levels_[num_levels_ - 1].second;

    const ImageSource& src_source = levels_[num_levels_ - 2].first;
    const ImageShape& src_shape = levels_[num_levels_ - 2].second;

    scaler_.scale(src_source, src_shape, dst_source, dst_shape);
    scaler_.compute_new_size(w, h);
  }
}

int ImagePyramidU8::num_levels() const { return num_levels_; }

const ImagePyramidU8::ImageSourceAndShape& ImagePyramidU8::operator[](int i) const { return levels_[i]; }

void ImagePyramidU8::clear() { num_levels_ = 0; }

void ImagePyramidU8::add_level(int width, int height) {
  assert(num_levels_ < MaxLevels);
  assert(0 < width);
  assert(0 < height);

  auto size = width * height * sizeof(uint8_t);
  pool_[num_levels_].resize(size);

  ImageSource& image = levels_[num_levels_].first;
  ImageShape& shape = levels_[num_levels_].second;

  image.data = pool_[num_levels_].data();
  shape.width = width;
  shape.height = height;
  image.type = ImageSource::TypeAsEnum<uint8_t>::value;
  ++num_levels_;
}

}  // namespace cuvslam::sof
