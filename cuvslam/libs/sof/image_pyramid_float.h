
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

#include <array>

#include "common/image_matrix.h"
#include "common/log.h"
#include "common/types.h"
#include "common/vector_2t.h"

namespace cuvslam::sof {

template <typename _PixelType>
Vector2N ImageDims(const ImageMatrix<_PixelType>& image) {
  return Vector2N(image.cols(), image.rows());
}

template <typename _PixelType, typename _Index>
_PixelType& Pixel(ImageMatrix<_PixelType>& image, const Vector2<_Index>& idx) {
  return image(Index(idx.y()), Index(idx.x()));
}

class ImageGaussianScaler {
public:
  bool operator()(const ImageMatrixT& inputImage, ImageMatrixT& outputImage) const;
};

template <typename _PixelType>
class ImageDummyScaler {
public:
  using ImageType = ImageMatrix<_PixelType>;
  bool operator()(const ImageType& inputImage, ImageType& outputImage) {
    (void)inputImage;
    (void)outputImage;
    return false;
  }
};

// Compute bound box patch in pixels (negative coordinates are possible)
inline void compute_patch_bbox(const Vector2T& xy, Index dim, Vector2<int32_t>& tlxy, Vector2<int32_t>& brxy) noexcept {
  // shift so center of the pixel has .0 coordinates
  const Vector2T xy0 = xy - Vector2T::Constant(0.5f);

  const Vector2<int32_t> ixy = xy0.template cast<int32_t>();  // floor
  const auto int_dim = static_cast<int32_t>(dim);

  tlxy = ixy - Vector2<int32_t>::Constant(int_dim / 2);
  brxy = tlxy + Vector2<int32_t>::Constant(int_dim + 1);
}

template <typename _PixelType, typename _ScalerType, int _MaxPyramidLevels>
class ImagePyramid {
  static_assert(_MaxPyramidLevels > 1, "Compile time _MaxPyramidLevels validation error");

public:
  using PixelType = _PixelType;
  using ScalerType = _ScalerType;
  using ImageType = ImageMatrix<PixelType>;
  using ArrayOfImages = std::array<ImageType, _MaxPyramidLevels>;

  enum { MaxPyramidLevels = _MaxPyramidLevels };

  ImagePyramid() = default;
  explicit ImagePyramid(const ImageType& image) { imageLevels_[0] = image; }

  // Build levels of image pyramid from the input image.
  // Returns number of actual levels built but no more than _MaxPyramidLevels.
  // Returns 0 in the case of error.
  int buildPyramid() {
    const ImageType& image(imageLevels_[0]);

    if (image.data() == nullptr) {
      TraceError("Empty input image");
      return 0;
    }

    ScalerType scaler;
    int level = 1;

    while (scaler(imageLevels_[level - 1], imageLevels_[level]) && ++level < _MaxPyramidLevels) {
      ;
    }

    levelCount_ = level;
    return level;
  }

  bool isPointInImage(const Vector2T& xy, const int level) const {
    const ImageType& image = imageLevels_[level];
    const Index height = image.rows();
    const Index width = image.cols();

    // - 0.5f as it will be subtracted to set origin of patch in computePatch()
    const float x = xy.x();
    const float y = xy.y();
    return x >= 0 && x < width && y >= 0 && y < height;
  }

  // Scale down pixel from base level (=0) to some level
  Vector2T ScaleDownPoint(const Vector2T& xy, int level) const {
    assert(level < getLevelsCount());

    const float scale = 1.f / static_cast<float>(1 << level);  // scale = pow(0.5, level);
    // account for the fact that the image content is scaled around (0.5 0.5)
    const Vector2T shift(0.5f, 0.5f);
    return (xy - shift) * scale + shift;
  }

  // Scale up point from level to (level - 1)
  Vector2T ScaleUpPointToNextLevel(const Vector2T& xy) const {
    // account for the fact that the image content is scaled around (0.5 0.5)
    return xy * 2.f - Vector2T(0.5f, 0.5f);
  }

  int getLevelsCount() const { return levelCount_; }

  void setLevelsCount(const int levels) {
    assert(levels <= MaxPyramidLevels);
    levelCount_ = levels;
  }

  const ImageType& operator[](int i) const {
    assert(i < levelCount_);
    return imageLevels_[i];
  }
  ImageType& operator[](int i) {
    assert(i < MaxPyramidLevels);
    return imageLevels_[i];
  }
  ImageType& base() { return imageLevels_[0]; }

  // Samples patch from the specified location (xy and level) using bilinear
  // interpolation.
  template <typename PixelType, int Dim, int Cols, int Mode, int Options>
  bool computePatch(Eigen::Matrix<PixelType, Dim, Cols, Mode, Options>& patch, const Vector2T& xy, int level) const {
    static_assert(Dim == Cols, "Should be square matrix");
    static_assert(Dim % 2 == 1, "Should be fixed odd sized square matrix");

    if (level >= levelCount_) {
      assert(false);
      return false;
    }

    const ImageType& image = imageLevels_[level];
    const Vector2<int32_t> imageSize = ImageDims(image).template cast<int32_t>();
    Vector2<int32_t> tlxy, brxy;

    compute_patch_bbox(xy, Dim, tlxy, brxy);
    if (tlxy.x() < 0 || brxy.x() >= imageSize.x() || tlxy.y() < 0 || brxy.y() >= imageSize.y()) {
      return false;  // the patch bound box is out of the image
    }

    // shift so center of the pixel has .0 coordinates
    const Vector2T xy0 = xy - Vector2T::Constant(0.5f);
    const Vector2<int32_t> ixy = xy0.template cast<int32_t>();  // floor

    // interpolation coefficients
    const Vector2T dxy = xy0 - ixy.template cast<float>();
    if (!(cuvslam::operator<=(Vector2T(0, 0), dxy) && cuvslam::operator<=(dxy, Vector2T(1, 1)))) {
      // not sure, how it can happened, but can. float acc?
      // TODO: clamp and continue
      return false;
    }

    const float k0 = (1.f - dxy.x()) * (1.f - dxy.y());
    const float k1 = dxy.x() * (1.f - dxy.y());
    const float k2 = (1.f - dxy.x()) * dxy.y();
    const float k3 = dxy.x() * dxy.y();

    const int w0 = static_cast<int>(image.cols());
    const int w1 = w0 + 1;
    const int w2 = w0 + 2;
    const int w3 = w0 + 3;
    const int w4 = w0 + 4;

    for (int y = 0; y < Dim; ++y) {
      const float* src = image.data() + tlxy.x() + (tlxy.y() + y) * w0;

      int x = 0;
      float* dst = &patch(y, x);
      for (; x < Dim - 3; x += 4) {
        dst[x] = k0 * src[0] + k1 * src[1] + k2 * src[w0] + k3 * src[w1];
        dst[x + 1] = k0 * src[1] + k1 * src[2] + k2 * src[w1] + k3 * src[w2];
        dst[x + 2] = k0 * src[2] + k1 * src[3] + k2 * src[w2] + k3 * src[w3];
        dst[x + 3] = k0 * src[3] + k1 * src[4] + k2 * src[w3] + k3 * src[w4];
        src += 4;
      }

      for (; x < Dim; ++x) {
        dst[x] = k0 * src[0] + k1 * src[1] + k2 * src[w0] + k3 * src[w1];
        ++src;
      }
    }

    return true;
  }

private:
  int levelCount_ = 0;
  ArrayOfImages imageLevels_;
};

const int MAX_PYRAMID_LEVELS = 10;
using ImagePyramidT = ImagePyramid<float, ImageGaussianScaler, MAX_PYRAMID_LEVELS>;
using ImageNoScalePyramidT = ImagePyramid<float, ImageDummyScaler<float>, MAX_PYRAMID_LEVELS>;

}  // namespace cuvslam::sof
