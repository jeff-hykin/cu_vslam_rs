
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

#include "sof/image_pyramid_float.h"

namespace {
using namespace cuvslam;

void SmoothImage(const ImageMatrixT &inImage, ImageMatrixT &smoothedImage) {
  ImageMatrixT inExtImage;
  static const Matrix5T kernel = (Matrix5T() << 1.f, 4.f, 6.f, 4.f, 1.f, 4.f, 16.f, 24.f, 16.f, 4.f, 6.f, 24.f, 36.f,
                                  24.f, 6.f, 4.f, 16.f, 24.f, 16.f, 4.f, 1.f, 4.f, 6.f, 4.f, 1.f)
                                     .finished() /
                                 256.f;
  assert(kernel.sum() == 1.f);

  const Index h = inImage.rows();
  const Index w = inImage.cols();

  inExtImage.resize(h + 4, w + 4);
  inExtImage.block(2, 2, h, w) = inImage;  // copy pixels
  // mirror borders
  const Index lastColumn = w + 4 - 1;
  const Index lastRow = h + 4 - 1;
  inExtImage.col(0) = inExtImage.col(3);
  inExtImage.col(1) = inExtImage.col(2);
  inExtImage.col(lastColumn) = inExtImage.col(lastColumn - 3);
  inExtImage.col(lastColumn - 1) = inExtImage.col(lastColumn - 2);
  inExtImage.row(0) = inExtImage.row(3);
  inExtImage.row(1) = inExtImage.row(2);
  inExtImage.row(lastRow) = inExtImage.row(lastRow - 3);
  inExtImage.row(lastRow - 1) = inExtImage.row(lastRow - 2);

  inExtImage.topLeftCorner(2, 2) = Matrix2T::Zero();
  inExtImage.bottomLeftCorner(2, 2) = Matrix2T::Zero();
  inExtImage.topRightCorner(2, 2) = Matrix2T::Zero();
  inExtImage.bottomRightCorner(2, 2) = Matrix2T::Zero();
  smoothedImage.resize(h, w);

  for (Index y = 0; y < h; ++y) {
    for (Index x = 0; x < w; ++x) {
      smoothedImage(y, x) = inExtImage.block(y, x, 5, 5).cwiseProduct(kernel).sum();
    }
  }
}

Index ScaleDownDim(const Index dim) { return (dim + 1) / 2; }

bool PrepOutput(const ImageMatrixT &inputImage, ImageMatrixT &outputImage) {
  const float MinImageDimSize = 15;

  if (inputImage.data() == nullptr) {
    TraceError("Empty input image");
    return false;
  }

  const Index inHeight = inputImage.rows();
  const Index inWidth = inputImage.cols();
  const Index outHeight = ScaleDownDim(inHeight);
  const Index outWidth = ScaleDownDim(inWidth);

  assert(outHeight >= 0 && outWidth >= 0);

  if (size_t(outHeight) < MinImageDimSize || size_t(outWidth) < MinImageDimSize) {
    TraceMessage(
        "Input image is too small, it will produce output"
        " image less than MinImageDimSize");
    return false;
  }

  // make sure output is properly sized
  outputImage.resize(outHeight, outWidth);

  if (outputImage.data() == nullptr) {
    TraceError("Failed to resize output image");
    return false;
  }

  return true;
}
}  // namespace

namespace cuvslam::sof {

bool ImageGaussianScaler::operator()(const ImageMatrixT &inputImage, ImageMatrixT &outputImage) const {
  if (!PrepOutput(inputImage, outputImage)) {
    return false;
  }

  ImageMatrixT smoothedImage;
  SmoothImage(inputImage, smoothedImage);

#ifndef NDEBUG
  const Vector2N inDims = ImageDims(smoothedImage);
#endif
  const Vector2N outDims = ImageDims(outputImage);

  for (Index outRowIndex = 0; outRowIndex < outDims.y(); outRowIndex++) {
    const Index inRowIndex0 = outRowIndex * 2;
    assert(inRowIndex0 < inDims.y());

    const auto &inputRow0 = smoothedImage.row(inRowIndex0);
    auto outputRow = outputImage.row(outRowIndex);

    for (Index outColIndex = 0; outColIndex < outDims.x(); outColIndex++) {
      const Index inColIndex0 = outColIndex * 2;
      assert(inColIndex0 < inDims.x());

      outputRow[outColIndex] = inputRow0[inColIndex0];
      assert(std::isfinite(outputRow[outColIndex]));
    }
  }

  return true;
}

}  // namespace cuvslam::sof
