
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

#include <vector>

#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "sof/gftt.h"
#include "sof/gradient_pyramid.h"
#include "sof/sof.h"

namespace cuvslam::sof {

// The reason for this class is only eliminate memory reallocation.
// It doesn't store any internal states.
// Each call of selectFeature is independent.
class GoodFeaturesToTrackDetector {
public:
  GoodFeaturesToTrackDetector();
  ~GoodFeaturesToTrackDetector();

  // burnHalfSize and alreadySelectedBurnHalfSize should depend on image width (i.e. aperture in pixels)
  void computeGFTTAndSelectFeatures(const GradientPyramidT& gradientPyramid, int border_top, int border_bottom,
                                    int border_left, int border_right, const ImageMatrix<uint8_t>* input_mask,
                                    const TracksVector& alreadySelectedFeatures, size_t desiredNewFeaturesCount,
                                    std::vector<Vector2T>& newSelectedFeatures, size_t nBinX = 8, size_t nBinY = 8,
                                    size_t burnHalfSize = 4, size_t alreadySelectedBurnHalfSize = 4);

  void selectFeatures(const ImageMatrixT& imageGFTT, const ImageMatrix<uint8_t>* input_mask, int border_top,
                      int border_bottom, int border_left, int border_right, const TracksVector& alreadySelectedFeatures,
                      size_t desiredNewFeaturesCount, std::vector<Vector2T>& newSelectedFeatures, size_t nBinX = 8,
                      size_t nBinY = 8, size_t burnHalfSize = 4, size_t alreadySelectedBurnHalfSize = 4);

  // for internal use
  struct Bin;

private:
  ImageMatrix<uint8_t> mask_;
  std::vector<Bin> bins_;
  GFTT gftt_;

  static void cutBinFromGFFT(const Vector2N& start, size_t binW, size_t binH, Bin& bin, const ImageMatrixT& strength);

  void splitGFFTToBins(size_t nRows, size_t nCols, const ImageMatrixT& strength);

  float calculateSummGFTT() const noexcept;
  void mask(size_t x, size_t y, size_t half) noexcept;
  void mask_border(int border_top, int border_bottom, int border_left, int border_right) noexcept;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0xFF0000;
};

}  // namespace cuvslam::sof
