
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

#include <memory>
#include <vector>

#include "common/image_matrix.h"

#include "sof/convolutor.h"
#include "sof/image_pyramid_float.h"

namespace cuvslam::sof {

class GradientPyramidT {
public:
  GradientPyramidT();
  GradientPyramidT(const GradientPyramidT&, const ImagePyramidT*);

  int getLevelsCount() const;

  void setNumLevels(int numLevels);
  bool isGradientsLazyEvaluated(int level) const noexcept;
  ImageNoScalePyramidT& gradX();
  const ImageNoScalePyramidT& gradX() const;
  ImageNoScalePyramidT& gradY();
  const ImageNoScalePyramidT& gradY() const;

  bool set(const ImagePyramidT& image, bool horizontal = false);

  void forceNonLazyEvaluation(int level) const;

  void calcPatchGradients(const Vector2T& xy, int dim, int level) const noexcept;

  // get fixed square Eigen Matrix representing a patch around x, y point
  template <int Dim, typename PixelType>
  bool getGradXYPatches(ImageMatrixPatch<PixelType, Dim, Dim>& patchX, ImageMatrixPatch<PixelType, Dim, Dim>& patchY,
                        const Vector2T& xy, int level) const {
    calcPatchGradients(xy, Dim, level);
    return gradX_.computePatch(patchX, xy, level) && gradY_.computePatch(patchY, xy, level);
  }

  template <int Dim, typename PixelType>
  bool getGradXPatch(ImageMatrixPatch<PixelType, Dim, Dim>& patchX, const Vector2T& xy, int level) const {
    calcPatchGradients(xy, Dim, level);
    return gradX_.computePatch(patchX, xy, level);
  }

private:
  // number of level with lazy evaluated gradients
  int n_lazy_evaluated_levels_;
  bool horizontal_ = false;

  std::unique_ptr<IConvolutor> convolutor_;

  bool isSet_ = false;
  int levelsCount_ = 0;

  mutable ImageNoScalePyramidT gradX_;
  mutable ImageNoScalePyramidT gradY_;
  mutable std::vector<ImageMatrix<bool>> evaluated_;  // per level
  const ImagePyramidT* image_ = nullptr;              // keep source for lazy evaluation
};

}  // namespace cuvslam::sof
