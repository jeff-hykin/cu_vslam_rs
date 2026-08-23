
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

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

namespace cuvslam::epipolar {

///-------------------------------------------------------------------------------------------------
/// @brief Class handling the computation of the fundamental matrix. The general description of
/// algorithms to determine the fundamental matrix is beyond the scope of this documentation and
/// can be found at:
///
/// @Book{
/// Hartley2004, author = "Hartley, R.~I. and Zisserman, A.", title = "Multiple View Geometry in
/// Computer Vision", edition = "Second", year = "2004", publisher = "Cambridge University Press,
/// ISBN: 0521540518", chapter: 11 pages: 279-283 }
///
/// @tparam InternalScalarT Type of the internal scalar. The internal scalar type enables us to
/// run the computations with different precisions (e.g. double, float), internally, while all
/// interfaces remain in the globally defined float type.
///-------------------------------------------------------------------------------------------------
class ComputeFundamental {
public:
  enum class ReturnCode {
    Success,
    NotEnoughPoints,            // The operation failed because the object was not instantiated with enough points (@see
                                // MINIMUM_NUMBER_OF_POINTS)
    ContainersOfDifferentSize,  // The operation failed because the object was not instantiated with container of
                                // different size
    InvalidScale,          // The operation failed because it could not determine the proper scale of the input data
    UnsolvableDegeneracy,  // Degenerate Fundamental due to NULL space of A matrix, in Ax = 0 equation (vector x = F),
                           // dimension above 3
    PotentialHomography,   // Homography (planar surface or nodal camera), Ax = 0 has null space of dimension 3
    RuledQuadric,  // Ruled Quadric (3D points and 2 camera centers are on a quadric surface - HZ, 2-nd editio, p.296
    NonEssential   // we work in normalized space where Fundamental Matrix should be an Essential one
  };

  ComputeFundamental(const Vector2TVector& points1, const Vector2TVector& points2);

  ReturnCode findFundamental(Matrix3T& fundamental) const;

  ReturnCode getStatus() const;

  // This is for testing purposes only - will eventually be removed.
  const Vector9T& getMatATASingularValues() const;

  bool isPotentialHomography() const;

  static constexpr float EssentialThreshold();

  static const size_t MINIMUM_NUMBER_OF_POINTS = 8;

private:
  ReturnCode composeFundamentalSystem(const Vector2TVector& points1, const Vector2TVector& points2);

  ReturnCode returnCode_;
  Matrix3T fundamental_;
  Vector9T singularValues_;
};

void ExtractRotationTranslationFromEssential(const Matrix3T& e, Matrix3T& rot1, Matrix3T& rot2, Vector3T& t);

float ComputeQuadraticResidual(const Vector2T& pt1, const Vector2T& pt2, const Matrix3T& fundamental1To2);

bool FindOptimalCameraMatrixFromEssential(Vector2TPairVectorCIt points2DStart, Vector2TPairVectorCIt points2DEnd,
                                          const Matrix3T& essential, Isometry3T& relativeTransform,
                                          const size_t minCount = 3);

bool GoldStandardEssentialEstimate(const Vector2TPairVector& sampleSequence, Isometry3T& relative2To1CameraTransform);

bool RLM_EssentialEstimate(const Vector2TPairVector& sampleSequence, Isometry3T& relative2To1CameraTransform);

}  // namespace cuvslam::epipolar
