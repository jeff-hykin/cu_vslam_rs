
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
/// @brief Convenience method to create a camera matrix (positional matrix) from parameters.
///
/// @param position              The position of the camera matrix. This is the location of the
/// desired camera.
/// @param lookingAt             A point the camera looks at (in world coordinates). +Z in the
/// constructed camera frame points from the camera toward this point (OpenCV-style forward).
/// @param up                    The up vector is a hint for the direction of the y-axis. The up
/// vector will be tranformed to the y-axis in a way that ensure an orthonormal basis.
/// @param [in,out] cameraMatrix The camera matrix that was created, if the call to
/// CreateCameraMatrix was successful. Unchanged otherwise.
///
/// @return true if the camera matrix was created successfully; false if the up vector is too
/// colinear to the {origin, position} direction.
///-------------------------------------------------------------------------------------------------
bool CreateCameraMatrix(const Vector3T& position, const Vector3T& lookingAt, const Vector3T& up,
                        Isometry3T& cameraMatrix);

///-------------------------------------------------------------------------------------------------
/// @brief Projects a 3D point onto a camera plane in local coordinates.
///
/// @param toLocalSpace            The transform to local space.
/// @param point3D                 The 3D point.
/// @param [in,out] projectedPoint If non-null, the projected point.
///
/// @return false if the 3D point is at the transform's origin, true otherwise.
///-------------------------------------------------------------------------------------------------
float Project3DPointInLocalCoordinates(const Isometry3T& toLocalSpace, const Vector3T& point3D,
                                       Vector2T& projectedPoint);

template <int _Size, typename _Vector3, typename _Scalar = typename _Vector3::Scalar>
auto Project3DPoint(const _Vector3& local, const _Scalar eps = epsilon<_Scalar>()) {
  return local.head(_Size) / AvoidZero(local.z(), eps);
}

///-------------------------------------------------------------------------------------------------
/// @brief Projects a vector of 3D points onto a camera plane in local coordinates.
///
/// @param toLocalSpace                    The transform to local space.
/// @param points3DBeginIt,points3DEndIt   3D points iterator
/// @param [out] projectedPoint            the projected point.
///-------------------------------------------------------------------------------------------------
void Project3DPointsInLocalCoordinates(const Isometry3T& toLocalSpace, Vector3TVectorCIt points3DBeginIt,
                                       Vector3TVectorCIt points3DEndIt, Vector2TVector& projectedPoints);

// function above call function below
void Project3DPointsInLocalCoordinates(const Isometry3T& toLocalSpace, const Vector3TVector& points3D,
                                       Vector2TVector& projectedPoints);

///-------------------------------------------------------------------------------------------------
/// @brief Generates a pair of 2D points from 3D points and two camera matrices.
///
/// @param points3D                The points 3D.
/// @param camera1                 The first camera.
/// @param camera2                 The second camera.
/// @param [in,out] pairOf2DPoints If non-null, the pair of 2D points.
///-------------------------------------------------------------------------------------------------
void GeneratePairOf2DPointsFrom3DPoints(const Vector3TVector& points3D, const Isometry3T& camera1,
                                        const Isometry3T& camera2, Vector2TPairVector* pairOf2DPoints);

///-------------------------------------------------------------------------------------------------
/// @brief Calculates the distance from expected rotation matrix.
///
/// @param Matrix Type of the matrix.
/// @param expected The expected matrix.
/// @param actual   The actual matrix.
///
/// @return The calculated distance from expected rotation matrix.
///-------------------------------------------------------------------------------------------------
float CalculateDistanceFromExpectedRotationMatrix(const QuaternionT::RotationMatrixType& expected,
                                                  const QuaternionT::RotationMatrixType& actual);

///-------------------------------------------------------------------------------------------------
/// @brief Calculates the Frobenius norm of the difference of 2 matrices.
///
/// @param Matrix  Type of the matrix.
/// @param expected The expected matrix.
/// @param actual   The actual matrix.
///
/// @return The resulting Normalized Frobenius norm.
///-------------------------------------------------------------------------------------------------
template <typename Matrix>
float CalculateMatricesFrobeniusDistance(const Eigen::MatrixBase<Matrix>& expected,
                                         const Eigen::MatrixBase<Matrix>& actual) {
  return std::sqrt((expected - actual).squaredNorm() / std::min(expected.squaredNorm(), actual.squaredNorm()));
}

}  // namespace cuvslam::epipolar
