
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

#include "common/types.h"

namespace test::epipolar {
using namespace cuvslam;

///-------------------------------------------------------------------------------------------------
/// @brief Optimize the camera matrix (extrinsics) parameters from a container of 3D points and
/// a container of matching 2D points.
///
/// @param points3DBeginIt       The begin iterator of the 3D point container.
/// @param points3DEndIt         The end iterator of the 3D point container.
/// @param points2DBeginIt       The begin iterator of the 2D point container.
/// @param points2DEndIt         The end iterator of the 2D point container.
/// @param [in,out] cameraMatrix The camera matrix. It is required to pass this parameter an
/// initial guess of the camera matrix (e.g. previous frame) as the internal solver will use it
/// to improve convergence. The camera matrix is the position (rotation and translation) of the
/// camera expressed in world coordinates.
///
/// @return true if the algorithm converged; false if (i) the algorithm did not converge, (ii)
/// the number of points for the 2D and 3D containers do not match, (iii)
/// OptimizeCameraExtrinsics is called with less than 4 points.
///-------------------------------------------------------------------------------------------------
bool OptimizeCameraExtrinsicsOpenCV(Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt,
                                    Vector2TVectorCIt points2DBeginIt, Vector2TVectorCIt points2DEndIt,
                                    Isometry3T& cameraMatrix);

}  // namespace test::epipolar
