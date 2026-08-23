
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

#include "Eigen/Eigenvalues"

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

namespace cuvslam::epipolar {

const auto DEFAULT_HUBER_DELTA = 1.0f;

template <int _Size, bool _useRotPrealign = true>
bool OptimizeCameraExtrinsicsExpMap(Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt,
                                    Vector2TVectorCIt points2DBeginIt, Vector2TVectorCIt points2DEndIt,
                                    Isometry3T& cameraMatrix, const float hDelta = DEFAULT_HUBER_DELTA);

template <bool _useRotPrealign = true>
bool OptimizeCameraExtrinsicsExpMapConstrained(Vector3TVectorCIt points3DBeginIt, Vector3TVectorCIt points3DEndIt,
                                               Vector2TVectorCIt points2DBeginIt, Vector2TVectorCIt points2DEndIt,
                                               Isometry3T& cameraMatrix, const Vector3T& constraint,
                                               const float hDelta = DEFAULT_HUBER_DELTA);

}  // namespace cuvslam::epipolar
