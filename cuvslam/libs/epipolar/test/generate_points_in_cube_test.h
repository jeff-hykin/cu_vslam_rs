
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

#include <random>

#include "common/vector_3t.h"

namespace test {
using namespace cuvslam;

///-------------------------------------------------------------------------------------------------
/// @brief Generates numPoints in a cube defined by its minimum and maximum range. The range can
/// be thought of as the lower-left-back and the up-right-front corners. Points are generated
/// randomly (uniform distribution) in the cube.
///
/// @param numPoints Number of points.
/// @param minRange  The minimum range.
/// @param maxRange  The maximum range.
///
/// @return A vector of points uniformly distributed in the cube.
///-------------------------------------------------------------------------------------------------
template <typename _Validador>
Vector3TVector GeneratePointsInCube(const size_t numPoints, const Vector3T& minRange, const Vector3T& maxRange,
                                    const _Validador validator) {
  // std::rand() is seeded per-test; each call advances its state, so repeated
  // calls to GeneratePointsInCube within the same test get different seeds.
  auto seed = std::rand();
  std::mt19937 gen(seed);
  std::uniform_real_distribution uniformDistX(minRange.x(), maxRange.x());
  std::uniform_real_distribution uniformDistY(minRange.y(), maxRange.y());
  std::uniform_real_distribution uniformDistZ(minRange.z(), maxRange.z());
  Vector3TVector points3D;
  const size_t maxIter = numPoints * 200;

  for (size_t i = 0; i < maxIter && points3D.size() < numPoints; i++) {
    const Vector3T p(uniformDistX(gen), uniformDistY(gen), uniformDistZ(gen));

    if (!validator(p)) {
      continue;
    }

    points3D.push_back(p);
  }

  return points3D;
}

inline Vector3TVector GeneratePointsInCube(size_t numPoints, const Vector3T& minRange, const Vector3T& maxRange) {
  return GeneratePointsInCube(numPoints, minRange, maxRange, [](const Vector3T&) -> bool { return true; });
}

}  // namespace test
