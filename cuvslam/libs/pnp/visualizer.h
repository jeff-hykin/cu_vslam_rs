
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

#ifdef USE_RERUN
#include <memory>
#include <string>
#include <vector>

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/rerun.h"
#include "common/vector_3t.h"

#include "visualizer/visualizer.hpp"

namespace cuvslam::pnp {
void clearViewport(const std::string& viewport_name);

void logLandmarks(const std::vector<std::reference_wrapper<const Vector3T>>& landmarks,
                  const Isometry3T& camera_from_world, const camera::ICameraModel* camera_model,
                  const std::string& viewport_name, const Color& color);

void logObservations(const std::vector<std::reference_wrapper<const camera::Observation>>& observations,
                     const camera::Rig& rig, const std::string& viewport_name, const Color& color);
}  // namespace cuvslam::pnp
#endif
