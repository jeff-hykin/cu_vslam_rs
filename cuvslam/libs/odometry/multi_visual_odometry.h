
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
#include "common/vector_3t.h"
#include "pipelines/sfm_solver_interface.h"
#include "pipelines/track_online_multi.h"

#include "odometry/ipredictor.h"
#include "odometry/multi_visual_odometry_base.h"
#include "odometry/svo_config.h"

namespace cuvslam::odom {

class MultiVisualOdometry : public MultiVisualOdometryBase {
public:
  MultiVisualOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig, const Settings& svo_settings,
                      bool use_gpu);
  ~MultiVisualOdometry() override = default;

  bool do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) override;
  pipelines::ISFMSolver& get_solver() override { return solver_; }

private:
  pipelines::SolverSfMMulti solver_;
};

}  // namespace cuvslam::odom
