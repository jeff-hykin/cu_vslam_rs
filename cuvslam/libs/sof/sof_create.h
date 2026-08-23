
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

#include "sof/feature_prediction_interface.h"
#include "sof/sof_config.h"
#include "sof/sof_mono_interface.h"
#include "sof/sof_multicamera_interface.h"

namespace cuvslam::sof {

enum class Implementation { kCPU, kGPU };

std::unique_ptr<IMonoSOF> CreateMonoSOF(Implementation implementation, CameraId cam_id,
                                        const camera::ICameraModel& intrinsics, std::unique_ptr<ISelector> selector,
                                        FeaturePredictorPtr feature_predictor, const Settings& sof_settings);

std::unique_ptr<IMultiSOF> CreateMultiSOF(Implementation implementation, const camera::Rig& rig,
                                          const camera::FrustumIntersectionGraph& fid,
                                          sof::FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                                          const odom::KeyFrameSettings& keyframe_settings);

}  // namespace cuvslam::sof
