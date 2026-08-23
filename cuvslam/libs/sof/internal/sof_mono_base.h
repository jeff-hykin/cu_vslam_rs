
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

#include <optional>
#include <vector>

#include "common/camera_id.h"
#include "common/image_matrix.h"

#include "sof/feature_prediction_interface.h"
#include "sof/selector_interface.h"
#include "sof/sof.h"
#include "sof/sof_mono_interface.h"

namespace cuvslam::sof {

class MonoSOFBase : public IMonoSOF {
public:
  MonoSOFBase(CameraId cam_id, std::unique_ptr<ISelector> selector, FeaturePredictorPtr& feature_predictor)
      : feature_predictor_(feature_predictor), cam_id_(cam_id), feature_selector_(std::move(selector)) {}
  CameraId camera_id() const override { return cam_id_; }

protected:
  void PredictFeatureLocations(const Isometry3T& predicted_world_from_rig);
  void filterByPredictionError();
  void IncrementTracksAge();

  void PrepareInputMask(const ImageShape& shape);
  void KillTracksWithinMask();
  bool SelectKeyframe(const MonoSOFFrameSettings& frame_settings);

  void save_base_state(serial::Writer& w) const {
    w.write_tag(0x4D4F4E4F);  // "MONO"
    w.write_pod(cam_id_);
    tracks_.save_state(w);
    feature_selector_->save_state(w);
  }
  void load_base_state(serial::Reader& r) {
    r.expect_tag(0x4D4F4E4F, "MonoSOFBase");
    const CameraId saved_cam_id = r.read_pod<CameraId>();
    if (saved_cam_id != cam_id_) {
      throw std::runtime_error("cuVSLAM state deserialization: mono SOF camera id mismatch");
    }
    tracks_.load_state(r);
    feature_selector_->load_state(r);
  }

  // internal state
  TracksVector tracks_;  // current tracks

  FeaturePredictorPtr feature_predictor_;
  const CameraId cam_id_;
  std::unique_ptr<ISelector> feature_selector_;

  std::vector<TrackId> predictionTrackIds_;
  Prediction predictedUVs_;
  std::vector<float> prediction_deltas_;
  std::vector<float> original_prediction_deltas_;

  bool input_mask_present_ = false;
  ImageMatrix<uint8_t> input_mask_;
};

void KillTracksOnBorder(size_t w, size_t h, size_t border_top, size_t border_bottom, size_t border_left,
                        size_t border_right, TracksVector& tracks);

}  // namespace cuvslam::sof
