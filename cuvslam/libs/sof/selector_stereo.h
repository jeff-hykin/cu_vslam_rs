
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

#include "sof/selector_interface.h"
#include "sof/sof.h"

namespace cuvslam::sof {

struct SelectorStereoSettings {
  // Initialize new primary-camera feature tracks if the percent of survivor tracks
  // from the last feature initialization is less than this value.
  float survivor_from_last = 41.f;
};

class SelectorStereo : public ISelector {
public:
  SelectorStereo(const SelectorStereoSettings& settings);

  void set_image_width(uint32_t width) override;
  void reset_selector() override;
  bool select(const TracksVector& cur_frame_tracks) override;
  void set_tracks(const TracksVector& cur_frame_tracks) override;

private:
  SelectorStereoSettings settings_;
  bool first_feature_initialization_selected_ = false;
  TracksVector last_feature_initialization_tracks_;

public:
  void save_state(serial::Writer& w) const final {
    w.write_tag(0x53454C53);  // "SELS"
    w.write_bool(first_feature_initialization_selected_);
    last_feature_initialization_tracks_.save_state(w);
  }
  void load_state(serial::Reader& r) final {
    r.expect_tag(0x53454C53, "SelectorStereo");
    first_feature_initialization_selected_ = r.read_bool();
    last_feature_initialization_tracks_.load_state(r);
  }
};

}  // namespace cuvslam::sof
