
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

#include "odometry/svo_config.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "sof/sof.h"

namespace cuvslam::sof {

class KFSelector {
public:
  KFSelector(const odom::KeyFrameSettings& kf_settings);

  bool first_kf_selected() const;
  void reset();

  bool select(const TracksVector& cur_frame_tracks, const int64_t current_timestamp_ns,
              const TracksVector& last_kf_tracks, const int64_t last_kf_timestamp,
              const odom::KeyFrameSettings& kf_settings);

  void save_state(serial::Writer& w) const { w.write_bool(first_kf_selected_); }
  void load_state(serial::Reader& r) { first_kf_selected_ = r.read_bool(); }

private:
  bool first_kf_selected_ = false;

  // profiler
  profiler::VioProfiler::DomainHelper profiler_domain_ = profiler::VioProfiler::DomainHelper("VIO");
  const uint32_t profiler_color_ = 0x00FF00;
};

}  // namespace cuvslam::sof
