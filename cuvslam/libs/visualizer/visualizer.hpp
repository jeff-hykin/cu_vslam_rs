
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
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "rerun.hpp"
#include "rerun/demo_utils.hpp"

namespace cuvslam::visualizer {

/**
 * Rerun-based 3D visualizer (Singleton)
 */
class RerunVisualizer {
public:
  // Singleton access - returns reference to static local object
  static RerunVisualizer& getInstance();

  rerun::RecordingStream& getRecordingStream() { return recording_; }

  // Delete copy constructor and assignment operator
  RerunVisualizer(const RerunVisualizer&) = delete;
  RerunVisualizer& operator=(const RerunVisualizer&) = delete;

  ~RerunVisualizer();

  /**
   * Set up the timeline for the current frame
   * @param frame_id Frame ID
   * @param timestamp Timestamps in nanoseconds
   */
  void setupTimeline(size_t frame_id, int64_t timestamp);

  /**
   * Clear viewport
   * @param viewport_name Rerun viewport name to clear
   */
  void clearViewport(const std::string& viewport_name);

  /**
   * Shutdown the visualizer and flush any pending data
   * Call this before program exit to avoid threading issues
   */
  void shutdown();

private:
  // Private constructor for singleton
  RerunVisualizer();

  rerun::RecordingStream recording_ = rerun::RecordingStream("cuvslam");
};

}  // namespace cuvslam::visualizer

// Global shutdown function
namespace cuvslam {
void shutdownVisualizer();
}
#endif
