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

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <sl/Camera.hpp>  // Main ZED camera API

// Volatile flag to safely communicate with the main loop
volatile sig_atomic_t keep_running = 1;

void CtrlCHandler(int signum) {
  std::cout << "\nCaught signal " << signum << " (SIGINT). Performing cleanup..." << std::endl;
  // Set flag to stop the main loop gracefully
  keep_running = 0;
}

std::string GenerateFileName(const std::string& prefix) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  const std::tm* local_tm = std::localtime(&now_time);

  std::ostringstream filename;
  filename << prefix << "_" << std::put_time(local_tm, "%Y%m%d_%H_%M_%S") << ".svo2";
  std::cout << "Filename: " << filename.str() << std::endl;
  return filename.str();
}

bool SetupBreakOnCtrlC() {
  struct sigaction sa = {};
  sa.sa_handler = CtrlCHandler;  // Set the function to be called
  sigemptyset(&sa.sa_mask);      // Clear the mask of signals to block
  sa.sa_flags = 0;               // Use default flags

  if (sigaction(SIGINT, &sa, nullptr) == -1) {
    return false;
  }
  return true;
}

void ConfigureZedTimestampClock() {
#if defined(ZED_SDK_MAJOR_VERSION) && defined(ZED_SDK_MINOR_VERSION) && \
    ((ZED_SDK_MAJOR_VERSION > 5) || (ZED_SDK_MAJOR_VERSION == 5 && ZED_SDK_MINOR_VERSION >= 3))
  sl::Camera::setTimestampClock(sl::TIMESTAMP_CLOCK::MONOTONIC_CLOCK);
#endif
}

int main() {
  if (!SetupBreakOnCtrlC()) {
    std::cerr << "Can't setup Ctrl+C signal handler." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Press Ctrl+C to stop recording." << std::endl;

  std::cout << "ZED SDK version: " << sl::Camera::getSDKVersion() << std::endl;
  ConfigureZedTimestampClock();
  sl::Camera zed;
  sl::InitParameters init_params;
  constexpr int kFPS = 100;
  init_params.camera_resolution = sl::RESOLUTION::VGA;
  init_params.sensors_required = false;
  init_params.camera_fps = kFPS;
  init_params.depth_mode = sl::DEPTH_MODE::NONE;

  if (zed.open(init_params) != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Can't init camera\n";
    return EXIT_FAILURE;
  }
  std::cout << "Camera initialized camera successfully.\n";

  const std::string filename = GenerateFileName("rec");
  sl::RecordingParameters rec_params(filename.c_str(), sl::SVO_COMPRESSION_MODE::H265);
  if (zed.enableRecording(rec_params) != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Can't enable recording.\n";
    return EXIT_FAILURE;
  }

  const sl::RuntimeParameters runtime(false, false);

  std::cout << "Start grab\n";
  int i = 0;
  while (keep_running) {  // Ctrl+C is not pressed
    const sl::ERROR_CODE err = zed.grab(runtime);
    if (err == sl::ERROR_CODE::SUCCESS) {
      if (i % kFPS == 0) {
        std::cout << "." << std::flush;
      }
      ++i;
    } else {
      std::cout << "error: " << err << std::endl;
    }
    if (i % (kFPS * 60) == 0) {
      const int num_minutes = i / (kFPS * 60);
      std::cout << std::endl << "recorded minutes: " << num_minutes << std::endl;
    }
  }
  std::cout << "Finish grab\n";

  zed.disableRecording();
  zed.close();
  std::cout << "All done\n";

  return EXIT_SUCCESS;
}
