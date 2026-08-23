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

#include <map>
#include <string>

#include "cuvslam/cuvslam2.h"

namespace cuvslam {

/**
 * @file cuvslam_yaml_config.h
 * @brief Optional YAML file loaders.
 *
 * Implemented in the `utils` static library under libs/utils (links yaml-cpp). Tools and
 * bindings should link `utils` alongside libcuvslam if they want file-based config.
 */

/// @brief Load per-frame internals from the `internals:` section of a YAML file.
/// @param filepath Path to the YAML file.
/// @param internals Output Internals to populate.
/// @return true if the `internals` section was found and applied, false if the section is absent.
/// @throws std::runtime_error if the file cannot be opened or parsed.
bool LoadInternalsFromFile(const char* filepath, cuvslam::internal::Internals& internals);

/// @brief Load odometry configuration from the `odometry:` section of a YAML file.
/// @param filepath Path to the YAML file.
/// @param config Output Config to populate.
/// @return true if the `odometry` section was found and applied, false if the section is absent.
/// @throws std::runtime_error if the file cannot be opened or parsed, or if `multicam_mode` /
///         `odometry_mode` contains an unrecognised string value.
bool LoadOdometryConfigFromFile(const char* filepath, Odometry::Config& config);

/// @brief Load SLAM configuration from the `slam:` section of a YAML file.
/// @param filepath Path to the YAML file.
/// @param config Output Config to populate.
/// @return true if the `slam` section was found and applied, false if the section is absent.
/// @throws std::runtime_error if the file cannot be opened or parsed.
bool LoadSlamConfigFromFile(const char* filepath, Slam::Config& config);

/// @brief Load expert parameters from the `expert_params:` section of a YAML file.
///
/// Every key–value pair under `expert_params` is inserted into @p params as a string,
/// exactly as `ApplyPersistentInternalParameters` expects.  Any key already present in @p params is
/// left untouched, so CLI-flag overrides set before this call are preserved.
///
/// @param filepath Path to the YAML file.
/// @param params   Output map to populate (existing entries are NOT overwritten).
/// @return true if the `expert_params` section was found and at least one entry written,
///         false if the section is absent.
/// @throws std::runtime_error if the file cannot be opened or parsed.
bool LoadExpertParamsFromFile(const char* filepath, std::map<std::string, std::string>& params);

}  // namespace cuvslam
