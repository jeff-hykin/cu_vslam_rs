
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

#include "utils/cuvslam_yaml_config.h"
#include "cuvslam/cuvslam2_internal.h"

#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.h"
#include "yaml-cpp/yaml.h"

namespace cuvslam {
namespace {

void WarnUnknownKeys(const YAML::Node& node, const char* section, std::initializer_list<const char*> known_keys) {
  for (const auto& entry : node) {
    const std::string key = entry.first.as<std::string>();
    bool found = false;
    for (const char* k : known_keys) {
      if (key == k) {
        found = true;
        break;
      }
    }
    if (!found) {
      TraceWarning("Load%sFromFile: unknown key '%s' (ignored)\n", section, key.c_str());
    }
  }
}

YAML::Node LoadYamlRootMap(const char* filepath) {
  if (!filepath) {
    throw std::runtime_error("config filepath is null");
  }
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("cannot open file '") + filepath + "'");
  }
  YAML::Node root;
  try {
    root = YAML::LoadFile(filepath);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(std::string("failed to parse YAML file '") + filepath + "': " + e.what());
  }
  if (!root || !root.IsMap()) {
    throw std::runtime_error("YAML file must contain a map at root level");
  }
  return root;
}

std::optional<bool> ParseOptionalBool(const YAML::Node& node, const char* key) {
  if (node.IsNull()) {
    return std::nullopt;
  }
  if (!node.IsScalar()) {
    throw std::runtime_error(std::string("internals.") + key + " must be a bool or null");
  }
  return node.as<bool>();
}

}  // namespace

bool LoadInternalsFromFile(const char* filepath, cuvslam::internal::Internals& internals) {
  YAML::Node root = LoadYamlRootMap(filepath);
  YAML::Node node = root["internals"];
  if (!node || !node.IsMap()) {
    return false;
  }
  if (YAML::Node v = node["num_desired_tracks"]) internals.num_desired_tracks = v.as<int32_t>();
  if (YAML::Node v = node["border_top"]) internals.border_top = v.as<int32_t>();
  if (YAML::Node v = node["border_bottom"]) internals.border_bottom = v.as<int32_t>();
  if (YAML::Node v = node["border_left"]) internals.border_left = v.as<int32_t>();
  if (YAML::Node v = node["border_right"]) internals.border_right = v.as<int32_t>();
  if (YAML::Node v = node["box3_prefilter"]) internals.box3_prefilter = v.as<bool>();
  if (YAML::Node v = node["ransac_filter"]) internals.ransac_filter = v.as<bool>();
  if (YAML::Node v = node["kf_survivor_from_last"]) internals.kf_survivor_from_last = v.as<float>();
  if (YAML::Node v = node["kf_max_timedelta_between_kfs_s"]) internals.kf_max_timedelta_between_kfs_s = v.as<int64_t>();
  if (YAML::Node v = node["kf_override_frame_selection"]) {
    internals.kf_override_frame_selection = ParseOptionalBool(v, "kf_override_frame_selection");
    if (internals.kf_override_frame_selection.has_value()) {
      TraceWarning(
          "LoadInternalsFromFile: 'kf_override_frame_selection' is a per-frame override; loading it from YAML "
          "applies the same keyframe decisions to every frame.\n");
    }
  }
  if (YAML::Node v = node["vo_pnp_lambda"]) internals.vo_pnp_lambda = v.as<float>();
  if (YAML::Node v = node["vo_pnp_huber"]) internals.vo_pnp_huber = v.as<float>();
  if (YAML::Node v = node["vo_pnp_max_iteration"]) internals.vo_pnp_max_iteration = v.as<int32_t>();
  if (YAML::Node v = node["vo_pnp_recalculate_cov"]) internals.vo_pnp_recalculate_cov = v.as<bool>();
  if (YAML::Node v = node["vo_pnp_filter_new_observations"]) internals.vo_pnp_filter_new_observations = v.as<bool>();
  if (YAML::Node v = node["vo_pnp_max_obs_per_camera"]) internals.vo_pnp_max_obs_per_camera = v.as<int32_t>();
  if (YAML::Node v = node["vo_pnp_point_z_thresh"]) internals.vo_pnp_point_z_thresh = v.as<float>();
  if (YAML::Node v = node["vo_pnp_min_observations"]) internals.vo_pnp_min_observations = v.as<int32_t>();
  if (YAML::Node v = node["vo_pnp_cost_thresh"]) internals.vo_pnp_cost_thresh = v.as<float>();
  if (YAML::Node v = node["inertial_stereo_pnp_lambda"]) internals.inertial_stereo_pnp_lambda = v.as<float>();
  if (YAML::Node v = node["inertial_stereo_pnp_huber"]) internals.inertial_stereo_pnp_huber = v.as<float>();
  if (YAML::Node v = node["inertial_stereo_pnp_max_iteration"])
    internals.inertial_stereo_pnp_max_iteration = v.as<int32_t>();
  if (YAML::Node v = node["inertial_stereo_pnp_recalculate_cov"])
    internals.inertial_stereo_pnp_recalculate_cov = v.as<bool>();
  if (YAML::Node v = node["inertial_stereo_pnp_filter_new_observations"])
    internals.inertial_stereo_pnp_filter_new_observations = v.as<bool>();
  if (YAML::Node v = node["inertial_stereo_pnp_max_obs_per_camera"])
    internals.inertial_stereo_pnp_max_obs_per_camera = v.as<int32_t>();
  if (YAML::Node v = node["inertial_stereo_pnp_point_z_thresh"])
    internals.inertial_stereo_pnp_point_z_thresh = v.as<float>();
  if (YAML::Node v = node["inertial_stereo_pnp_min_observations"])
    internals.inertial_stereo_pnp_min_observations = v.as<int32_t>();
  if (YAML::Node v = node["inertial_stereo_pnp_cost_thresh"]) internals.inertial_stereo_pnp_cost_thresh = v.as<float>();
  if (YAML::Node v = node["imu_pnp_robustifier_scale"]) internals.imu_pnp_robustifier_scale = v.as<float>();
  if (YAML::Node v = node["imu_pnp_max_iteration"]) internals.imu_pnp_max_iteration = v.as<int32_t>();
  if (YAML::Node v = node["imu_pnp_min_observations"]) internals.imu_pnp_min_observations = v.as<int32_t>();
  if (YAML::Node v = node["icp_lambda"]) internals.icp_lambda = v.as<float>();
  if (YAML::Node v = node["icp_huber_vis"]) internals.icp_huber_vis = v.as<float>();
  if (YAML::Node v = node["icp_huber_depth"]) internals.icp_huber_depth = v.as<float>();
  if (YAML::Node v = node["icp_max_iteration"]) internals.icp_max_iteration = v.as<int32_t>();
  if (YAML::Node v = node["icp_cost_thresh"]) internals.icp_cost_thresh = v.as<float>();
  if (YAML::Node v = node["icp_min_scale_level"]) internals.icp_min_scale_level = v.as<int32_t>();
  if (YAML::Node v = node["icp_max_scale_level"]) internals.icp_max_scale_level = v.as<int32_t>();
  if (YAML::Node v = node["icp_num_iters_per_scale"]) internals.icp_num_iters_per_scale = v.as<int32_t>();
  if (YAML::Node v = node["icp_blending_alpha"]) internals.icp_blending_alpha = v.as<float>();
  WarnUnknownKeys(node, "Internals",
                  {"num_desired_tracks",
                   "border_top",
                   "border_bottom",
                   "border_left",
                   "border_right",
                   "box3_prefilter",
                   "ransac_filter",
                   "kf_survivor_from_last",
                   "kf_max_timedelta_between_kfs_s",
                   "kf_override_frame_selection",
                   "vo_pnp_lambda",
                   "vo_pnp_huber",
                   "vo_pnp_max_iteration",
                   "vo_pnp_recalculate_cov",
                   "vo_pnp_filter_new_observations",
                   "vo_pnp_max_obs_per_camera",
                   "vo_pnp_point_z_thresh",
                   "vo_pnp_min_observations",
                   "vo_pnp_cost_thresh",
                   "inertial_stereo_pnp_lambda",
                   "inertial_stereo_pnp_huber",
                   "inertial_stereo_pnp_max_iteration",
                   "inertial_stereo_pnp_recalculate_cov",
                   "inertial_stereo_pnp_filter_new_observations",
                   "inertial_stereo_pnp_max_obs_per_camera",
                   "inertial_stereo_pnp_point_z_thresh",
                   "inertial_stereo_pnp_min_observations",
                   "inertial_stereo_pnp_cost_thresh",
                   "imu_pnp_robustifier_scale",
                   "imu_pnp_max_iteration",
                   "imu_pnp_min_observations",
                   "icp_lambda",
                   "icp_huber_vis",
                   "icp_huber_depth",
                   "icp_max_iteration",
                   "icp_cost_thresh",
                   "icp_min_scale_level",
                   "icp_max_scale_level",
                   "icp_num_iters_per_scale",
                   "icp_blending_alpha"});
  return true;
}

bool LoadOdometryConfigFromFile(const char* filepath, Odometry::Config& config) {
  YAML::Node root = LoadYamlRootMap(filepath);
  YAML::Node node = root["odometry"];
  if (!node || !node.IsMap()) {
    return false;
  }
  if (YAML::Node v = node["use_gpu"]) config.use_gpu = v.as<bool>();
  if (YAML::Node v = node["async_sba"]) config.async_sba = v.as<bool>();
  if (YAML::Node v = node["use_motion_model"]) config.use_motion_model = v.as<bool>();
  if (YAML::Node v = node["use_denoising"]) config.use_denoising = v.as<bool>();
  if (YAML::Node v = node["rectified_stereo_camera"]) config.rectified_stereo_camera = v.as<bool>();
  if (YAML::Node v = node["enable_observations_export"]) config.enable_observations_export = v.as<bool>();
  if (YAML::Node v = node["enable_landmarks_export"]) config.enable_landmarks_export = v.as<bool>();
  if (YAML::Node v = node["enable_final_landmarks_export"]) config.enable_final_landmarks_export = v.as<bool>();
  if (YAML::Node v = node["max_frame_delta_s"]) config.max_frame_delta_s = v.as<float>();
  if (YAML::Node v = node["debug_imu_mode"]) config.debug_imu_mode = v.as<bool>();
  auto to_lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  if (YAML::Node v = node["multicam_mode"]) {
    const std::string s = to_lower(v.as<std::string>());
    if (s == "performance")
      config.multicam_mode = Odometry::MulticameraMode::Performance;
    else if (s == "precision")
      config.multicam_mode = Odometry::MulticameraMode::Precision;
    else if (s == "moderate")
      config.multicam_mode = Odometry::MulticameraMode::Moderate;
    else
      throw std::runtime_error("unknown multicam_mode: " + s);
  }
  if (YAML::Node v = node["odometry_mode"]) {
    const std::string s = to_lower(v.as<std::string>());
    if (s == "multicamera")
      config.odometry_mode = Odometry::OdometryMode::Multicamera;
    else if (s == "inertial")
      config.odometry_mode = Odometry::OdometryMode::Inertial;
    else if (s == "rgbd")
      config.odometry_mode = Odometry::OdometryMode::RGBD;
    else if (s == "mono")
      config.odometry_mode = Odometry::OdometryMode::Mono;
    else if (s == "multisensor")
      config.odometry_mode = Odometry::OdometryMode::Multisensor;
    else
      throw std::runtime_error("unknown odometry_mode: " + s);
  }
  if (YAML::Node rgbd = node["rgbd_settings"]; rgbd && rgbd.IsMap()) {
    if (YAML::Node v = rgbd["depth_scale_factor"]) config.rgbd_settings.depth_scale_factor = v.as<float>();
    if (YAML::Node v = rgbd["depth_camera_id"]) config.rgbd_settings.depth_camera_id = v.as<int32_t>();
    if (YAML::Node v = rgbd["enable_depth_stereo_tracking"])
      config.rgbd_settings.enable_depth_stereo_tracking = v.as<bool>();
  }
  if (YAML::Node ms = node["multisensor_settings"]; ms && ms.IsMap()) {
    if (YAML::Node v = ms["depth_camera_ids"])
      config.multisensor_settings.depth_camera_ids = v.as<std::vector<int32_t>>();
    if (YAML::Node v = ms["depth_scale_factor"]) config.multisensor_settings.depth_scale_factor = v.as<float>();
    if (YAML::Node v = ms["enable_depth_stereo_tracking"])
      config.multisensor_settings.enable_depth_stereo_tracking = v.as<bool>();
  }
  if (node["debug_dump_directory"]) {
    TraceWarning(
        "LoadOdometryConfigFromFile: 'debug_dump_directory' cannot be loaded from YAML (string_view does not own "
        "memory); set it directly on the returned config.\n");
  }
  WarnUnknownKeys(node, "OdometryConfig",
                  {"use_gpu", "async_sba", "use_motion_model", "use_denoising", "rectified_stereo_camera",
                   "enable_observations_export", "enable_landmarks_export", "enable_final_landmarks_export",
                   "max_frame_delta_s", "debug_imu_mode", "multicam_mode", "odometry_mode", "rgbd_settings",
                   "multisensor_settings", "debug_dump_directory"});
  return true;
}

bool LoadSlamConfigFromFile(const char* filepath, Slam::Config& config) {
  YAML::Node root = LoadYamlRootMap(filepath);
  YAML::Node node = root["slam"];
  if (!node || !node.IsMap()) {
    return false;
  }
  if (YAML::Node v = node["use_gpu"]) config.use_gpu = v.as<bool>();
  if (YAML::Node v = node["sync_mode"]) config.sync_mode = v.as<bool>();
  if (YAML::Node v = node["enable_reading_internals"]) config.enable_reading_internals = v.as<bool>();
  if (YAML::Node v = node["planar_constraints"]) config.planar_constraints = v.as<bool>();
  if (YAML::Node v = node["gt_align_mode"]) config.gt_align_mode = v.as<bool>();
  if (YAML::Node v = node["map_cell_size"]) config.map_cell_size = v.as<float>();
  if (YAML::Node v = node["max_landmarks_distance"]) config.max_landmarks_distance = v.as<float>();
  if (YAML::Node v = node["max_map_size"]) config.max_map_size = v.as<uint32_t>();
  if (YAML::Node v = node["throttling_time_ms"]) config.throttling_time_ms = v.as<uint32_t>();
  if (node["map_cache_path"]) {
    TraceWarning(
        "LoadSlamConfigFromFile: 'map_cache_path' cannot be loaded from YAML (string_view does not own memory); set it "
        "directly on the returned config.\n");
  }
  WarnUnknownKeys(node, "SlamConfig",
                  {"use_gpu", "sync_mode", "enable_reading_internals", "planar_constraints", "gt_align_mode",
                   "map_cell_size", "max_landmarks_distance", "max_map_size", "throttling_time_ms", "map_cache_path"});
  return true;
}

bool LoadExpertParamsFromFile(const char* filepath, std::map<std::string, std::string>& params) {
  YAML::Node root = LoadYamlRootMap(filepath);
  YAML::Node node = root["expert_params"];
  if (!node || !node.IsMap()) {
    return false;
  }
  bool any_written = false;
  for (const auto& entry : node) {
    const std::string key = entry.first.as<std::string>();
    // CLI-flag entries already in params take precedence — do not overwrite them.
    if (params.find(key) == params.end()) {
      params[key] = entry.second.as<std::string>();
      any_written = true;
    }
  }
  return any_written;
}

}  // namespace cuvslam
