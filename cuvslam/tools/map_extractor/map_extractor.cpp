
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

#include <fstream>
#include <iostream>

#include "gflags/gflags.h"
#include "json/json.h"

#include "camera/rig.h"
#include "slam/map/database/lmdb_slam_database.h"
#include "slam/map/map.h"

using namespace cuvslam::slam;

DEFINE_string(map_path, ".", "Path to the saved map");

int main(int arg_c, char** arg_v) {
  gflags::ParseCommandLineFlags(&arg_c, &arg_v, /*remove flags = */ true);
  if (arg_c != 1) {
    std::cout << "This tools doesn't expect command line arguments, only flags listed with --help" << std::endl
              << "If you see this message, please check your command line." << std::endl;
    return EXIT_FAILURE;
  }

  auto lmdb = std::make_shared<LmdbSlamDatabase>();
  if (!lmdb->Open(FLAGS_map_path.c_str(), LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS)) {
    std::cout << "Failed to load the map!" << std::endl;
    return -1;
  }

  cuvslam::camera::Rig rig;

  Map map(rig, FeatureDescriptorType::kNone, false);

  if (!map.AttachDatabase(lmdb, true)) {
    std::cout << "Failed to attach empty database!" << std::endl;
    return -1;
  }

  auto landmarks_spatial_index = map.landmarks_spatial_index_;
  const PoseGraphHypothesis& pose_graph_hypothesis = map.pose_graph_hypothesis_;
  const PoseGraph& pose_graph = map.pose_graph_;

  Json::Value root;

  pose_graph.QueryKeyframes([&](KeyFrameId kf_id) {
    const cuvslam::Isometry3T* rig_from_world = pose_graph_hypothesis.GetKeyframePose(kf_id);
    if (rig_from_world == nullptr) {
      return;
    }
    Json::Value kf;
    kf["keyframe_id"] = static_cast<int>(kf_id);

    cuvslam::QuaternionT qp{rig_from_world->linear()};

    kf["quaternion"]["x"] = qp.x();
    kf["quaternion"]["y"] = qp.y();
    kf["quaternion"]["z"] = qp.z();
    kf["quaternion"]["w"] = qp.w();

    kf["translation"]["x"] = rig_from_world->translation().x();
    kf["translation"]["y"] = rig_from_world->translation().y();
    kf["translation"]["z"] = rig_from_world->translation().z();

    root["poses"].append(kf);
  });

  pose_graph.QueryEdges(
      [&](KeyFrameId from, KeyFrameId to, const cuvslam::Isometry3T&, const cuvslam::Matrix6T&) -> bool {
        Json::Value edge;
        edge["start"] = static_cast<int>(from);
        edge["end"] = static_cast<int>(to);

        root["edges"].append(edge);
        return true;
      });

  landmarks_spatial_index->Query([&](LandmarkId id) {
    const cuvslam::Vector3T xyz = landmarks_spatial_index->GetLandmarkOrStagedCoords(id, pose_graph_hypothesis);

    Json::Value landmark;

    landmark["pose"]["x"] = xyz.x();
    landmark["pose"]["y"] = xyz.y();
    landmark["pose"]["z"] = xyz.z();

    landmark["id"] = static_cast<int>(id);

    landmarks_spatial_index->QueryLandmarkRelations(id, [&](KeyFrameId kf_id, const cuvslam::Vector2T&) {
      landmark["keyframes"].append(static_cast<int>(kf_id));
      return true;
    });

    root["landmarks"].append(landmark);

    return true;
  });

  std::ofstream file("map.json");

  Json::StreamWriterBuilder builder;
  std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

  writer->write(root, &file);
  file.close();

  return 0;
}
