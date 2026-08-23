
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

#include <iostream>

#include "camera/frustum_intersection_graph.h"
#include "camera_rig_edex/camera_rig_edex.h"
#include "common/include_gtest.h"
#include "common/log.h"

namespace Test {
using namespace cuvslam;
using namespace cuvslam::camera;

using CameraGraphNode = FrustumIntersectionGraph::CameraGraphNode;
using ConnectedCam = CameraGraphNode::ConnectedCam;

class FIDTest : public testing::Test {
public:
  void SetUp() {
    graph.resize(6);

    graph[0] = {{{1, 0.5}, {2, 0.3}, {4, 0.3}, {5, 0.5}}, 4};

    graph[1] = {{{0, 0.5}, {2, 0.5}, {3, 0.3}, {5, 0.3}}, 4};

    graph[2] = {{{0, 0.3}, {1, 0.5}, {3, 0.5}, {4, 0.3}}, 4};

    graph[3] = {{{1, 0.3}, {2, 0.5}, {4, 0.5}, {5, 0.3}}, 4};

    graph[4] = {{{2, 0.3}, {3, 0.5}, {5, 0.5}, {0, 0.3}}, 4};

    graph[5] = {{{0, 0.5}, {1, 0.3}, {3, 0.3}, {4, 0.5}}, 4};
  }

protected:
  std::vector<CameraGraphNode> graph;
};

class FIDTest2 : public testing::TestWithParam<MulticameraMode> {};

TEST_F(FIDTest, FID_Precision) {
  FigSettings settings;
  settings.mode = MulticameraMode::Precision;
  FrustumIntersectionGraph fid(graph, settings);

  const auto& prim_cams = fid.primary_cameras();

  ASSERT_EQ((int)prim_cams.size(), 6);

  for (size_t cam_id = 0; cam_id < graph.size(); cam_id++) {
    const auto& sec_cams = fid.secondary_cameras(cam_id);
    for (const auto& [sec_id, fir] : graph[cam_id].stereo_camera_pairs) {
      auto it = std::find(sec_cams.begin(), sec_cams.end(), sec_id);
      ASSERT_NE(it, sec_cams.end());
    }
  }
}

TEST_F(FIDTest, FID_Moderate) {
  FigSettings settings;
  settings.mode = MulticameraMode::Moderate;
  FrustumIntersectionGraph fid(graph, settings);

  const auto& prim_cams = fid.primary_cameras();

  ASSERT_EQ((int)prim_cams.size(), 2);

  {
    const auto& connected_cams_prim_1 = graph[prim_cams[0]].stereo_camera_pairs;
    auto it = std::find_if(connected_cams_prim_1.begin(), connected_cams_prim_1.end(),
                           [&](const ConnectedCam& cam) { return cam.id == prim_cams[1]; });

    ASSERT_EQ(it, connected_cams_prim_1.end());
  }

  {
    const auto& connected_cams_prim_2 = graph[prim_cams[1]].stereo_camera_pairs;
    auto it = std::find_if(connected_cams_prim_2.begin(), connected_cams_prim_2.end(),
                           [&](const ConnectedCam& cam) { return cam.id == prim_cams[0]; });

    ASSERT_EQ(it, connected_cams_prim_2.end());
  }

  for (CameraId prim_cam : prim_cams) {
    ASSERT_EQ(fid.secondary_cameras(prim_cam).size(), graph[prim_cam].stereo_camera_pairs.size());
  }
}

TEST_F(FIDTest, FID_Performance) {
  FigSettings settings;
  settings.mode = MulticameraMode::Performance;
  FrustumIntersectionGraph fid(graph, settings);

  const auto& prim_cams = fid.primary_cameras();

  ASSERT_EQ((int)prim_cams.size(), 2);

  {
    const auto& connected_cams_prim_1 = graph[prim_cams[0]].stereo_camera_pairs;
    auto it = std::find_if(connected_cams_prim_1.begin(), connected_cams_prim_1.end(),
                           [&](const ConnectedCam& cam) { return cam.id == prim_cams[1]; });

    ASSERT_EQ(it, connected_cams_prim_1.end());
  }

  {
    const auto& connected_cams_prim_2 = graph[prim_cams[1]].stereo_camera_pairs;
    auto it = std::find_if(connected_cams_prim_2.begin(), connected_cams_prim_2.end(),
                           [&](const ConnectedCam& cam) { return cam.id == prim_cams[0]; });

    ASSERT_EQ(it, connected_cams_prim_2.end());
  }

  std::unordered_map<CameraId, std::vector<CameraId>> primary_from_secondary;
  for (CameraId prim_cam : prim_cams) {
    for (CameraId sec : fid.secondary_cameras(prim_cam)) {
      primary_from_secondary[sec].push_back(prim_cam);
    }
  }

  for (const auto& [sec_cam_id, prim_cams] : primary_from_secondary) {
    ASSERT_EQ((int)prim_cams.size(), 1);
    CameraId max_prim_cam = prim_cams[0];

    auto nodes = graph[sec_cam_id].stereo_camera_pairs;

    auto it = std::remove_if(nodes.begin(), nodes.end(), [&](const ConnectedCam& cam) {
      return std::find(prim_cams.begin(), prim_cams.end(), cam.id) == prim_cams.end();
    });
    nodes.erase(it, nodes.end());

    std::vector<float> firs;
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(firs),
                   [](const ConnectedCam& cam) { return cam.frustrim_intersection_ratio; });

    auto max_it = std::max_element(firs.begin(), firs.end());
    size_t argmax = std::distance(firs.begin(), max_it);

    ASSERT_EQ(nodes[argmax].id, max_prim_cam);
  }
}

TEST_F(FIDTest, ManualSetupChecks) {
  auto make_settings = [](const MulticamManualSetup& manual) {
    return FigSettings{MulticameraMode::Manual, /*depth_ids=*/{0}, /*allow_stereo_track_for_depth=*/false, manual};
  };

  {
    FigSettings settings_no_manual;
    settings_no_manual.mode = MulticameraMode::Manual;
    EXPECT_THROW(FrustumIntersectionGraph(graph, settings_no_manual), std::runtime_error);
  }

  MulticamManualSetup full_setup{{1, 2, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 4}, {1, 2, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 4}};
  EXPECT_NO_THROW(FrustumIntersectionGraph(graph, make_settings(full_setup)));

  MulticamManualSetup valid_setup{{1, 2}, {0, 2}, {}, {4}, {}, {0}};
  EXPECT_NO_THROW(FrustumIntersectionGraph(graph, make_settings(valid_setup)));

  MulticamManualSetup too_short{{1}, {2}, {3}};
  EXPECT_THROW(FrustumIntersectionGraph(graph, make_settings(too_short)), std::runtime_error);

  MulticamManualSetup too_long{{1}, {2}, {3}, {4}, {5}, {0}, {0} /*!*/};
  EXPECT_THROW(FrustumIntersectionGraph(graph, make_settings(too_long)), std::runtime_error);

  MulticamManualSetup wrong_id{{1}, {2}, {3}, {4}, {5}, {6 /*!*/}};
  EXPECT_THROW(FrustumIntersectionGraph(graph, make_settings(wrong_id)), std::runtime_error);

  MulticamManualSetup link_itself{{0 /*!*/}, {2}, {3}, {4}, {5}, {0}};
  EXPECT_THROW(FrustumIntersectionGraph(graph, make_settings(link_itself)), std::runtime_error);

  MulticamManualSetup link_not_in_graph{{3 /*!*/}, {2}, {3}, {4}, {5}, {0}};
  // only a warning is printed
  EXPECT_NO_THROW(FrustumIntersectionGraph(graph, make_settings(link_not_in_graph)));
}

std::vector<CameraId> PrimaryCameras(const MulticamManualSetup& setup) {
  std::vector<CameraId> primary;
  for (size_t cam_id = 0; cam_id < setup.size(); cam_id++) {
    if (!setup[cam_id].empty()) primary.push_back(cam_id);
  }
  return primary;
}

TEST_F(FIDTest, DISABLED_ManualValidSetups) {
  std::vector<MulticamManualSetup> cases{
      {{1, 2, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 4}, {1, 2, 4, 5}, {0, 2, 3, 5}, {0, 1, 3, 4}},
      {{1}, {2}, {3}, {4}, {5}, {0}},
      {{1}, {}, {3}, {}, {5}, {}},
      {{1, 2}, {}, {}, {4, 5}, {}, {}},
      {{1, 2, 3}, {}, {}, {4, 5, 0}, {}, {}},  //  0-3 link not in the graph
  };
  for (size_t i = 0; i < cases.size(); i++) {
    SCOPED_TRACE("Setup " + std::to_string(i));
    const auto& setup = cases[i];
    FrustumIntersectionGraph fig{graph, FigSettings{MulticameraMode::Manual, /*depth_ids=*/{0},
                                                    /*allow_stereo_track_for_depth=*/false, setup}};
    EXPECT_EQ(fig.primary_cameras(), PrimaryCameras(setup));
    for (size_t cam_id = 0; cam_id < setup.size(); cam_id++) {
      if (setup[cam_id].empty()) {
        EXPECT_THROW(fig.secondary_cameras(cam_id), std::out_of_range);
      } else {
        EXPECT_EQ(fig.secondary_cameras(cam_id), setup[cam_id]);
      }
    }
  }
}

TEST_P(FIDTest2, DISABLED_SimulatorEdex) {
  MulticameraMode mode = GetParam();
  // Test code using the data
  cuvslam::Trace::SetVerbosity(cuvslam::Trace::Verbosity::Debug);

  const std::string testDataFolder = CUVSLAM_TEST_ASSETS;
  const std::string edex_path = testDataFolder + "fig.edex";

  camera_rig_edex::CameraRigEdex camera_rig(edex_path);
  ASSERT_TRUE(camera_rig.start() == ErrorCode::S_True);

  camera::Rig rig;
  rig.num_cameras = static_cast<int32_t>(camera_rig.getCamerasNum());
  for (int32_t i = 0; i < rig.num_cameras; i++) {
    rig.camera_from_rig[i] = camera_rig.getExtrinsic(i).inverse();
    rig.intrinsics[i] = &(camera_rig.getIntrinsic(i));
  }
  camera::FigSettings settings;
  settings.mode = mode;
  camera::FrustumIntersectionGraph fig(rig, settings);

  ASSERT_FALSE(fig.is_valid());
}

INSTANTIATE_TEST_SUITE_P(FIDTest, FIDTest2,
                         testing::Values(MulticameraMode::Performance, MulticameraMode::Moderate,
                                         MulticameraMode::Precision));

}  // namespace Test
