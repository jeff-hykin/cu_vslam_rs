
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
#include <vector>

#include "common/include_gtest.h"
#include "common/types.h"
#include "common/vector_3t.h"
#include "map/map.h"
#include "pipelines/track.h"

namespace test::math {
using namespace cuvslam;
using namespace cuvslam::map;

using namespace std::chrono_literals;

TEST(Map, AddKeyframe) {
  size_t map_size = 300;
  UnifiedMap map(map_size);

  const int num_obs_per_keyframe = 500;
  const float new_tracks_ratio = 0.5;

  TrackId id = 0;

  std::list<camera::Observation> tracks;

  int64_t start_time_ns = 0;
  constexpr int64_t time_delta = 2 * 1e9;  // 2 sec

  int64_t curr_time = start_time_ns;

  const int M = static_cast<int>(num_obs_per_keyframe * new_tracks_ratio);

  for (size_t i = 0; i < 2 * map.capacity(); i++) {
    std::vector<pipelines::Landmark> triangulated_tracks;

    std::vector<camera::Observation> new_tracks;
    while (tracks.size() + new_tracks.size() < num_obs_per_keyframe) {
      new_tracks.emplace_back((uint8_t)0, id, Vector2T::Zero(), Matrix2T::Zero());

      triangulated_tracks.push_back({id, Vector3T::Zero()});
      id++;
    }

    std::copy(new_tracks.begin(), new_tracks.end(), std::back_inserter(tracks));

    std::vector<camera::Observation> obs(tracks.begin(), tracks.end());

    auto start = std::chrono::high_resolution_clock::now();
    map.add_keyframe(curr_time, {}, {}, obs, triangulated_tracks);
    auto duration_basic =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    // std::cout << "dur = " << duration_basic.count() << "[mu s]" << std::endl;

    curr_time += time_delta;

    auto max_it = std::next(tracks.begin(), M);

    tracks.erase(tracks.begin(), max_it);
  }

  ASSERT_TRUE(map.size() == map.capacity());
}

}  // namespace test::math
