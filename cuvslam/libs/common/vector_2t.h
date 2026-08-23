
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
#include <vector>

#include "Eigen/Core"

#include "common/camera_id.h"
#include "common/frame_id.h"
#include "common/track_id.h"

namespace cuvslam {

using Vector2T = Eigen::Matrix<float, 2, 1>;

using TrackIdVector2TPair = std::pair<TrackId, Vector2T>;
using TrackIdVector2TPairVector = std::vector<TrackIdVector2TPair>;

using Vector2TVector = std::vector<Vector2T>;
using Vector2TVectorVector = std::vector<Vector2TVector>;
using Vector2TPair = std::pair<Vector2T, Vector2T>;
using Vector2TPairVector = std::vector<Vector2TPair>;

using Vector2TPairVectorCIt = Vector2TPairVector::const_iterator;
using Vector2TVectorCIt = Vector2TVector::const_iterator;
using Tracks2DMap = std::map<TrackId, Vector2T>;
using Tracks2DFrameMap = std::map<FrameId, Tracks2DMap>;
using Tracks2DVectorsMap = std::map<FrameId, TrackIdVector2TPairVector>;
using Tracks2DCIt = Tracks2DMap::const_iterator;

struct Track2D {
  CameraId cam_id;
  TrackId track_id;
  Vector2T uv;
};

inline bool operator<(const Vector2T &l, const Vector2T &r) { return l.x() < r.x() && l.y() < r.y(); }
inline bool operator<=(const Vector2T &l, const Vector2T &r) { return l.x() <= r.x() && l.y() <= r.y(); }
inline bool operator>(const Vector2T &l, const Vector2T &r) { return l.x() > r.x() && l.y() > r.y(); }
inline bool operator>=(const Vector2T &l, const Vector2T &r) { return l.x() >= r.x() && l.y() >= r.y(); }

}  // namespace cuvslam
