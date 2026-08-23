
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

#include <memory>

#include "common/isometry_utils.h"
#include "sof/internal/sof_multicamera_gpu.h"
#include "sof/sof_create.h"

namespace cuvslam::sof {

std::unique_ptr<GPULKFeatureTracker> CreateGPUTracker(TrackerType type) {
  switch (type) {
    case TrackerType::LK:
      return std::make_unique<GPULKFeatureTracker>();
    case TrackerType::LKHorizontal:
      return std::make_unique<GPULKTrackerHorizontal>();
    default:
      TraceError("Unsupported GPU tracker type");
      return nullptr;
  }
}

MultiSOFGPU::MultiSOFGPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                         FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                         const odom::KeyFrameSettings& keyframe_settings)
    : MultiSOFBase(rig, fid, sof_settings, keyframe_settings), sof_settings_(sof_settings) {
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_cam_id : primary_cams) {
    const camera::ICameraModel& intrinsics = *rig_.intrinsics[primary_cam_id];
    auto selector = std::make_unique<SelectorStereo>(sof_settings.feature_selection_settings);
    mono_sof_.emplace_back(CreateMonoSOF(Implementation::kGPU, primary_cam_id, intrinsics, std::move(selector),
                                         feature_predictor, sof_settings));

    const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);

    auto& tracker_from_secondary_cam = secondary_from_primary_sof_[primary_cam_id];
    for (CameraId secondary_cam_id : secondary_cams) {
      auto tracker_ptr = CreateGPUTracker(sof_settings.lr_tracker);
      tracker_from_secondary_cam[secondary_cam_id].tracker = std::move(tracker_ptr);
    }
  }
}

void MultiSOFGPU::LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id,
                                                   const Sources& curr_sources, Images& curr_images,
                                                   const std::vector<camera::Observation>& primary_obs,
                                                   std::vector<camera::Observation>* /* secondary_obs */) {
  const ImageContextPtr primary_image = curr_images[primary_id];
  const ImageSource& secondary_source = curr_sources[secondary_id];
  const ImageContextPtr secondary_image = curr_images[secondary_id];

  const camera::ICameraModel& intrinsicsP = *rig_.intrinsics[primary_id];
  const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

  PrimaryToSecondaryGPUTracker& tracker = secondary_from_primary_sof_[primary_id][secondary_id];

  GPUArrayPinned<TrackData>& tracks_data = tracker.tracks_data;
  Stream& stream = tracker.stream;

  const Isometry3T secondary_from_primary =
      rig_.camera_from_rig[secondary_id] * rig_.camera_from_rig[primary_id].inverse();

  const float baseline = secondary_from_primary.translation().norm();
  const float avg_focal = 0.5f * (intrinsicsS.getFocal().x() + intrinsicsS.getFocal().y());
  const float cross_cam_search_radius = std::max(20.f, baseline * avg_focal * 2.f);

  for (size_t i = 0; i < primary_obs.size(); i++) {
    const camera::Observation& trackL = primary_obs[i];
    const Vector2T& xyL = trackL.xy;

    TrackData& data = tracks_data[i];
    Vector2T uvL;
    intrinsicsP.denormalizePoint(xyL, uvL);

    const Vector3T ray_in_secondary = secondary_from_primary.linear() * xyL.homogeneous();
    const float z = ray_in_secondary.z();

    data.track = {uvL.x(), uvL.y()};
    data.track_status = false;
    data.ncc_threshold = 0.8f;

    if (z < 1e-8) {
      data.offset = {0.f, 0.f};
      data.search_radius_px = 0.f;
      continue;
    }

    const Vector2T xyR_init(ray_in_secondary.x() / z, ray_in_secondary.y() / z);
    Vector2T uvR_init;
    intrinsicsS.denormalizePoint(xyR_init, uvR_init);

    const Vector2T offset = uvR_init - uvL;
    data.offset = {offset.x(), offset.y()};
    data.search_radius_px = cross_cam_search_radius;
  }

  tracks_data.copy_top_n(ToGPU, primary_obs.size(), stream.get_stream());

  secondary_image->build_gpu_image_pyramid(secondary_source, box_prefilter_, stream.get_stream());
  secondary_image->build_gpu_gradient_pyramid(true, stream.get_stream());

  tracker.tracker->track_points(primary_image->gpu_gradient_pyramid(), secondary_image->gpu_gradient_pyramid(),
                                primary_image->gpu_image_pyramid(), secondary_image->gpu_image_pyramid(), tracks_data,
                                primary_obs.size(), stream.get_stream());
  tracks_data.copy_top_n(ToCPU, primary_obs.size(), stream.get_stream());
  tracker.was_launched = true;
}

void MultiSOFGPU::GetTrackingResults(MulticamObservations& observations) {
  std::vector<std::vector<camera::Observation>> secondary_observations(observations.size());
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_id : primary_cams) {
    if (primary_id >= observations.size()) {
      continue;
    }
    const std::vector<camera::Observation>& primary_obs = observations[primary_id];
    const auto& secondary_cams = fid_.secondary_cameras(primary_id);

    for (CameraId secondary_id : secondary_cams) {
      PrimaryToSecondaryGPUTracker& tracker = secondary_from_primary_sof_[primary_id][secondary_id];
      if (!tracker.was_launched) {
        continue;
      }

      GPUArrayPinned<TrackData>& tracks_data = tracker.tracks_data;
      Stream& stream = tracker.stream;

      const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

      cudaStreamSynchronize(stream.get_stream());

      Matrix2T info;
      Vector2T xyR;
      Vector2T uvR;
      // secondary observations can be added, so only iterate over primary observations
      // FIXME: tracks_data.size() should return data size and not capacity, fix GPUArray[Pinned] and get rid of
      // obs_sizes
      for (size_t i = 0; i < primary_obs.size(); i++) {
        auto& data = tracks_data[i];
        const TrackId& trackId = primary_obs[i].id;
        if (data.track_status) {
          uvR << data.track.x, data.track.y;
          info << data.info[0], data.info[1], data.info[2], data.info[3];
          intrinsicsS.normalizePoint(uvR, xyR);

          secondary_observations[secondary_id].push_back(
              {secondary_id, trackId, xyR, camera::ObservationInfoUVToXY(intrinsicsS, uvR, xyR, info)});
        }
      }
    }
  }

  for (CameraId cam_id = 0; cam_id < secondary_observations.size(); ++cam_id) {
    auto& obs_vector = secondary_observations[cam_id];
    auto& x = observations[cam_id];
    std::move(obs_vector.begin(), obs_vector.end(), std::back_inserter(x));
  }
}

void MultiSOFGPU::StartKeyframe() {
  for (auto& [_, x] : secondary_from_primary_sof_) {
    for (auto& [cam_id, tracker] : x) {
      tracker.was_launched = false;
    }
  }
}

void MultiSOFGPU::reset() {
  for (const auto& sof : mono_sof_) {
    sof->reset();
  }

  reset_keyframe_selector();
}

void MultiSOFGPU::rebuild_prev_context(CameraId cam_id, const ImageSource& source, const ImageContextPtr& ctx) {
  // Replays the pyramid builds that trackNextFrame() performed for this camera in the frame that
  // was checkpointed: primary cameras get vertical+horizontal gradients (mono path, built first),
  // pure secondary cameras only ever get the horizontal gradient (stereo path).
  Stream stream(/*sync_on_destroy=*/true);
  ctx->build_gpu_image_pyramid(source, sof_settings_.box3_prefilter, stream.get_stream());
  const bool horizontal = !is_primary_cam(cam_id);
  ctx->build_gpu_gradient_pyramid(horizontal, stream.get_stream());
}

}  // namespace cuvslam::sof
