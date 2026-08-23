
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

#include "camera_rig_edex/camera_rig_edex.h"

#include <filesystem>
#include <numeric>

#include "common/camera_id.h"
#include "common/coordinate_system.h"
#include "common/imu_measurement.h"
#include "common/include_json.h"
#include "common/log.h"
#include "common/log_types.h"
#include "utils/image_loader.h"

namespace cuvslam::camera_rig_edex {

int64_t CameraRigEdex::get_event_timestamp(const Event& event) {
  return event.type == Frame ? event.frame_metadata.camera.at(0).timestamp_ns : event.imu_measurement.time_ns;
}

CameraRigEdex::CameraRigEdex(const std::string& edexFileName, const std::vector<CameraId>& useCameras)
    : edexFileName_(edexFileName),
      seqPath_(std::filesystem::path(edexFileName).parent_path()),
      useCameras_(useCameras),
      eventIndex_(0),
      frameIndex_(0),
      replayForward_(true) {}

bool CameraRigEdex::setCurrentFrame(int frame) {
  size_t frames_counter = 0;
  for (size_t i = 0; i < events_.size(); i++) {
    if (events_[i].type == EventType::Frame) {
      if (frames_counter == static_cast<size_t>(frame)) {
        eventIndex_ = i;
        return true;
      }
      frames_counter++;
    }
  }
  return false;
}

void CameraRigEdex::setReplayForward(bool forward) {
  if (forward != replayForward_) {
    replayForward_ = forward;
    std::reverse(events_.begin(), events_.end());
    // start from the same event but in the reversed order, so we always increment the event index
    eventIndex_ = events_.size() - eventIndex_;
  }
}

bool CameraRigEdex::getReplayForward() const { return replayForward_; }

int64_t CameraRigEdex::getFirstTimestamp() const {
  const Event& start_event = events_.front();
  return start_event.type == Frame ? start_event.frame_metadata.camera.at(0).timestamp_ns
                                   : start_event.imu_measurement.time_ns;
}

int64_t CameraRigEdex::getLastTimestamp() const {
  const Event& end_event = events_.back();
  return end_event.type == Frame ? end_event.frame_metadata.camera.at(0).timestamp_ns
                                 : end_event.imu_measurement.time_ns;
}

bool CameraRigEdex::load_edex() {
  if (!edex_.read(edexFileName_)) {
    TraceError("Error loading file %s", edexFileName_.c_str());
    return false;
  }

  if (useCameras_.empty()) {
    useCameras_.resize(edex_.cameras_.size());
    std::iota(useCameras_.begin(), useCameras_.end(), 0);
  } else {
    for (auto id : useCameras_) {
      if (id >= edex_.cameras_.size()) {
        TraceError("Wrong camera id (%d) in a list of cameras to use", id);
        return false;
      }
    }
  }
  const size_t nCameras = useCameras_.size();
  cameras_.resize(nCameras);

  for (size_t i = 0; i < nCameras; ++i) {
    const edex::Camera& c = edex_.cameras_[useCameras_[i]];

    cameras_[i].intrinsic =
        camera::CreateCameraModel(c.intrinsics.resolution, c.intrinsics.focal, c.intrinsics.principal,
                                  camera::StringToDistortionModel(c.intrinsics.distortion_model),
                                  c.intrinsics.distortion_params.data(), c.intrinsics.distortion_params.size());
    cameras_[i].transform = LegacyEdexIsometryToOpenCV(c.transform);
    if (!cameras_[i].intrinsic) {
      TraceError("Wrong distortion type or number of parameters");
      return false;
    }
    if (c.depth_id.has_value()) {
      cameras_[i].depth_id = c.depth_id.value();
      cameras_[i].has_depth = true;
    }
  }

  return true;
}

ErrorCode CameraRigEdex::start() {
  if (!load_edex()) {
    return ErrorCode::E_Failure;
  }

  if (!edex_.frame_meta_log_path_.empty()) {
    const std::string frame_meta_filename = seqPath_ / edex_.frame_meta_log_path_;

    if (!read_frame_metadata(frame_meta_filename)) {
      TraceError("Can't read metadata from %s", frame_meta_filename.c_str());
      return ErrorCode::E_Failure;
    }
  } else {
    if (edex_.fps_ <= 0) {
      assert(false);
      return ErrorCode::E_Failure;
    }
    size_t sequence_size = edex_.cameras_[0].sequence.size();
    for (size_t i = 0; i < sequence_size; i++) {
      Event event;
      event.frame_metadata.frame_id = i;  // TODO: need to extract frame from filename
      for (size_t j = 0; j < useCameras_.size(); j++) {
        event.type = EventType::Frame;
        event.frame_metadata.camera[j].filename = seqPath_ / edex_.cameras_[useCameras_[j]].sequence[i];
        const auto frame = static_cast<double>(i);  // current frame number
        const double seconds = frame / edex_.fps_;  // current time in seconds
        const double nanoseconds = seconds * 1e9;   // current time in nanoseconds
        const auto timestamp = std::lround(nanoseconds);
        event.frame_metadata.camera[j].timestamp_ns = timestamp;
      }
      events_.push_back(std::move(event));
    }
  }

  if (!edex_.imu_.imu_log_path_.empty()) {
    const std::string imu_log_filename = seqPath_ / edex_.imu_.imu_log_path_;

    // optional
    const Matrix3T change_basis = CoordinateSystemTocuVSLAM(edex_.imu_.coordinate_system);
    if (!read_imu_data(imu_log_filename, change_basis)) {
      TraceError("Can't read IMU measurements from %s", imu_log_filename.c_str());
      return ErrorCode::E_Failure;
    }
  }

  std::sort(events_.begin(), events_.end(), [&](const Event& lhs, const Event& rhs) {
    auto lhs_timestamp = get_event_timestamp(lhs);
    auto rhs_timestamp = get_event_timestamp(rhs);
    return replayForward_ ? lhs_timestamp < rhs_timestamp : lhs_timestamp > rhs_timestamp;
  });
  // first event must be an image!
  {
    auto it = std::find_if(events_.begin(), events_.end(), [](const Event& e) { return e.type == Frame; });
    events_.erase(events_.begin(), it);
  }

  return ErrorCode::S_True;
}

ErrorCode CameraRigEdex::stop() { return ErrorCode::S_True; }

uint32_t CameraRigEdex::getCamerasNum() const { return cameras_.size(); }

std::vector<CameraId> CameraRigEdex::getCamerasWithDepth() const {
  std::vector<CameraId> cam_with_depth;
  for (size_t i = 0; i < cameras_.size(); ++i) {
    if (cameras_[i].has_depth) {
      cam_with_depth.push_back(static_cast<CameraId>(i));
    }
  }
  return cam_with_depth;
}

const camera::ICameraModel& CameraRigEdex::getIntrinsic(uint32_t index) const {
  assert(index <= cameras_.size());
  return *cameras_.at(index).intrinsic;
}

const Isometry3T& CameraRigEdex::getExtrinsic(uint32_t index) const {
  assert(index <= cameras_.size());
  return cameras_.at(index).transform;
}

ErrorCode CameraRigEdex::getFrame(Sources& sources, Metas& metas, Sources& masks_sources, DepthSources& depth_sources) {
  sources.assign(cameras_.size(), {});
  metas.assign(cameras_.size(), {});
  masks_sources.assign(cameras_.size(), {});
  depth_sources.assign(cameras_.size(), {});

  if (eventIndex_ >= events_.size()) {
    return ErrorCode::E_Bounds;
  }

  Event event = events_[eventIndex_];

  auto handle_imu_events = [&]() {
    while (event.type == EventType::IMU) {
      if (imuCallback_ && replayForward_) {
        imuCallback_(event.imu_measurement);
      }
      ++eventIndex_;
      if (eventIndex_ >= events_.size()) {
        return false;
      }
      event = events_[eventIndex_];
    }
    return true;
  };

  if (!handle_imu_events()) {
    return ErrorCode::E_Bounds;
  }

  log::Log<LogFrames>([&]() {
    for (const auto& metadata_per_cam : event.frame_metadata.camera) {
      log::Value<LogFrames>(("image_file" + std::to_string(metadata_per_cam.first)).c_str(),
                            metadata_per_cam.second.filename);
      log::Value<LogFrames>(("image_timestamp" + std::to_string(metadata_per_cam.first)).c_str(),
                            metadata_per_cam.second.timestamp_ns);
    }
    log::Value<LogFrames>("frame_number", static_cast<int>(event.frame_metadata.frame_id));
  });

  int i = 0;
  utils::ImageLoaderT nonTransformedImageInMemory;
  for (const auto& metadata_per_cam : event.frame_metadata.camera) {
    const auto& camera_image = metadata_per_cam.second.filename;
    CameraId cam_id = metadata_per_cam.first;
    ImageSource& source = sources[cam_id];
    ImageSource& mask_source = masks_sources[cam_id];
    ImageMeta& meta = metas[cam_id];

    meta.frame_id = frameIndex_;
    meta.frame_number = static_cast<int>(event.frame_metadata.frame_id);
    meta.camera_index = static_cast<int>(cam_id);
    source.type = ImageSource::U8;
    meta.filename = camera_image;

    if (!nonTransformedImageInMemory.load(camera_image)) {
      return ErrorCode::S_False;
    }
    const ImageMatrix<uint8_t>& matrix = nonTransformedImageInMemory.getImage();

    const size_t w = matrix.cols();
    const size_t h = matrix.rows();
    cameras_[i].data.resize(w * h);

    std::copy(matrix.data(), matrix.data() + w * h, cameras_[i].data.data());

    source.data = cameras_[i].data.data();
    meta.shape.width = static_cast<int>(w);
    meta.shape.height = static_cast<int>(h);
    // Pretend that frames are at 1 fps
    meta.timestamp = metadata_per_cam.second.timestamp_ns;

    size_t last_dot_pos = camera_image.find_last_of('.');
    std::string input_mask_filename;
    if (last_dot_pos != std::string::npos && last_dot_pos != 0) {
      input_mask_filename =
          camera_image.substr(0, last_dot_pos) + "_mask" + camera_image.substr(last_dot_pos, camera_image.size());
      meta.filename_mask = input_mask_filename;
    }
    if (std::filesystem::exists(input_mask_filename)) {
      utils::ImageLoaderT nonTransformedMaskInMemory;
      if (!nonTransformedMaskInMemory.load(input_mask_filename)) {
        return ErrorCode::S_False;
      }
      const ImageMatrix<uint8_t>& mask_matrix = nonTransformedMaskInMemory.getImage();
      const size_t mask_w = mask_matrix.cols();
      const size_t mask_h = mask_matrix.rows();
      cameras_[i].mask_data.resize(mask_w * mask_h);
      std::copy(mask_matrix.data(), mask_matrix.data() + mask_w * mask_h, cameras_[i].mask_data.data());
      mask_source.data = cameras_[i].mask_data.data();
      mask_source.type = ImageSource::U8;
      meta.mask_shape.width = static_cast<int>(mask_w);
      meta.mask_shape.height = static_cast<int>(mask_h);
    }

    if (metadata_per_cam.second.depth_filename.has_value()) {
      const auto& depth_image = metadata_per_cam.second.depth_filename.value();

      ImageSource& dsource = depth_sources[cam_id];
      dsource.type = ImageSource::F32;
      utils::DepthLoader nonTransformedDepthInMemory(depth_image.c_str());
      const ImageMatrix<float>& depth = nonTransformedDepthInMemory.getImage();
      cameras_[i].depth.resize(w * h);
      std::copy(depth.data(), depth.data() + w * h, cameras_[i].depth.data());
      dsource.data = cameras_[i].depth.data();
    }

    i++;
  }

  ++frameIndex_;
  ++eventIndex_;
  return ErrorCode::S_True;
}

bool CheckKey(const Json::Value root, const std::vector<std::string>& keys) {
  for (const auto& k : keys) {
    if (!root.isMember(k)) {
      TraceError("\"%s\" key is expected", k.c_str());
      return false;
    }
  }

  return true;
}

bool CameraRigEdex::read_frame_metadata(const std::string& filename) {
  std::ifstream frame_meta_log(filename);

  if (!frame_meta_log) {
    return false;
  }

  Json::Value root;
  std::string errors;
  for (std::string line; std::getline(frame_meta_log, line);) {
    if (!JsonUtils::readJsonFromStringNoThrow(line, root, errors) || !CheckKey(root, {"frame_id", "cams"})) {
      TraceError("%s", errors.c_str());
      return false;
    }

    Event event;
    event.type = EventType::Frame;
    event.frame_metadata.frame_id = root["frame_id"].asInt();

    for (const auto& frame_metadata_per_cam : root["cams"]) {
      if (!CheckKey(frame_metadata_per_cam, {"id", "filename", "timestamp"})) {
        TraceError("Wrong key for images in frame_metadata.jsonl");
      }
      CameraId cam_id = frame_metadata_per_cam["id"].asInt();
      auto it = std::find(useCameras_.begin(), useCameras_.end(), cam_id);
      if (it == useCameras_.end()) {
        continue;
      }

      CameraId cam = std::distance(useCameras_.begin(), it);
      event.frame_metadata.camera[cam].filename = seqPath_ / frame_metadata_per_cam["filename"].asString();
      event.frame_metadata.camera[cam].timestamp_ns = frame_metadata_per_cam["timestamp"].asInt64();
      // Note: use 'cam' (index into cameras_) not 'cam_id' (original camera ID from JSON)
      if (cameras_[cam].has_depth) {
        if (!root.isMember("depth")) {
          TraceError("No depth field in frame_metadata.jsonl");
        } else {
          for (const auto& depth_metadata_per_cam : root["depth"]) {
            if (!CheckKey(depth_metadata_per_cam, {"id", "filename", "timestamp"})) {
              TraceError("Wrong key for depth in frame_metadata.jsonl");
            }

            if (cameras_[cam].depth_id.has_value() &&
                depth_metadata_per_cam["id"].asInt() == static_cast<int>(cameras_[cam].depth_id.value())) {
              event.frame_metadata.camera[cam].depth_id = cameras_[cam].depth_id;
              event.frame_metadata.camera[cam].depth_filename =
                  seqPath_ / depth_metadata_per_cam["filename"].asString();
            }
          }
        }
      }
    }
    events_.push_back(std::move(event));
  }
  return true;
}

bool CameraRigEdex::read_imu_data(const std::string& filename, const Matrix3T& transform) {
  assert(Eigen::internal::isApprox(transform.determinant(), 1.f, 0.001f));

  std::ifstream frame_meta_log(filename);
  if (!frame_meta_log) {
    return false;
  }

  Json::Value root;
  std::string errors;
  for (std::string line; std::getline(frame_meta_log, line);) {
    if (!JsonUtils::readJsonFromStringNoThrow(line, root, errors) ||
        !CheckKey(root, {"AngularVelocityX", "AngularVelocityY", "AngularVelocityZ", "LinearAccelerationX",
                         "LinearAccelerationY", "LinearAccelerationZ", "timestamp"})) {
      TraceError("%s", errors.c_str());
      return false;
    }

    Event event;
    event.type = EventType::IMU;
    event.imu_measurement.angular_velocity = Vector3T{
        root["AngularVelocityX"].asFloat(),
        root["AngularVelocityY"].asFloat(),
        root["AngularVelocityZ"].asFloat(),
    };
    event.imu_measurement.linear_acceleration = Vector3T{
        root["LinearAccelerationX"].asFloat(),
        root["LinearAccelerationY"].asFloat(),
        root["LinearAccelerationZ"].asFloat(),
    };
    event.imu_measurement.angular_velocity = transform * event.imu_measurement.angular_velocity;
    event.imu_measurement.linear_acceleration = transform * event.imu_measurement.linear_acceleration;

    event.imu_measurement.time_ns = root["timestamp"].asInt64();
    events_.push_back(std::move(event));
  }
  return true;
}

void CameraRigEdex::registerIMUCallback(const std::function<void(const imu::ImuMeasurement& integrator)>& func) {
  imuCallback_ = func;
}

}  // namespace cuvslam::camera_rig_edex
