#include "cuvslam_shim.h"

#include <cuvslam/cuvslam2.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

void write_error(char* error_message, int32_t capacity, const char* text) {
    if (error_message == nullptr || capacity <= 0) {
        return;
    }
    std::snprintf(error_message, static_cast<size_t>(capacity), "%s", text);
}

cuvslam::Pose to_cuvslam_pose(const CuvPose& pose) {
    cuvslam::Pose result;
    std::copy(pose.rotation_xyzw, pose.rotation_xyzw + 4, result.rotation.begin());
    std::copy(pose.translation, pose.translation + 3, result.translation.begin());
    return result;
}

cuvslam::Image to_cuvslam_image(const CuvImage& image) {
    cuvslam::Image result{};
    result.pixels = image.pixels;
    result.width = image.width;
    result.height = image.height;
    result.pitch = 0;
    result.encoding = static_cast<cuvslam::ImageData::Encoding>(image.encoding);
    result.data_type = static_cast<cuvslam::ImageData::DataType>(image.data_type);
    result.is_gpu_mem = false;
    result.timestamp_ns = image.timestamp_ns;
    result.camera_index = image.camera_index;
    return result;
}

}  // namespace

struct CuvTracker {
    cuvslam::Odometry odometry;
};

extern "C" {

int32_t cuv_tracker_create(const CuvCamera* cameras, int32_t camera_count, const CuvImuCalibration* imu_or_null,
                           const CuvConfig* config, CuvTracker** out_tracker, char* error_message,
                           int32_t error_message_capacity) {
    try {
        cuvslam::Rig rig;
        for (int32_t index = 0; index < camera_count; index++) {
            const CuvCamera& source = cameras[index];
            cuvslam::Camera camera;
            camera.size = {source.width, source.height};
            camera.principal = {source.principal[0], source.principal[1]};
            camera.focal = {source.focal[0], source.focal[1]};
            camera.rig_from_camera = to_cuvslam_pose(source.rig_from_camera);
            camera.distortion.model = static_cast<cuvslam::Distortion::Model>(source.distortion_model);
            camera.distortion.parameters.assign(source.distortion_parameters,
                                                source.distortion_parameters + source.distortion_parameter_count);
            rig.cameras.push_back(camera);
        }
        if (imu_or_null != nullptr) {
            cuvslam::ImuCalibration imu;
            imu.rig_from_imu = to_cuvslam_pose(imu_or_null->rig_from_imu);
            imu.gyroscope_noise_density = imu_or_null->gyroscope_noise_density;
            imu.gyroscope_random_walk = imu_or_null->gyroscope_random_walk;
            imu.accelerometer_noise_density = imu_or_null->accelerometer_noise_density;
            imu.accelerometer_random_walk = imu_or_null->accelerometer_random_walk;
            imu.frequency = imu_or_null->frequency;
            rig.imus.push_back(imu);
        }

        cuvslam::Odometry::Config cuvslam_config = cuvslam::Odometry::GetDefaultConfig();
        cuvslam_config.odometry_mode = static_cast<cuvslam::Odometry::OdometryMode>(config->odometry_mode);
        cuvslam_config.use_gpu = config->use_gpu;
        // Async SBA throws from cuVSLAM's own thread on failure, which would crash the process.
        cuvslam_config.async_sba = false;
        cuvslam_config.rectified_stereo_camera = config->rectified_stereo_camera;
        cuvslam_config.rgbd_settings.depth_scale_factor = config->rgbd_depth_scale_factor;
        cuvslam_config.rgbd_settings.depth_camera_id = config->rgbd_depth_camera_id;

        *out_tracker = new CuvTracker{cuvslam::Odometry(rig, cuvslam_config)};
        return 0;
    } catch (const std::exception& error) {
        write_error(error_message, error_message_capacity, error.what());
        return 1;
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown C++ exception");
        return 1;
    }
}

int32_t cuv_tracker_track(CuvTracker* tracker, const CuvImage* images, int32_t image_count, const CuvImage* depths,
                          int32_t depth_count, CuvPoseEstimate* out_estimate, char* error_message,
                          int32_t error_message_capacity) {
    try {
        cuvslam::Odometry::ImageSet image_set;
        for (int32_t index = 0; index < image_count; index++) {
            image_set.push_back(to_cuvslam_image(images[index]));
        }
        cuvslam::Odometry::ImageSet depth_set;
        for (int32_t index = 0; index < depth_count; index++) {
            depth_set.push_back(to_cuvslam_image(depths[index]));
        }

        const cuvslam::PoseEstimate estimate = tracker->odometry.Track(image_set, {}, depth_set);

        std::memset(out_estimate, 0, sizeof(*out_estimate));
        out_estimate->timestamp_ns = estimate.timestamp_ns;
        out_estimate->has_pose = estimate.world_from_rig.has_value();
        if (estimate.world_from_rig.has_value()) {
            const cuvslam::PoseWithCovariance& pose_with_covariance = *estimate.world_from_rig;
            std::copy(pose_with_covariance.pose.rotation.begin(), pose_with_covariance.pose.rotation.end(),
                      out_estimate->world_from_rig.rotation_xyzw);
            std::copy(pose_with_covariance.pose.translation.begin(), pose_with_covariance.pose.translation.end(),
                      out_estimate->world_from_rig.translation);
            std::copy(pose_with_covariance.covariance_xyz_rpy.begin(), pose_with_covariance.covariance_xyz_rpy.end(),
                      out_estimate->covariance_xyz_rpy);
        }
        return 0;
    } catch (const std::exception& error) {
        write_error(error_message, error_message_capacity, error.what());
        return 1;
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown C++ exception");
        return 1;
    }
}

int32_t cuv_tracker_register_imu(CuvTracker* tracker, const CuvImuMeasurement* measurement, char* error_message,
                                 int32_t error_message_capacity) {
    try {
        cuvslam::ImuMeasurement imu{};
        imu.timestamp_ns = measurement->timestamp_ns;
        std::copy(measurement->linear_accelerations, measurement->linear_accelerations + 3,
                  imu.linear_accelerations.begin());
        std::copy(measurement->angular_velocities, measurement->angular_velocities + 3,
                  imu.angular_velocities.begin());
        tracker->odometry.RegisterImuMeasurement(0, imu);
        return 0;
    } catch (const std::exception& error) {
        write_error(error_message, error_message_capacity, error.what());
        return 1;
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown C++ exception");
        return 1;
    }
}

void cuv_tracker_destroy(CuvTracker* tracker) { delete tracker; }
}
