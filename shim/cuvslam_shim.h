// Thin extern-"C" shim over the cuVSLAM C++ API (cuvslam2.h).
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CuvTracker CuvTracker;

typedef struct {
    float rotation_xyzw[4];
    float translation[3];
} CuvPose;

// distortion_model values match cuvslam::Distortion::Model
enum {
    CUV_DISTORTION_PINHOLE = 0,
    CUV_DISTORTION_FISHEYE = 1,
    CUV_DISTORTION_BROWN = 2,
    CUV_DISTORTION_POLYNOMIAL = 3,
};

typedef struct {
    int32_t width;
    int32_t height;
    float principal[2];
    float focal[2];
    CuvPose rig_from_camera;
    uint8_t distortion_model;
    const float* distortion_parameters;
    int32_t distortion_parameter_count;
} CuvCamera;

typedef struct {
    CuvPose rig_from_imu;
    float gyroscope_noise_density;
    float gyroscope_random_walk;
    float accelerometer_noise_density;
    float accelerometer_random_walk;
    float frequency;
} CuvImuCalibration;

// odometry_mode values match cuvslam::Odometry::OdometryMode
enum {
    CUV_ODOMETRY_MULTICAMERA = 0,
    CUV_ODOMETRY_INERTIAL = 1,
    CUV_ODOMETRY_RGBD = 2,
    CUV_ODOMETRY_MONO = 3,
};

typedef struct {
    uint8_t odometry_mode;
    bool use_gpu;
    bool rectified_stereo_camera;
    float rgbd_depth_scale_factor;
    int32_t rgbd_depth_camera_id;
} CuvConfig;

// encoding values match cuvslam::ImageData::Encoding, data_type matches DataType
enum {
    CUV_ENCODING_MONO = 0,
    CUV_ENCODING_RGB = 1,
};
enum {
    CUV_DATA_UINT8 = 0,
    CUV_DATA_UINT16 = 1,
    CUV_DATA_FLOAT32 = 2,
};

typedef struct {
    const void* pixels;
    int32_t width;
    int32_t height;
    uint8_t encoding;
    uint8_t data_type;
    int64_t timestamp_ns;
    uint32_t camera_index;
} CuvImage;

typedef struct {
    int64_t timestamp_ns;
    float linear_accelerations[3];
    float angular_velocities[3];
} CuvImuMeasurement;

typedef struct {
    int64_t timestamp_ns;
    bool has_pose;
    CuvPose world_from_rig;
    double covariance_xyz_rpy[36];  // row-major 6x6, valid only when has_pose
} CuvPoseEstimate;

// All functions return 0 on success. On failure they return nonzero and write a
// NUL-terminated message into error_message (truncated to error_message_capacity).

int32_t cuv_tracker_create(const CuvCamera* cameras, int32_t camera_count, const CuvImuCalibration* imu_or_null,
                           const CuvConfig* config, CuvTracker** out_tracker, char* error_message,
                           int32_t error_message_capacity);

int32_t cuv_tracker_track(CuvTracker* tracker, const CuvImage* images, int32_t image_count, const CuvImage* depths,
                          int32_t depth_count, CuvPoseEstimate* out_estimate, char* error_message,
                          int32_t error_message_capacity);

int32_t cuv_tracker_register_imu(CuvTracker* tracker, const CuvImuMeasurement* measurement, char* error_message,
                                 int32_t error_message_capacity);

void cuv_tracker_destroy(CuvTracker* tracker);

#ifdef __cplusplus
}
#endif
