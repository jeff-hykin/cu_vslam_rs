// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: LicenseRef-NVIDIA-Community
//
// Raw bindings to shim/cuvslam_shim.h, the extern-"C" wrapper over the cuVSLAM C++ API.

use std::os::raw::{c_char, c_void};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CuvPose {
    pub rotation_xyzw: [f32; 4],
    pub translation: [f32; 3],
}

impl Default for CuvPose {
    fn default() -> Self {
        Self {
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
        }
    }
}

pub const CUV_DISTORTION_PINHOLE: u8 = 0;
pub const CUV_DISTORTION_BROWN: u8 = 2;

#[repr(C)]
pub struct CuvCamera {
    pub width: i32,
    pub height: i32,
    pub principal: [f32; 2],
    pub focal: [f32; 2],
    pub rig_from_camera: CuvPose,
    pub distortion_model: u8,
    pub distortion_parameters: *const f32,
    pub distortion_parameter_count: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CuvImuCalibration {
    pub rig_from_imu: CuvPose,
    pub gyroscope_noise_density: f32,
    pub gyroscope_random_walk: f32,
    pub accelerometer_noise_density: f32,
    pub accelerometer_random_walk: f32,
    pub frequency: f32,
}

pub const CUV_ODOMETRY_MULTICAMERA: u8 = 0;
pub const CUV_ODOMETRY_INERTIAL: u8 = 1;
pub const CUV_ODOMETRY_RGBD: u8 = 2;
pub const CUV_ODOMETRY_MONO: u8 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CuvConfig {
    pub odometry_mode: u8,
    pub use_gpu: bool,
    pub rectified_stereo_camera: bool,
    pub rgbd_depth_scale_factor: f32,
    pub rgbd_depth_camera_id: i32,
}

pub const CUV_ENCODING_MONO: u8 = 0;
pub const CUV_ENCODING_RGB: u8 = 1;
pub const CUV_DATA_UINT8: u8 = 0;
pub const CUV_DATA_UINT16: u8 = 1;

#[repr(C)]
pub struct CuvImage {
    pub pixels: *const c_void,
    pub width: i32,
    pub height: i32,
    pub encoding: u8,
    pub data_type: u8,
    pub timestamp_ns: i64,
    pub camera_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CuvImuMeasurement {
    pub timestamp_ns: i64,
    pub linear_accelerations: [f32; 3],
    pub angular_velocities: [f32; 3],
}

#[repr(C)]
pub struct CuvPoseEstimate {
    pub timestamp_ns: i64,
    pub has_pose: bool,
    pub world_from_rig: CuvPose,
    pub covariance_xyz_rpy: [f64; 36],
}

#[repr(C)]
pub struct CuvTracker {
    _private: [u8; 0],
}

extern "C" {
    pub fn cuv_tracker_create(
        cameras: *const CuvCamera,
        camera_count: i32,
        imu_or_null: *const CuvImuCalibration,
        config: *const CuvConfig,
        out_tracker: *mut *mut CuvTracker,
        error_message: *mut c_char,
        error_message_capacity: i32,
    ) -> i32;

    pub fn cuv_tracker_track(
        tracker: *mut CuvTracker,
        images: *const CuvImage,
        image_count: i32,
        depths: *const CuvImage,
        depth_count: i32,
        out_estimate: *mut CuvPoseEstimate,
        error_message: *mut c_char,
        error_message_capacity: i32,
    ) -> i32;

    pub fn cuv_tracker_register_imu(
        tracker: *mut CuvTracker,
        measurement: *const CuvImuMeasurement,
        error_message: *mut c_char,
        error_message_capacity: i32,
    ) -> i32;

    pub fn cuv_tracker_destroy(tracker: *mut CuvTracker);
}
