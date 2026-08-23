// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: LicenseRef-NVIDIA-Community
//
// Safe wrapper around the cuVSLAM shim: owns the tracker handle, turns error
// buffers into Result<_, String>, keeps the raw pointers behind slices.

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};

use crate::ffi;

const ERROR_CAPACITY: usize = 512;

pub struct CameraParams {
    pub width: i32,
    pub height: i32,
    pub principal: [f32; 2],
    pub focal: [f32; 2],
    pub rig_from_camera: ffi::CuvPose,
    pub distortion_model: u8,
    pub distortion_parameters: Vec<f32>,
}

pub struct ImageRef<'a> {
    pub pixels: &'a [u8],
    pub width: i32,
    pub height: i32,
    pub encoding: u8,
    pub data_type: u8,
    pub timestamp_ns: i64,
    pub camera_index: u32,
}

pub struct PoseEstimate {
    pub timestamp_ns: i64,
    /// `None` while tracking is lost.
    pub world_from_rig: Option<(ffi::CuvPose, [f64; 36])>,
}

pub struct Tracker {
    raw: *mut ffi::CuvTracker,
}

// The module serializes all tracker calls on its one event loop; cuVSLAM itself
// only forbids concurrent calls, not cross-thread ownership.
unsafe impl Send for Tracker {}

fn error_string(buffer: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn to_ffi_image(image: &ImageRef) -> ffi::CuvImage {
    ffi::CuvImage {
        pixels: image.pixels.as_ptr() as *const c_void,
        width: image.width,
        height: image.height,
        encoding: image.encoding,
        data_type: image.data_type,
        timestamp_ns: image.timestamp_ns,
        camera_index: image.camera_index,
    }
}

impl Tracker {
    pub fn new(
        cameras: &[CameraParams],
        imu: Option<&ffi::CuvImuCalibration>,
        config: &ffi::CuvConfig,
    ) -> Result<Self, String> {
        let ffi_cameras: Vec<ffi::CuvCamera> = cameras
            .iter()
            .map(|camera| ffi::CuvCamera {
                width: camera.width,
                height: camera.height,
                principal: camera.principal,
                focal: camera.focal,
                rig_from_camera: camera.rig_from_camera,
                distortion_model: camera.distortion_model,
                distortion_parameters: camera.distortion_parameters.as_ptr(),
                distortion_parameter_count: camera.distortion_parameters.len() as i32,
            })
            .collect();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let mut raw: *mut ffi::CuvTracker = std::ptr::null_mut();
        let status = unsafe {
            ffi::cuv_tracker_create(
                ffi_cameras.as_ptr(),
                ffi_cameras.len() as i32,
                imu.map_or(std::ptr::null(), |calibration| calibration as *const _),
                config,
                &mut raw,
                error.as_mut_ptr(),
                ERROR_CAPACITY as i32,
            )
        };
        if status != 0 {
            return Err(error_string(&error));
        }
        Ok(Self { raw })
    }

    pub fn track(&mut self, images: &[ImageRef], depths: &[ImageRef]) -> Result<PoseEstimate, String> {
        let ffi_images: Vec<ffi::CuvImage> = images.iter().map(to_ffi_image).collect();
        let ffi_depths: Vec<ffi::CuvImage> = depths.iter().map(to_ffi_image).collect();
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let mut estimate = ffi::CuvPoseEstimate {
            timestamp_ns: 0,
            has_pose: false,
            world_from_rig: ffi::CuvPose::default(),
            covariance_xyz_rpy: [0.0; 36],
        };
        let status = unsafe {
            ffi::cuv_tracker_track(
                self.raw,
                ffi_images.as_ptr(),
                ffi_images.len() as i32,
                ffi_depths.as_ptr(),
                ffi_depths.len() as i32,
                &mut estimate,
                error.as_mut_ptr(),
                ERROR_CAPACITY as i32,
            )
        };
        if status != 0 {
            return Err(error_string(&error));
        }
        Ok(PoseEstimate {
            timestamp_ns: estimate.timestamp_ns,
            world_from_rig: estimate
                .has_pose
                .then_some((estimate.world_from_rig, estimate.covariance_xyz_rpy)),
        })
    }

    pub fn register_imu(&mut self, measurement: &ffi::CuvImuMeasurement) -> Result<(), String> {
        let mut error = [0 as c_char; ERROR_CAPACITY];
        let status = unsafe {
            ffi::cuv_tracker_register_imu(self.raw, measurement, error.as_mut_ptr(), ERROR_CAPACITY as i32)
        };
        if status != 0 {
            return Err(error_string(&error));
        }
        Ok(())
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        unsafe { ffi::cuv_tracker_destroy(self.raw) };
    }
}
