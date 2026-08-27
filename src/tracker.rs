// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: LicenseRef-NVIDIA-Community
//
// Safe wrapper around the cuVSLAM shim.

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
    odometry_mode: u8,
}

// cuVSLAM forbids concurrent calls, not cross-thread ownership.
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

/// cuVSLAM takes the rig and the mode separately and reconciles neither, so every mismatch
/// here builds a tracker that runs and quietly ignores part of what it was given.
fn check_rig(
    cameras: &[CameraParams],
    imu: Option<&ffi::CuvImuCalibration>,
    config: &ffi::CuvConfig,
) -> Result<(), String> {
    let minimum_cameras = match config.odometry_mode {
        ffi::CUV_ODOMETRY_MONO | ffi::CUV_ODOMETRY_RGBD => 1,
        ffi::CUV_ODOMETRY_MULTICAMERA | ffi::CUV_ODOMETRY_INERTIAL => 2,
        other => {
            return Err(format!(
                "odometry_mode {other} is not one of cuVSLAM's four modes"
            ))
        }
    };
    if cameras.len() < minimum_cameras {
        return Err(format!(
            "odometry_mode {} needs at least {minimum_cameras} cameras, got {}",
            config.odometry_mode,
            cameras.len()
        ));
    }
    // The calibration reaches the rig either way, but only Inertial reads it.
    if imu.is_some() != (config.odometry_mode == ffi::CUV_ODOMETRY_INERTIAL) {
        return Err(format!(
            "an IMU calibration is used by odometry_mode {} alone, and is required there; \
             got mode {} with imu {}",
            ffi::CUV_ODOMETRY_INERTIAL,
            config.odometry_mode,
            if imu.is_some() { "set" } else { "unset" }
        ));
    }
    if config.odometry_mode == ffi::CUV_ODOMETRY_RGBD {
        // Its -1 default is in range for cuVSLAM and means no camera, so depth is dropped.
        if !(0..cameras.len() as i32).contains(&config.rgbd_depth_camera_id) {
            return Err(format!(
                "rgbd_depth_camera_id {} is outside the {} cameras of the rig, so depth would \
                 never be used",
                config.rgbd_depth_camera_id,
                cameras.len()
            ));
        }
        if !(config.rgbd_depth_scale_factor > 0.0 && config.rgbd_depth_scale_factor.is_finite()) {
            return Err(format!(
                "rgbd_depth_scale_factor must be positive and finite, got {}",
                config.rgbd_depth_scale_factor
            ));
        }
    }
    Ok(())
}

impl Tracker {
    pub fn new(
        cameras: &[CameraParams],
        imu: Option<&ffi::CuvImuCalibration>,
        config: &ffi::CuvConfig,
    ) -> Result<Self, String> {
        check_rig(cameras, imu, config)?;
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
        Ok(Self {
            raw,
            odometry_mode: config.odometry_mode,
        })
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
        // cuVSLAM accepts the measurement in any mode and only integrates it in Inertial.
        if self.odometry_mode != ffi::CUV_ODOMETRY_INERTIAL {
            return Err(format!(
                "odometry_mode {} does not read IMU measurements; only mode {} does",
                self.odometry_mode,
                ffi::CUV_ODOMETRY_INERTIAL
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraParams {
        CameraParams {
            width: 640,
            height: 480,
            principal: [320.0, 240.0],
            focal: [400.0, 400.0],
            rig_from_camera: ffi::CuvPose::default(),
            distortion_model: ffi::CUV_DISTORTION_PINHOLE,
            distortion_parameters: Vec::new(),
        }
    }

    fn config(odometry_mode: u8) -> ffi::CuvConfig {
        ffi::CuvConfig {
            odometry_mode,
            use_gpu: true,
            rectified_stereo_camera: true,
            rgbd_depth_scale_factor: 1000.0,
            rgbd_depth_camera_id: 0,
        }
    }

    #[test]
    fn an_imu_is_refused_outside_inertial_mode() {
        let calibration = ffi::CuvImuCalibration::default();
        for mode in [
            ffi::CUV_ODOMETRY_MULTICAMERA,
            ffi::CUV_ODOMETRY_RGBD,
            ffi::CUV_ODOMETRY_MONO,
        ] {
            let error = check_rig(&[camera(), camera()], Some(&calibration), &config(mode))
                .expect_err("only inertial mode reads an IMU");
            assert!(error.contains("IMU calibration"), "{error}");
        }
    }

    #[test]
    fn inertial_mode_without_an_imu_is_refused() {
        let error = check_rig(
            &[camera(), camera()],
            None,
            &config(ffi::CUV_ODOMETRY_INERTIAL),
        )
        .expect_err("inertial mode has nothing to integrate without a calibration");
        assert!(error.contains("IMU calibration"), "{error}");
    }

    #[test]
    fn rgbd_refuses_a_depth_camera_outside_the_rig() {
        let mut rgbd = config(ffi::CUV_ODOMETRY_RGBD);
        rgbd.rgbd_depth_camera_id = -1;
        let error =
            check_rig(&[camera()], None, &rgbd).expect_err("-1 silently drops every depth image");
        assert!(error.contains("rgbd_depth_camera_id"), "{error}");
    }

    #[test]
    fn rgbd_refuses_a_scale_factor_that_cannot_convert_depth() {
        for factor in [0.0, -1000.0, f32::NAN] {
            let mut rgbd = config(ffi::CUV_ODOMETRY_RGBD);
            rgbd.rgbd_depth_scale_factor = factor;
            let error = check_rig(&[camera()], None, &rgbd)
                .expect_err("a non-positive or non-finite scale cannot convert depth");
            assert!(error.contains("rgbd_depth_scale_factor"), "{error}");
        }
    }

    #[test]
    fn a_stereo_mode_needs_a_second_camera() {
        let error = check_rig(&[camera()], None, &config(ffi::CUV_ODOMETRY_MULTICAMERA))
            .expect_err("multicamera cannot triangulate from one camera");
        assert!(error.contains("at least 2 cameras"), "{error}");
    }

    #[test]
    fn a_usable_rig_passes() {
        let calibration = ffi::CuvImuCalibration::default();
        assert!(check_rig(
            &[camera(), camera()],
            None,
            &config(ffi::CUV_ODOMETRY_MULTICAMERA)
        )
        .is_ok());
        assert!(check_rig(&[camera()], None, &config(ffi::CUV_ODOMETRY_MONO)).is_ok());
        assert!(check_rig(&[camera()], None, &config(ffi::CUV_ODOMETRY_RGBD)).is_ok());
        assert!(check_rig(
            &[camera(), camera()],
            Some(&calibration),
            &config(ffi::CUV_ODOMETRY_INERTIAL)
        )
        .is_ok());
    }
}
