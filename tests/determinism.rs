// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use cu_vslam_rs::{ffi, CameraParams, ImageRef, PoseEstimate, Tracker};

mod common;
use common::{
    render_depth, render_gray, stereo_cameras, stereo_frames, StereoFrame, FOCAL_PIXELS,
    FRAME_INTERVAL_NS, IMAGE_HEIGHT, IMAGE_WIDTH, PRINCIPAL_X, PRINCIPAL_Y, RIG_STEP_METERS,
};

const FRAME_COUNT: usize = 24;

// cuVSLAM forbids concurrent calls into itself, and cargo runs test functions in parallel.
static TRACKER_LOCK: Mutex<()> = Mutex::new(());
static SEQUENCE: OnceLock<Vec<RgbdFrame>> = OnceLock::new();
static STEREO_SEQUENCE: OnceLock<Vec<StereoFrame>> = OnceLock::new();

struct RgbdFrame {
    color: Vec<u8>,
    depth: Vec<u8>,
    timestamp_ns: i64,
}

fn rendered_sequence() -> &'static [RgbdFrame] {
    SEQUENCE.get_or_init(|| {
        (0..FRAME_COUNT)
            .map(|index| {
                let step = index as f32;
                let origin = [
                    RIG_STEP_METERS[0] * step,
                    RIG_STEP_METERS[1] * step,
                    RIG_STEP_METERS[2] * step,
                ];
                RgbdFrame {
                    color: render_gray(origin),
                    depth: render_depth(origin),
                    timestamp_ns: index as i64 * FRAME_INTERVAL_NS,
                }
            })
            .collect()
    })
}

fn stereo_sequence() -> &'static [StereoFrame] {
    STEREO_SEQUENCE.get_or_init(|| stereo_frames(FRAME_COUNT))
}

fn depth_camera() -> Vec<CameraParams> {
    vec![CameraParams {
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        principal: [PRINCIPAL_X, PRINCIPAL_Y],
        focal: [FOCAL_PIXELS, FOCAL_PIXELS],
        rig_from_camera: ffi::CuvPose {
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
        },
        distortion_model: ffi::CUV_DISTORTION_PINHOLE,
        distortion_parameters: Vec::new(),
    }]
}

fn stereo_config(use_gpu: bool) -> ffi::CuvConfig {
    ffi::CuvConfig {
        odometry_mode: ffi::CUV_ODOMETRY_MULTICAMERA,
        use_gpu,
        rectified_stereo_camera: true,
        rgbd_depth_scale_factor: 1.0,
        rgbd_depth_camera_id: 0,
    }
}

fn depth_config(use_gpu: bool) -> ffi::CuvConfig {
    ffi::CuvConfig {
        odometry_mode: ffi::CUV_ODOMETRY_RGBD,
        use_gpu,
        rectified_stereo_camera: false,
        rgbd_depth_scale_factor: 1.0,
        rgbd_depth_camera_id: 0,
    }
}

fn backend_name(use_gpu: bool) -> &'static str {
    if use_gpu {
        "GPU"
    } else {
        "CPU"
    }
}

// An ENFORCE_GPU=ON SDK carries only its own backend and refuses to construct the other, so
// which backends exist is a property of the SDK on disk. The tests used to fall back from GPU
// to CPU without saying so, which let a GPU run stand in for a CPU determinism result.
fn backend_is_available(use_gpu: bool) -> bool {
    Tracker::new(&stereo_cameras(), None, &stereo_config(use_gpu)).is_ok()
}

fn track_stereo_sequence<'a>(frames: &'a [StereoFrame], use_gpu: bool) -> Vec<PoseEstimate> {
    let backend = backend_name(use_gpu);
    let mut tracker = Tracker::new(&stereo_cameras(), None, &stereo_config(use_gpu))
        .unwrap_or_else(|error| panic!("{backend} tracker creation failed: {error}"));
    frames
        .iter()
        .map(|frame| {
            let image = |pixels: &'a [u8], camera_index| ImageRef {
                pixels,
                width: IMAGE_WIDTH,
                height: IMAGE_HEIGHT,
                encoding: ffi::CUV_ENCODING_MONO,
                data_type: ffi::CUV_DATA_UINT8,
                timestamp_ns: frame.timestamp_ns,
                camera_index,
            };
            tracker
                .track(&[image(&frame.left, 0), image(&frame.right, 1)], &[])
                .expect("track failed")
        })
        .collect()
}

fn track_sequence(frames: &[RgbdFrame], use_gpu: bool) -> Vec<PoseEstimate> {
    let backend = backend_name(use_gpu);
    let mut tracker = Tracker::new(&depth_camera(), None, &depth_config(use_gpu))
        .unwrap_or_else(|error| panic!("{backend} tracker creation failed: {error}"));
    frames
        .iter()
        .map(|frame| {
            let color = [ImageRef {
                pixels: &frame.color,
                width: IMAGE_WIDTH,
                height: IMAGE_HEIGHT,
                encoding: ffi::CUV_ENCODING_MONO,
                data_type: ffi::CUV_DATA_UINT8,
                timestamp_ns: frame.timestamp_ns,
                camera_index: 0,
            }];
            let depth = [ImageRef {
                pixels: &frame.depth,
                width: IMAGE_WIDTH,
                height: IMAGE_HEIGHT,
                encoding: ffi::CUV_ENCODING_MONO,
                data_type: ffi::CUV_DATA_FLOAT32,
                timestamp_ns: frame.timestamp_ns,
                camera_index: 0,
            }];
            tracker.track(&color, &depth).expect("track failed")
        })
        .collect()
}

// Bit-identical poses prove nothing unless the tracker was really tracking.
fn assert_tracking_really_happened(estimates: &[PoseEstimate]) {
    let translations: Vec<[f32; 3]> = estimates
        .iter()
        .filter_map(|estimate| estimate.world_from_rig.map(|(pose, _)| pose.translation))
        .collect();
    assert_eq!(translations.len(), FRAME_COUNT, "tracking was lost on some frame");

    let first = translations.first().unwrap();
    let last = translations.last().unwrap();
    let travelled = ((last[0] - first[0]).powi(2)
        + (last[1] - first[1]).powi(2)
        + (last[2] - first[2]).powi(2))
    .sqrt();
    let commanded = (RIG_STEP_METERS[0].powi(2) + RIG_STEP_METERS[2].powi(2)).sqrt()
        * (FRAME_COUNT - 1) as f32;
    assert!(
        travelled > 0.5 * commanded && travelled < 2.0 * commanded,
        "estimated travel {travelled} m is not in the same league as the commanded {commanded} m"
    );
}

fn run_on(backends: &[bool], body: impl Fn(bool)) {
    let mut ran_any = false;
    for &use_gpu in backends {
        if !backend_is_available(use_gpu) {
            println!("skipping {}: not available here", backend_name(use_gpu));
            continue;
        }
        println!("running on {}", backend_name(use_gpu));
        body(use_gpu);
        ran_any = true;
    }
    assert!(ran_any, "no requested backend is available here");
}

fn on_every_backend(body: impl Fn(bool)) {
    run_on(&[false, true], body);
}

// cuVSLAM lifts depth pixels to points only in a CUDA kernel, so an RGBD tracker built with
// use_gpu=false finds no landmarks and reports no motion at all.
fn on_gpu_only(body: impl Fn(bool)) {
    run_on(&[true], body);
}

#[test]
fn the_stereo_tracker_locks_onto_the_synthetic_scene_and_follows_the_commanded_motion() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    on_every_backend(|use_gpu| {
        assert_tracking_really_happened(&track_stereo_sequence(stereo_sequence(), use_gpu));
    });
}

#[test]
fn two_stereo_trackers_fed_the_same_sequence_produce_identical_poses() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    on_every_backend(|use_gpu| {
        let frames = stereo_sequence();
        let first_run = track_stereo_sequence(frames, use_gpu);
        let second_run = track_stereo_sequence(frames, use_gpu);
        assert_tracking_really_happened(&first_run);
        assert_runs_match(&first_run, &second_run);
    });
}

#[test]
fn the_tracker_locks_onto_the_synthetic_scene_and_follows_the_commanded_motion() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    on_gpu_only(|use_gpu| {
        assert_tracking_really_happened(&track_sequence(rendered_sequence(), use_gpu));
    });
}

#[test]
fn two_trackers_fed_the_same_sequence_produce_identical_poses() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    on_gpu_only(|use_gpu| {
        let frames = rendered_sequence();
        let first_run = track_sequence(frames, use_gpu);
        let second_run = track_sequence(frames, use_gpu);
        assert_tracking_really_happened(&first_run);
        assert_runs_match(&first_run, &second_run);
    });
}

fn assert_runs_match(first_run: &[PoseEstimate], second_run: &[PoseEstimate]) {
    assert_eq!(first_run.len(), second_run.len());
    for (index, (first, second)) in first_run.iter().zip(second_run).enumerate() {
        assert_eq!(first.timestamp_ns, second.timestamp_ns, "frame {index} timestamp");
        match (first.world_from_rig, second.world_from_rig) {
            (Some((first_pose, first_covariance)), Some((second_pose, second_covariance))) => {
                assert_eq!(
                    first_pose.translation, second_pose.translation,
                    "frame {index} translation"
                );
                assert_eq!(
                    first_pose.rotation_xyzw, second_pose.rotation_xyzw,
                    "frame {index} rotation"
                );
                assert_eq!(
                    first_covariance.as_slice(),
                    second_covariance.as_slice(),
                    "frame {index} covariance"
                );
            }
            (None, None) => {}
            _ => panic!("frame {index} tracked in one run but not the other"),
        }
    }
}
