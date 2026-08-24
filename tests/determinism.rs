// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use cu_vslam_rs::{ffi, CameraParams, ImageRef, PoseEstimate, Tracker};

const IMAGE_WIDTH: i32 = 640;
const IMAGE_HEIGHT: i32 = 480;
const FOCAL_PIXELS: f32 = 460.0;
const PRINCIPAL_X: f32 = 320.0;
const PRINCIPAL_Y: f32 = 240.0;
const FRAME_COUNT: usize = 24;
const FRAME_INTERVAL_NS: i64 = 33_000_000;
// Mostly sideways, because a forward-only walk gives the image centre almost no parallax.
const RIG_STEP_METERS: [f32; 3] = [0.02, 0.0, 0.01];
const BASELINE_METERS: f32 = 0.12;

// cuVSLAM forbids concurrent calls into itself, and cargo runs test functions in parallel.
static TRACKER_LOCK: Mutex<()> = Mutex::new(());
static SEQUENCE: OnceLock<Vec<RgbdFrame>> = OnceLock::new();
static STEREO_SEQUENCE: OnceLock<Vec<StereoFrame>> = OnceLock::new();

struct Plane {
    normal: [f32; 3],
    offset: f32,
}

const SCENE_PLANES: [Plane; 2] = [
    Plane { normal: [0.3, 0.0, 1.0], offset: 6.0 },
    Plane { normal: [0.0, 1.0, 0.0], offset: 1.5 },
];

struct RgbdFrame {
    color: Vec<u8>,
    /// Metres, little-endian f32.
    depth: Vec<u8>,
    timestamp_ns: i64,
}

struct StereoFrame {
    left: Vec<u8>,
    right: Vec<u8>,
    timestamp_ns: i64,
}

fn lattice_value(x: i32, y: i32, z: i32) -> f32 {
    let mut hash = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0x5bd1_e995);
    hash ^= hash >> 15;
    (hash >> 8) as f32 / (1u32 << 24) as f32
}

fn value_noise(point: [f32; 3], frequency: f32) -> f32 {
    let scaled = [point[0] * frequency, point[1] * frequency, point[2] * frequency];
    let cell = [scaled[0].floor(), scaled[1].floor(), scaled[2].floor()];
    let mut blend = [0.0; 3];
    for axis in 0..3 {
        let fraction = scaled[axis] - cell[axis];
        blend[axis] = fraction * fraction * (3.0 - 2.0 * fraction);
    }
    let mut total = 0.0;
    for corner in 0..8 {
        let corner_offsets = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
        let mut weight = 1.0;
        for axis in 0..3 {
            weight *= if corner_offsets[axis] == 1 { blend[axis] } else { 1.0 - blend[axis] };
        }
        total += weight
            * lattice_value(
                cell[0] as i32 + corner_offsets[0],
                cell[1] as i32 + corner_offsets[1],
                cell[2] as i32 + corner_offsets[2],
            );
    }
    total
}

fn surface_brightness(point: [f32; 3]) -> u8 {
    let texture = 0.50 * value_noise(point, 1.5)
        + 0.30 * value_noise(point, 4.0)
        + 0.20 * value_noise(point, 10.0);
    // Near-binary, the way cuVSLAM's own synthetic tests build their pattern.
    let contrasted = ((texture - 0.5) * 25.0 + 0.5).clamp(0.0, 1.0);
    (255.0 * contrasted) as u8
}

fn render_view(camera_origin: [f32; 3]) -> (Vec<u8>, Vec<u8>) {
    let pixel_count = (IMAGE_WIDTH * IMAGE_HEIGHT) as usize;
    let mut color = vec![0u8; pixel_count];
    let mut depth = vec![0u8; pixel_count * 4];
    for pixel_y in 0..IMAGE_HEIGHT {
        for pixel_x in 0..IMAGE_WIDTH {
            let direction = [
                (pixel_x as f32 - PRINCIPAL_X) / FOCAL_PIXELS,
                (pixel_y as f32 - PRINCIPAL_Y) / FOCAL_PIXELS,
                1.0,
            ];
            let mut nearest = f32::MAX;
            for plane in &SCENE_PLANES {
                let along_ray = plane.normal[0] * direction[0]
                    + plane.normal[1] * direction[1]
                    + plane.normal[2] * direction[2];
                if along_ray.abs() < 1e-6 {
                    continue;
                }
                let to_camera = plane.normal[0] * camera_origin[0]
                    + plane.normal[1] * camera_origin[1]
                    + plane.normal[2] * camera_origin[2];
                let distance = (plane.offset - to_camera) / along_ray;
                if distance > 0.3 && distance < nearest {
                    nearest = distance;
                }
            }
            if nearest == f32::MAX {
                continue;
            }
            let surface = [
                camera_origin[0] + nearest * direction[0],
                camera_origin[1] + nearest * direction[1],
                camera_origin[2] + nearest * direction[2],
            ];
            let index = (pixel_y * IMAGE_WIDTH + pixel_x) as usize;
            color[index] = surface_brightness(surface);
            // cuVSLAM wants depth along the optical axis, not distance along the ray.
            let along_axis = nearest * direction[2];
            depth[index * 4..index * 4 + 4].copy_from_slice(&along_axis.to_le_bytes());
        }
    }
    (color, depth)
}

fn rendered_sequence() -> &'static [RgbdFrame] {
    SEQUENCE.get_or_init(|| {
        (0..FRAME_COUNT)
            .map(|index| {
                let step = index as f32;
                let (color, depth) = render_view([
                    RIG_STEP_METERS[0] * step,
                    RIG_STEP_METERS[1] * step,
                    RIG_STEP_METERS[2] * step,
                ]);
                RgbdFrame { color, depth, timestamp_ns: index as i64 * FRAME_INTERVAL_NS }
            })
            .collect()
    })
}

fn stereo_sequence() -> &'static [StereoFrame] {
    STEREO_SEQUENCE.get_or_init(|| {
        (0..FRAME_COUNT)
            .map(|index| {
                let step = index as f32;
                let origin = [
                    RIG_STEP_METERS[0] * step,
                    RIG_STEP_METERS[1] * step,
                    RIG_STEP_METERS[2] * step,
                ];
                let (left, _) = render_view(origin);
                let (right, _) =
                    render_view([origin[0] + BASELINE_METERS, origin[1], origin[2]]);
                StereoFrame { left, right, timestamp_ns: index as i64 * FRAME_INTERVAL_NS }
            })
            .collect()
    })
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

fn stereo_cameras() -> Vec<CameraParams> {
    [[0.0, 0.0, 0.0], [BASELINE_METERS, 0.0, 0.0]]
        .into_iter()
        .map(|translation| CameraParams {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            principal: [PRINCIPAL_X, PRINCIPAL_Y],
            focal: [FOCAL_PIXELS, FOCAL_PIXELS],
            rig_from_camera: ffi::CuvPose {
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                translation,
            },
            distortion_model: ffi::CUV_DISTORTION_PINHOLE,
            distortion_parameters: Vec::new(),
        })
        .collect()
}

fn track_stereo_sequence<'a>(frames: &'a [StereoFrame]) -> Vec<PoseEstimate> {
    let config = |use_gpu| ffi::CuvConfig {
        odometry_mode: ffi::CUV_ODOMETRY_MULTICAMERA,
        use_gpu,
        rectified_stereo_camera: true,
        rgbd_depth_scale_factor: 1.0,
        rgbd_depth_camera_id: 0,
    };
    let mut tracker = Tracker::new(&stereo_cameras(), None, &config(true))
        .or_else(|_| Tracker::new(&stereo_cameras(), None, &config(false)))
        .expect("tracker creation failed");
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

fn track_sequence(frames: &[RgbdFrame]) -> Vec<PoseEstimate> {
    let config = |use_gpu| ffi::CuvConfig {
        odometry_mode: ffi::CUV_ODOMETRY_RGBD,
        use_gpu,
        rectified_stereo_camera: false,
        rgbd_depth_scale_factor: 1.0,
        rgbd_depth_camera_id: 0,
    };
    // An ENFORCE_GPU=ON SDK carries only its own backend and refuses to construct the other.
    let mut tracker = Tracker::new(&depth_camera(), None, &config(true))
        .or_else(|_| Tracker::new(&depth_camera(), None, &config(false)))
        .expect("tracker creation failed");
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

#[test]
fn the_stereo_tracker_locks_onto_the_synthetic_scene_and_follows_the_commanded_motion() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_tracking_really_happened(&track_stereo_sequence(stereo_sequence()));
}

#[test]
fn two_stereo_trackers_fed_the_same_sequence_produce_identical_poses() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let frames = stereo_sequence();
    let first_run = track_stereo_sequence(frames);
    let second_run = track_stereo_sequence(frames);
    assert_tracking_really_happened(&first_run);
    assert_runs_match(&first_run, &second_run);
}

#[test]
fn the_tracker_locks_onto_the_synthetic_scene_and_follows_the_commanded_motion() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_tracking_really_happened(&track_sequence(rendered_sequence()));
}

#[test]
fn two_trackers_fed_the_same_sequence_produce_identical_poses() {
    let _serialized = TRACKER_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let frames = rendered_sequence();
    let first_run = track_sequence(frames);
    let second_run = track_sequence(frames);
    assert_tracking_really_happened(&first_run);
    assert_runs_match(&first_run, &second_run);
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
