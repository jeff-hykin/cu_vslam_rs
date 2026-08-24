// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// A synthetic textured scene shared by the integration tests.

#![allow(dead_code)]

use cu_vslam_rs::{ffi, CameraParams};

pub const IMAGE_WIDTH: i32 = 640;
pub const IMAGE_HEIGHT: i32 = 480;
pub const FOCAL_PIXELS: f32 = 460.0;
pub const PRINCIPAL_X: f32 = 320.0;
pub const PRINCIPAL_Y: f32 = 240.0;
pub const FRAME_INTERVAL_NS: i64 = 33_000_000;
// Mostly sideways, because a forward-only walk gives the image centre almost no parallax.
pub const RIG_STEP_METERS: [f32; 3] = [0.02, 0.0, 0.01];
pub const BASELINE_METERS: f32 = 0.12;

struct Plane {
    normal: [f32; 3],
    offset: f32,
}

const SCENE_PLANES: [Plane; 2] = [
    Plane { normal: [0.3, 0.0, 1.0], offset: 6.0 },
    Plane { normal: [0.0, 1.0, 0.0], offset: 1.5 },
];

pub struct StereoFrame {
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub timestamp_ns: i64,
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

fn ray_direction(pixel_x: i32, pixel_y: i32) -> [f32; 3] {
    [
        (pixel_x as f32 - PRINCIPAL_X) / FOCAL_PIXELS,
        (pixel_y as f32 - PRINCIPAL_Y) / FOCAL_PIXELS,
        1.0,
    ]
}

/// Distance along the ray to the nearest plane, and the point it lands on.
fn nearest_surface(camera_origin: [f32; 3], direction: [f32; 3]) -> Option<(f32, [f32; 3])> {
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
        return None;
    }
    Some((
        nearest,
        [
            camera_origin[0] + nearest * direction[0],
            camera_origin[1] + nearest * direction[1],
            camera_origin[2] + nearest * direction[2],
        ],
    ))
}

pub fn render_gray(camera_origin: [f32; 3]) -> Vec<u8> {
    let mut color = vec![0u8; (IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    for pixel_y in 0..IMAGE_HEIGHT {
        for pixel_x in 0..IMAGE_WIDTH {
            let direction = ray_direction(pixel_x, pixel_y);
            if let Some((_, surface)) = nearest_surface(camera_origin, direction) {
                color[(pixel_y * IMAGE_WIDTH + pixel_x) as usize] = surface_brightness(surface);
            }
        }
    }
    color
}

/// Little-endian f32 metres along the optical axis, not along the ray, which is what RGBD wants.
pub fn render_depth(camera_origin: [f32; 3]) -> Vec<u8> {
    let mut depth = vec![0u8; (IMAGE_WIDTH * IMAGE_HEIGHT) as usize * 4];
    for pixel_y in 0..IMAGE_HEIGHT {
        for pixel_x in 0..IMAGE_WIDTH {
            let direction = ray_direction(pixel_x, pixel_y);
            if let Some((distance, _)) = nearest_surface(camera_origin, direction) {
                let index = (pixel_y * IMAGE_WIDTH + pixel_x) as usize;
                let along_axis = distance * direction[2];
                depth[index * 4..index * 4 + 4].copy_from_slice(&along_axis.to_le_bytes());
            }
        }
    }
    depth
}

pub fn stereo_frames(count: usize) -> Vec<StereoFrame> {
    (0..count)
        .map(|index| {
            let step = index as f32;
            let origin = [
                RIG_STEP_METERS[0] * step,
                RIG_STEP_METERS[1] * step,
                RIG_STEP_METERS[2] * step,
            ];
            StereoFrame {
                left: render_gray(origin),
                right: render_gray([origin[0] + BASELINE_METERS, origin[1], origin[2]]),
                timestamp_ns: index as i64 * FRAME_INTERVAL_NS,
            }
        })
        .collect()
}

pub fn stereo_cameras() -> Vec<CameraParams> {
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
