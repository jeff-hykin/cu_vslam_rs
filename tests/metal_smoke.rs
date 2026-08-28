// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// A fast GPU-only stereo run. Every failure mode this guards against was silent once: textures
// that sampled as zeros, a prebuilt metallib cache that was never consulted, kernels lowered with
// real doubles. None of them produced an error — they produced a tracker that returned the
// identity pose forever, so the assertions here are about motion, not about status codes.

mod common;

use cu_vslam_rs::{ffi, ImageRef, Tracker};

const FRAME_COUNT: usize = 8;

fn track_gpu_stereo() -> Vec<[f32; 3]> {
    let config = ffi::CuvConfig {
        odometry_mode: ffi::CUV_ODOMETRY_MULTICAMERA,
        use_gpu: true,
        rectified_stereo_camera: true,
        rgbd_depth_scale_factor: 1.0,
        rgbd_depth_camera_id: 0,
        multisensor_depth_scale_factor: 1.0,
        multisensor_depth_stereo_tracking: false,
    };
    // No CPU fallback. The SDK now carries a CPU backend too, which is exactly why this has to be
    // spelled out: falling back would turn a GPU failure into a green test.
    let mut tracker = Tracker::new(&common::stereo_cameras(), None, &[], &config)
        .unwrap_or_else(|error| panic!("GPU tracker creation failed: {error}"));
    let frames = common::stereo_frames(FRAME_COUNT);
    let mut translations = Vec::with_capacity(FRAME_COUNT);
    for (index, frame) in frames.iter().enumerate() {
        let image = |pixels, camera_index| ImageRef {
            pixels,
            width: common::IMAGE_WIDTH,
            height: common::IMAGE_HEIGHT,
            encoding: ffi::CUV_ENCODING_MONO,
            data_type: ffi::CUV_DATA_UINT8,
            timestamp_ns: frame.timestamp_ns,
            camera_index,
        };
        let estimate = tracker
            .track(&[image(&frame.left[..], 0), image(&frame.right[..], 1)], &[])
            .expect("track failed");
        translations.push(
            estimate
                .world_from_rig
                .unwrap_or_else(|| panic!("frame {index}: tracking lost"))
                .0
                .translation,
        );
    }
    translations
}

#[test]
fn metal_stereo_tracks_and_repeats() {
    let first_run = track_gpu_stereo();

    for index in 1..first_run.len() {
        let previous = first_run[index - 1];
        let current = first_run[index];
        let moved = (0..3).any(|axis| current[axis] != previous[axis]);
        // A frame that reproduces the previous pose exactly means the feature detector found
        // nothing to track, which is what zero-reading textures looked like.
        assert!(moved, "frame {index}: pose is bit-identical to frame {}, so nothing was tracked", index - 1);
    }

    let travelled = {
        let first = first_run.first().unwrap();
        let last = first_run.last().unwrap();
        ((last[0] - first[0]).powi(2) + (last[1] - first[1]).powi(2) + (last[2] - first[2]).powi(2))
            .sqrt()
    };
    let commanded = (common::RIG_STEP_METERS[0].powi(2) + common::RIG_STEP_METERS[2].powi(2)).sqrt()
        * (FRAME_COUNT - 1) as f32;
    assert!(
        travelled > commanded * 0.5,
        "travelled {travelled:.4} m against a commanded {commanded:.4} m"
    );

    assert_eq!(first_run, track_gpu_stereo(), "two identical runs disagreed");
}
