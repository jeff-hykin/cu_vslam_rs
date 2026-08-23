// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Rust FFI for cuVSLAM (https://github.com/nvidia-isaac/cuVSLAM): raw bindings in
// `ffi`, a safe `Tracker` wrapper on top. Linking needs CUVSLAM_SDK_DIR at build
// time, pointing at an SDK layout (include/cuvslam/cuvslam2.h + lib/libcuvslam.*);
// the nix flake in this repository builds one per platform, including macOS via
// CuMetal.

pub mod ffi;
mod tracker;

pub use tracker::{CameraParams, ImageRef, PoseEstimate, Tracker};
