// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: LicenseRef-NVIDIA-Community

//! Rust FFI for [cuVSLAM](https://github.com/nvidia-isaac/cuVSLAM). Linking needs
//! `CUVSLAM_SDK_DIR`; see the README.

pub mod ffi;
mod tracker;

pub use tracker::{CameraParams, ImageRef, PoseEstimate, Tracker};
