cuVSLAM: CUDA-Accelerated Visual Odometry and Mapping
=====================================================

This page provides documentation for cuVSLAM C++ API.

Two main classes are `cuvslam::Odometry` and `cuvslam::Slam`.

Tracking modes
--------------

The tracker supports several odometry modes — see `cuvslam::Odometry::OdometryMode` for the enum
and per-mode requirements. `Multisensor` accepts at least one RGB-D camera or one overlapping camera
pair, with an optional IMU, and requires a cuNLS-enabled build. It is configured through
`cuvslam::Odometry::MultisensorSettings`. A high-level mode chooser table lives in the top-level
README's "Tracking modes" section.

For saving maps and relocalizing with `LocalizeInMap`, see [load_map.md](load_map.md).

---

Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
