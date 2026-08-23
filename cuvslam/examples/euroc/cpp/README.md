# EuRoC C++ Tracking Example

This example demonstrates how to use the cuVSLAM C++ API for visual-inertial odometry (VIO) and SLAM on the EuRoC MAV dataset.

## Overview

This is a single-file C++ project that closely mirrors the functionality of `examples/euroc/track_euroc.py` but uses the C++ API directly.
It also expands to add SLAM results visualization.

The example is integrated into the workspace CMake build and produces the `track_euroc` executable in the `bin/` directory alongside other tools.

## Requirements

- CMake 3.19 or higher
- C++17 compatible compiler
- CUDA 12 or 13

All dependencies (libpng, yaml-cpp) are managed via FetchContent in the workspace build — no system packages needed.

[Install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## Building

**All build commands must be run from the repository root** (the top-level directory containing the root `CMakeLists.txt`), not from `examples/euroc/cpp/`.

### Basic Build (Console Output Only)

The example is built as part of the main workspace:

```bash
# From the repository root
cmake -S . -B build
cmake --build build --target track_euroc
```

To re-build after code changes, run:

```bash
cmake --build build --target track_euroc
```

### Build with Rerun Visualization

To use Rerun Viewer for live visualization, see [Rerun C++ Installation](https://rerun.io/docs/getting-started/quick-start/cpp).

Enable Rerun visualization with the workspace-level `USE_RERUN` option, the Rerun C++ SDK will be fetched automatically:

```bash
# From the repository root
cmake -S . -B build -DUSE_RERUN=ON
cmake --build build --target track_euroc
```

### Customizing the Layout with Blueprints

With Rerun enabled the C++ example loads `euroc_blueprint.rbl` (shipped alongside the executable) to customize the visualization layout. To regenerate it:

```bash
# Uncomment blueprint.save(blueprint_path) in track_euroc.py, then run from examples/euroc/:
cd examples/euroc
python3 track_euroc.py
```

## Running

The example requires the EuRoC dataset (see [Python tutorial](../README.md) for downloading instructions):

```bash
./build/bin/track_euroc /path/to/euroc/dataset/mav0
```

## References

- [cuVSLAM C++ API Documentation](https://nvidia-isaac.github.io/cuVSLAM/cpp/)
- [Python version: examples/euroc/track_euroc.py](../track_euroc.py)
