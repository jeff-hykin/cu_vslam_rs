# Undistort Tool

A tool to undistort images using camera intrinsics from an edex file.

## Overview

This tool reads distorted images and camera calibration parameters from an edex file, then generates undistorted (rectified) images. It supports various distortion models including fisheye, brown5k, and rational6kt.

## Usage

### Single Image

```bash
./bin/undistort <input_image> <input_edex> <output_image> [<output_edex>]
```

**Arguments:**
- `input_image` - Path to the distorted input image
- `input_edex` - Path to edex file containing input camera intrinsics
- `output_image` - Path where the undistorted image will be saved
- `output_edex` (optional) - Path to edex file containing output camera intrinsics. If not provided, input intrinsics with pinhole model are used.

**Flags:**
- `--camera=N` - Camera number to use from the edex file (default: 0)

**Example:**
```bash
./bin/undistort distorted.png calibration.edex undistorted.png --camera=0
```

### Batch Processing

Use the Python script to undistort multiple images:

```bash
python batch_undistort.py <in_folder> <in_images_mask> <in_edex> <in_camera_id> <out_folder>
```

**Arguments:**
- `in_folder` - Directory containing input images
- `in_images_mask` - Glob pattern for input images (e.g., `*.jpg`, `cam0_*.png`)
- `in_edex` - Path to edex file with camera intrinsics
- `in_camera_id` - Camera index in the edex file (0-based)
- `out_folder` - Output directory for undistorted images

**Example:**
```bash
python batch_undistort.py ~/raw "cam0_right_*.jpg" ~/stereo.edex 1 ~/undistorted
```

**Environment Variable:**
- `CUVSLAM_UNDISTORT` - Path to the undistort binary (default: `~/cuvslam/build/release/bin/undistort`)

## Use Cases

1. **Verify calibration** - Visually check if distortion parameters are correct by examining the undistorted output
2. **Prepare data for external tools** - Generate rectified images for tools that don't support distortion models
3. **Debug stereo calibration** - Undistort left and right images to verify epipolar alignment
