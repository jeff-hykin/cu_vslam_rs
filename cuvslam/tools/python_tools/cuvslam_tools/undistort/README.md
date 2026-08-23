# Image Undistortion Tool

This directory contains tools for undistorting images using various camera models. The tool supports single image processing and batch processing of folders.

## Files in this Directory

- `undistort.py` - Main undistortion tool (supports single image and batch modes)
- `edex.py` - Lightweight EDEX file reader/writer for loading camera configurations
- `input.edex` - Example EDEX file for testing
- `input.png` - Example image with distortion
- `output.png` - Example undistorted output

## Shared Utilities (in `../common/`)

This tool uses shared utilities from the `common/` folder:
- **camera.py** - Camera model implementations (pinhole, fisheye, brown5k, etc.) and `Intrinsics` class


## Quick Start

The `undistort.py` tool supports two modes of operation:

### 1. Single Image Mode (default)

Undistort a single image:

```bash
undistort_edex_images input.png camera.edex output.png
```

With custom output camera model:
```bash
undistort_edex_images input.png input.edex output.png output.edex --camera 0
```

### 2. Batch Mode (`--batch`)

Process multiple loose images from a folder:

```bash
# Process all jpg images
undistort_edex_images /path/to/images camera.edex output_dir --batch --pattern "*.jpg"

# Auto-detect common formats (jpg, jpeg, png, tga, bmp)
undistort_edex_images /path/to/images camera.edex output_dir --batch

# Process specific camera from multi-camera setup
undistort_edex_images /path/to/images camera.edex output_dir --batch --camera 1
```

### Positional Arguments

- `input_image` - Input file path (single image) or image directory (batch mode)
- `input_edex` - EDEX file path with camera intrinsics
- `output_image` - Output file path (single image) or output directory (batch mode)
- `output_edex` (optional) - EDEX file with output camera model. If not provided, uses pinhole model with input intrinsics

### Optional Arguments

- `--camera N` or `-c N` - Camera index from input EDEX file (default: 0)
- `--batch` or `-b` - Enable batch mode to process multiple images from a folder
- `--pattern PATTERN` or `-p PATTERN` - Glob pattern for batch mode (default: "*.{jpg,jpeg,png,tga,bmp}")

### How It Works

1. Reads camera intrinsics from EDEX file(s)
2. Creates input and output camera models
3. For each pixel in the output image:
   - Converts output pixel coordinates to normalized undistorted coordinates using output model
   - Converts normalized coordinates to input pixel coordinates using input model
4. Uses OpenCV's remap to generate the final undistorted image

## EDEX File

The `edex.py` module provides a lightweight EDEX file reader/writer for camera configurations.

### EdexFile Class

EDEX file reader/writer:
- `version` - EDEX format version
- `cameras` - List of camera data (each containing an `Intrinsics` object)
- `frame_start` - Start frame number
- `frame_end` - End frame number

Methods:
- `read(filename)` - Read EDEX file from disk
- `write(filename)` - Write EDEX file to disk

### EDEX File Format

The tool reads EDEX files which are JSON files with the following structure:

```json
[
  {
    "version": "0.9",
    "frame_start": 0,
    "frame_end": 100,
    "cameras": [
      {
        "intrinsics": {
          "size": [1920, 1080],
          "focal": [1000.0, 1000.0],
          "principal": [960.0, 540.0],
          "distortion_model": "brown5k",
          "distortion_params": [0.1, -0.05, 0.01, 0.001, 0.002]
        }
      }
    ]
  },
  { /* body */ }
]
```

## See Also

- **[../README.md](../README.md)** - Main tools directory documentation with architecture overview
- **[../common/camera.py](../common/camera.py)** - Complete camera model implementations
- **[../tracker/README.md](../tracker/README.md)** - Visual SLAM tracker documentation

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
