#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""Python implementation of image undistortion tool.
Supports both single image and dataset (EDEX) batch processing.
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict

from cuvslam_tools.common.camera import CameraModel, create_camera_model
from cuvslam_tools.common.edex import EDEXMetadata, Intrinsics, DistortionModel


def create_camera_model_from_intrinsics(
    intrinsics: Intrinsics,
) -> Optional[CameraModel]:
    """Create camera model from intrinsics."""
    return create_camera_model(
        intrinsics.resolution,
        intrinsics.focal,
        intrinsics.principal,
        intrinsics.distortion_model,
        intrinsics.distortion_params,
    )


def undistort_image(
    input_model: CameraModel,
    output_model: CameraModel,
    input_image: np.ndarray,
    output_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Undistort image using input and output camera models.

    Args:
        input_model: Camera model for input image
        output_model: Camera model for output image
        input_image: Input image to undistort
        output_shape: Output image shape (height, width)

    Returns:
        Undistorted output image
    """
    height, width = output_shape

    # Create remapping grids
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # For each pixel in output image, find corresponding pixel in input image
    for y in range(height):
        for x in range(width):
            dst = np.array([x, y], dtype=np.float32)

            # Convert output pixel to normalized coordinates
            success, interim = output_model.normalize_point(dst)
            if not success:
                map_x[y, x] = -1
                map_y[y, x] = -1
                continue

            # Convert normalized coordinates to input pixel coordinates
            success, src = input_model.denormalize_point(interim)
            if not success:
                map_x[y, x] = -1
                map_y[y, x] = -1
                continue

            map_x[y, x] = src[0]
            map_y[y, x] = src[1]

    # Apply remapping
    output_image = cv2.remap(input_image, map_x, map_y, cv2.INTER_LINEAR)

    return output_image


def check_condition(condition: bool, message: str):
    """Check condition and exit with error message if false."""
    if not condition:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)


def process_single_image(args):
    """Process a single image (original functionality)."""
    # Read input EDEX file
    input_edex_file = EDEXMetadata.read(args.input_edex)
    check_condition(
        0 <= args.camera < len(input_edex_file.header.cameras),
        f"No camera {args.camera} in {args.input_edex}",
    )

    input_intr = input_edex_file.header.cameras[args.camera].intrinsics

    # Read output EDEX file or use input intrinsics with pinhole model
    if args.output_edex:
        output_edex_file = EDEXMetadata.read(args.output_edex)
        check_condition(
            len(output_edex_file.header.cameras) > 0, f"No camera in {args.output_edex}"
        )
        output_intr = output_edex_file.header.cameras[0].intrinsics
    else:
        # Use input intrinsics but with pinhole model (no distortion)
        output_intr = Intrinsics(
            resolution=input_intr.resolution.copy(),
            focal=input_intr.focal.copy(),
            principal=input_intr.principal.copy(),
            distortion_model=DistortionModel.PINHOLE,
            distortion_params=np.array([], dtype=np.float32),
        )

    # Create camera models
    input_model = create_camera_model_from_intrinsics(input_intr)
    check_condition(input_model is not None, "Cannot create input camera model")

    output_model = create_camera_model_from_intrinsics(output_intr)
    check_condition(output_model is not None, "Cannot create output camera model")

    # Load input image
    input_image = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)
    check_condition(
        input_image is not None, f"Cannot load input image: {args.input_image}"
    )

    # Get output size
    output_height = int(output_intr.resolution[1])
    output_width = int(output_intr.resolution[0])

    # Undistort image
    print(f"Undistorting image...")
    print(f"  Input model: {input_intr.distortion_model}")
    print(f"  Output model: {output_intr.distortion_model}")
    print(f"  Input size: {input_image.shape[1]}x{input_image.shape[0]}")
    print(f"  Output size: {output_width}x{output_height}")

    output_image = undistort_image(
        input_model, output_model, input_image, (output_height, output_width)
    )

    # Save output image
    success = cv2.imwrite(args.output_image, output_image)
    check_condition(success, f"Cannot save output image: {args.output_image}")

    print(f"Successfully saved undistorted image to {args.output_image}")


def process_batch(args):
    """Process batch of loose image files from a folder."""
    # Read EDEX file
    input_edex_file = EDEXMetadata.read(args.input_edex)
    check_condition(
        0 <= args.camera < len(input_edex_file.header.cameras),
        f"No camera {args.camera} in {args.input_edex}",
    )

    # Get input intrinsics
    input_intr = input_edex_file.header.cameras[args.camera].intrinsics

    # Read output EDEX file or use input intrinsics with pinhole model
    if args.output_edex:
        output_edex_file = EDEXMetadata.read(args.output_edex)
        check_condition(
            len(output_edex_file.header.cameras) > 0, f"No camera in {args.output_edex}"
        )
        output_intr = output_edex_file.header.cameras[0].intrinsics
    else:
        # Use input intrinsics but with pinhole model (no distortion)
        output_intr = Intrinsics(
            resolution=input_intr.resolution.copy(),
            focal=input_intr.focal.copy(),
            principal=input_intr.principal.copy(),
            distortion_model=DistortionModel.PINHOLE,
            distortion_params=np.array([], dtype=np.float32),
        )

    # Create camera models
    input_model = create_camera_model_from_intrinsics(input_intr)
    check_condition(input_model is not None, "Cannot create input camera model")

    output_model = create_camera_model_from_intrinsics(output_intr)
    check_condition(output_model is not None, "Cannot create output camera model")

    # Get output size
    output_height = int(output_intr.resolution[1])
    output_width = int(output_intr.resolution[0])

    print(f"Batch undistort configuration:")
    print(f"  Input model: {input_intr.distortion_model}")
    print(f"  Output model: {output_intr.distortion_model}")
    print(f"  Output size: {output_width}x{output_height}")
    print(f"  Camera ID: {args.camera}")
    print()

    # Create output folder
    output_dir = Path(args.output_image)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image pattern (default to common image formats)
    pattern = args.pattern if args.pattern else "*.{jpg,jpeg,png,tga,bmp}"

    # Process each image
    input_dir = Path(args.input_image)
    check_condition(
        input_dir.is_dir(),
        f"Input path must be a directory for batch mode: {args.input_image}",
    )

    # Handle glob pattern
    if "{" in pattern and "}" in pattern:
        # Multiple extensions pattern like "*.{jpg,png}"
        import glob

        image_files = []
        for ext_pattern in pattern.split("{")[1].split("}")[0].split(","):
            base_pattern = pattern.split("{")[0]
            image_files.extend(input_dir.glob(f"{base_pattern}{ext_pattern}"))
        image_files = sorted(set(image_files))
    else:
        image_files = sorted(input_dir.glob(pattern))

    check_condition(
        len(image_files) > 0,
        f"No images found matching pattern '{pattern}' in {args.input_image}",
    )

    print(f"Processing {len(image_files)} images...")

    processed_count = 0
    for i, in_image_path in enumerate(image_files, 1):
        out_image_path = output_dir / in_image_path.with_suffix(".png").name

        try:
            # Load input image
            input_image = cv2.imread(str(in_image_path), cv2.IMREAD_UNCHANGED)
            if input_image is None:
                print(
                    f"  [{i}/{len(image_files)}] ✗ Failed to load: {in_image_path.name}",
                    file=sys.stderr,
                )
                continue

            # Undistort image
            output_image = undistort_image(
                input_model, output_model, input_image, (output_height, output_width)
            )

            # Save output image
            success = cv2.imwrite(str(out_image_path), output_image)
            if not success:
                print(
                    f"  [{i}/{len(image_files)}] ✗ Failed to save: {out_image_path.name}",
                    file=sys.stderr,
                )
                continue

            processed_count += 1
            print(
                f"  [{i}/{len(image_files)}] ✓ {in_image_path.name} -> {out_image_path.name}"
            )

        except Exception as e:
            print(
                f"  [{i}/{len(image_files)}] ✗ Error processing {in_image_path.name}: {e}",
                file=sys.stderr,
            )
            continue

    print(f"\nBatch processing complete!")
    print(f"Processed {processed_count}/{len(image_files)} images")
    print(f"Output directory: {args.output_image}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Undistort images using camera models from EDEX files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image: Undistort to pinhole (no output edex)
  %(prog)s input.png input.edex output.png

  # Single image: Undistort using custom output camera model
  %(prog)s input.png input.edex output.png output.edex

  # Single image: Use specific camera from input edex
  %(prog)s input.png input.edex output.png --camera 1

  # Batch mode: Undistort folder of images with glob pattern
  %(prog)s /path/to/images camera.edex output_dir --batch --pattern "*.jpg"

  # Batch mode: Auto-detect common image formats
  %(prog)s /path/to/images camera.edex output_dir --batch --camera 0
        """,
    )

    parser.add_argument(
        "input_image",
        help="Input image file path or image directory (for --batch mode)",
    )
    parser.add_argument(
        "input_edex", help="Input EDEX file path with camera intrinsics"
    )
    parser.add_argument(
        "output_image",
        help="Output image file path or output directory (for --batch mode)",
    )
    parser.add_argument(
        "output_edex",
        nargs="?",
        default="",
        help="Output EDEX file path (optional). If not set, input intrinsics and pinhole model are used.",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=int,
        default=0,
        help="Camera number from input EDEX (default: 0)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        action="store_true",
        help="Process batch of loose images from folder",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default=None,
        help='Glob pattern for batch mode (default: "*.{jpg,jpeg,png,tga,bmp}")',
    )

    args = parser.parse_args()

    # Route to appropriate processing mode
    if args.batch:
        process_batch(args)
    else:
        process_single_image(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
