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

"""Installed console-script wrapper for the repository KITTI preparation flow."""

import argparse
import subprocess
from pathlib import Path
from typing import Optional


def _source_prepare_script() -> Optional[Path]:
    """Return the repository-local KITTI script when running from a source checkout."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = (
            parent
            / "tools"
            / "python_tools"
            / "cuvslam_tools"
            / "dataset_preparation"
            / "kitti"
            / "prepare_kitti.sh"
        )
        if candidate.exists():
            return candidate
    return None


def _bundled_prepare_script() -> Path:
    """Return the KITTI preparation script bundled with the installed package."""
    return Path(__file__).resolve().with_name("prepare_kitti.sh")


def _prepare_script() -> Path:
    """Return the best available KITTI preparation script path."""
    return _source_prepare_script() or _bundled_prepare_script()


def main(argv: Optional[list[str]] = None) -> int:
    """Run the KITTI preparation shell workflow from the console script."""
    source_script = _source_prepare_script()
    prepare_script = source_script or _bundled_prepare_script()

    parser = argparse.ArgumentParser(
        prog="prepare_kitti",
        description="Download KITTI odometry archives and convert them to cuVSLAM format.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory for raw KITTI archives. In a source checkout, defaults to the wrapped KITTI script default. "
            "In an installed package, defaults to ./datasets/kitti/raw."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted KITTI data and generated reporter configs. In a source checkout, defaults to "
            "the wrapped KITTI script default. In an installed package, defaults to ./datasets/converted."
        ),
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download archives even when they exist.")
    parser.add_argument("--download-only", action="store_true", help="Download archives but skip conversion.")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    if source_script is None:
        raw_dir = raw_dir or Path.cwd() / "datasets" / "kitti" / "raw"
        output_dir = output_dir or Path.cwd() / "datasets" / "converted"

    command = ["bash", str(prepare_script)]
    if raw_dir is not None:
        command.extend(["--raw-dir", str(raw_dir)])
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if args.force_download:
        command.append("--force-download")
    if args.download_only:
        command.append("--download-only")

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
