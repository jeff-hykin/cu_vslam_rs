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

"""Installed console-script wrapper for the TUM RGB-D preparation flow."""

import argparse
import subprocess
from pathlib import Path
from typing import Optional

from cuvslam_tools.dataset_preparation.common import (
    installed_output_dir,
    installed_raw_dir,
    resolve_prepare_script,
)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the TUM RGB-D preparation shell workflow from the console script."""
    prepare_script, is_source_checkout = resolve_prepare_script(__file__, "tum", "prepare_tum.sh")

    parser = argparse.ArgumentParser(
        prog="prepare_tum",
        description="Download and lay out the TUM RGB-D freiburg3 long_office_household dataset.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory for raw TUM archives. In a source checkout, defaults to the wrapped script default. "
            "In an installed package, defaults to ./datasets/tum/raw."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for prepared data. In a source checkout, defaults to the wrapped script default. "
            "In an installed package, defaults to ./datasets/converted."
        ),
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download archives even when they exist.")
    parser.add_argument("--download-only", action="store_true", help="Download archives but skip dataset layout.")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    if not is_source_checkout:
        raw_dir = raw_dir or installed_raw_dir("tum")
        output_dir = output_dir or installed_output_dir()

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
