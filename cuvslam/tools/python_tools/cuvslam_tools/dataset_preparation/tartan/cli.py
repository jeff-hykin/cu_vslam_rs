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

"""Installed console-script wrapper for the TartanGround preparation flow."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from cuvslam_tools.dataset_preparation.common import (
    installed_output_dir,
    installed_raw_dir,
    resolve_prepare_script,
)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the TartanGround preparation shell workflow from the console script."""
    prepare_script, is_source_checkout = resolve_prepare_script(__file__, "tartan", "prepare_tartan.sh")

    parser = argparse.ArgumentParser(
        prog="prepare_tartan",
        description=(
            "Download TartanGround data and convert TartanGround stereo pairs or "
            "classic TartanAir-layout sequences to EDEX."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory for raw Tartan downloads. In a source checkout, defaults to the wrapped script default. "
            "In an installed package, defaults to ./datasets/tartan/raw."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for converted data. In a source checkout, defaults to the wrapped script default. "
            "In an installed package, defaults to ./datasets/converted."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["multisensor", "multicamera"],
        default="multisensor",
        help="TartanGround download variant. Both variants include metadata; use multicamera for EDEX conversion.",
    )
    parser.add_argument("--force-download", action="store_true", help="Remove existing download/conversion output first.")
    parser.add_argument("--download-only", action="store_true", help="Download but skip conversion.")
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    if not is_source_checkout:
        raw_dir = raw_dir or installed_raw_dir("tartan")
        output_dir = output_dir or installed_output_dir()

    command = ["bash", str(prepare_script), "--variant", args.variant]
    if raw_dir is not None:
        command.extend(["--raw-dir", str(raw_dir)])
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if args.force_download:
        command.append("--force-download")
    if args.download_only:
        command.append("--download-only")

    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    return subprocess.run(command, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
