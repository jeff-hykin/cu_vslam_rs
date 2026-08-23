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

"""Legacy tools/cuvslam_app entry point.

This file preserves the old source-tree path for CI and scripts while the
implementation lives in the installable cuvslam_tools package.
"""

import os
from pathlib import Path
import sys
from typing import List, Optional


def _add_python_tools_to_path() -> None:
    """Add the source-tree python_tools package to sys.path when present."""
    python_tools = Path(__file__).resolve().parents[1] / "python_tools"
    if python_tools.exists():
        sys.path.insert(0, str(python_tools))


def _option_value(argv: List[str], option: str) -> Optional[str]:
    """Return the value supplied for an option in split or equals form."""
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def _has_option(argv: List[str], option: str) -> bool:
    """Return whether an option is present in split or equals form."""
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _replace_option(argv: List[str], old: str, new: str) -> List[str]:
    """Replace one option name while preserving its value spelling."""
    translated = []
    for arg in argv:
        if arg == old:
            translated.append(new)
        elif arg.startswith(old + "="):
            translated.append(new + "=" + arg.split("=", 1)[1])
        else:
            translated.append(arg)
    return translated


def _expand_legacy_test_config(argv: List[str]) -> List[str]:
    """Resolve legacy reporter config names relative to CUVSLAM_DATASETS."""
    test_config = _option_value(argv, "--test_config")
    if not test_config:
        return argv

    config_path = Path(test_config)
    if config_path.is_absolute() or config_path.exists():
        return argv

    datasets_root = os.environ.get("CUVSLAM_DATASETS")
    if not datasets_root:
        return argv

    candidate = Path(datasets_root) / test_config
    if not candidate.exists():
        return argv

    expanded = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--test_config" and index + 1 < len(argv):
            expanded.extend([arg, str(candidate)])
            skip_next = True
        elif arg.startswith("--test_config="):
            expanded.append("--test_config=" + str(candidate))
        else:
            expanded.append(arg)
    return expanded


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch legacy cuvslam_app arguments to tracker or reporter CLIs."""
    _add_python_tools_to_path()

    if argv is None:
        argv = sys.argv[1:]
    argv = _replace_option(list(argv), "--stereo_edex", "--config_path")
    argv = _expand_legacy_test_config(argv)

    if _has_option(argv, "--test_config"):
        from cuvslam_tools.reporter.cli import main as reporter_main

        return reporter_main(argv)

    from cuvslam_tools.tracker.cli import main as tracker_main

    return tracker_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
