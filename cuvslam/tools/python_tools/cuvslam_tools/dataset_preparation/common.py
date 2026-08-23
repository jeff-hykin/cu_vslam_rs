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

"""Shared helpers for installed dataset preparation console scripts."""

from pathlib import Path
from typing import Optional


def source_prepare_script(current_file: str, dataset_name: str, script_name: str) -> Optional[Path]:
    """Return the repository-local package script when running from a source checkout."""
    current = Path(current_file).resolve()
    for parent in current.parents:
        candidate = (
            parent
            / "tools"
            / "python_tools"
            / "cuvslam_tools"
            / "dataset_preparation"
            / dataset_name
            / script_name
        )
        if candidate.exists():
            return candidate
    return None


def bundled_prepare_script(current_file: str, script_name: str) -> Path:
    """Return the preparation script bundled next to a dataset CLI module."""
    return Path(current_file).resolve().with_name(script_name)


def resolve_prepare_script(current_file: str, dataset_name: str, script_name: str) -> tuple[Path, bool]:
    """Return the best script path and whether it came from a source checkout."""
    source_script = source_prepare_script(current_file, dataset_name, script_name)
    if source_script is not None:
        return source_script, True
    return bundled_prepare_script(current_file, script_name), False


def installed_raw_dir(dataset_name: str) -> Path:
    """Default raw-data directory for installed package runs."""
    return Path.cwd() / "datasets" / dataset_name / "raw"


def installed_output_dir() -> Path:
    """Default converted-data directory for installed package runs."""
    return Path.cwd() / "datasets" / "converted"
