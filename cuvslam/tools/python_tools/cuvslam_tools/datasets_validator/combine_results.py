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

"""Combine reporter JSON statistics into a single CSV validation summary."""

import csv
import json
import os
from pathlib import Path


def load_report_stats(report_dir: str) -> list[dict]:
    """Load all sequence statistics written by one reporter run."""
    stats_path = Path(report_dir) / "stats" / "all_stats.json"
    with open(stats_path) as f:
        return json.load(f)


def combine_report_stats(report_dirs: list[str], output_csv: str) -> list[dict]:
    """Combine reporter stats directories and write a CSV summary table."""
    rows = []
    for report_dir in report_dirs:
        for stat in load_report_stats(report_dir):
            row = {"report_dir": report_dir}
            row.update(stat)
            rows.append(row)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
