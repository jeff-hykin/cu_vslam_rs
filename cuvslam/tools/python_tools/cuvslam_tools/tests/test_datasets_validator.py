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

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.datasets_validator import cli


class TestDatasetsValidator(unittest.TestCase):
    def test_load_validation_config_rejects_non_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "validation.yaml"
            config_path.write_text("- not-a-mapping\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Validation config must be a YAML mapping"):
                cli._load_validation_config(str(config_path))

    def test_load_validation_config_rejects_invalid_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "validation.yaml"
            config_path.write_text("reporter_configs: [", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Validation config is not valid YAML"):
                cli._load_validation_config(str(config_path))

    def test_is_trivial_change_falls_back_to_validation_on_git_failure(self):
        with mock.patch.object(cli.subprocess, "run", side_effect=FileNotFoundError("git")):
            self.assertFalse(cli._is_trivial_change({"trivial_change_paths": ["docs/**"]}, "origin/main"))

        error = subprocess.CalledProcessError(128, ["git", "diff"])
        with mock.patch.object(cli.subprocess, "run", side_effect=error):
            self.assertFalse(cli._is_trivial_change({"trivial_change_paths": ["docs/**"]}, "origin/main"))

    def test_check_metrics_rejects_missing_metric(self):
        rows = [{"sequence_name": "seq-a", "ate": "1.0"}, {"sequence_name": "seq-b", "other": "1.0"}]
        with self.assertRaisesRegex(ValueError, "ate missing from rows: seq-b"):
            cli._check_metrics(rows, [{"metric": "ate", "max_value": 2.0}])

    def test_check_metrics_rejects_metric_without_values(self):
        with self.assertRaisesRegex(ValueError, "ate missing from rows: <unknown>"):
            cli._check_metrics([{"ate": ""}], [{"metric": "ate", "max_value": 2.0}])

    def test_check_metrics_keeps_min_max_validation(self):
        rows = [{"ate": "1.0"}, {"ate": "3.0"}]
        with self.assertRaisesRegex(ValueError, "ate max check failed"):
            cli._check_metrics(rows, [{"metric": "ate", "max_value": 2.0}])

        with self.assertRaisesRegex(ValueError, "ate min check failed"):
            cli._check_metrics(rows, [{"metric": "ate", "min_value": 2.0}])


if __name__ == "__main__":
    unittest.main()
