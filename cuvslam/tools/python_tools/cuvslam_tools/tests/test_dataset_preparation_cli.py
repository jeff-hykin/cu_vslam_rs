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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.dataset_preparation.euroc import cli as euroc_cli
from cuvslam_tools.dataset_preparation.tartan import cli as tartan_cli
from cuvslam_tools.dataset_preparation.tartan import download as tartan_download
from cuvslam_tools.dataset_preparation.tartan import stage_sequences as tartan_stage
from cuvslam_tools.dataset_preparation.tum import cli as tum_cli


class TestDatasetPreparationCli(unittest.TestCase):
    def test_prepare_euroc_installed_defaults(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        script = Path("/pkg/euroc/prepare_euroc.sh")

        with mock.patch.object(
            euroc_cli,
            "resolve_prepare_script",
            return_value=(script, False),
        ), mock.patch.object(euroc_cli.subprocess, "run", return_value=completed) as run:
            self.assertEqual(euroc_cli.main(["--download-only"]), 0)

        run.assert_called_once_with(
            [
                "bash",
                str(script),
                "--raw-dir",
                str(Path.cwd() / "datasets" / "euroc" / "raw"),
                "--output-dir",
                str(Path.cwd() / "datasets" / "converted"),
                "--download-only",
            ],
            check=False,
        )

    def test_prepare_tartan_passes_variant_and_paths(self):
        completed = subprocess.CompletedProcess(args=[], returncode=7)
        script = Path("/pkg/tartan/prepare_tartan.sh")

        with mock.patch.object(
            tartan_cli,
            "resolve_prepare_script",
            return_value=(script, True),
        ), mock.patch.object(tartan_cli.subprocess, "run", return_value=completed) as run:
            self.assertEqual(
                tartan_cli.main(
                    [
                        "--variant",
                        "multicamera",
                        "--raw-dir",
                        "/raw",
                        "--output-dir",
                        "/converted",
                        "--force-download",
                    ]
                ),
                7,
            )

        run.assert_called_once()
        command = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(
            command,
            [
                "bash",
                str(script),
                "--variant",
                "multicamera",
                "--raw-dir",
                "/raw",
                "--output-dir",
                "/converted",
                "--force-download",
            ],
        )
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["env"]["PYTHON_BIN"], sys.executable)

    def test_prepare_tum_source_defaults_defer_to_shell_script(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        script = Path("/src/tum/prepare_tum.sh")

        with mock.patch.object(
            tum_cli,
            "resolve_prepare_script",
            return_value=(script, True),
        ), mock.patch.object(tum_cli.subprocess, "run", return_value=completed) as run:
            self.assertEqual(tum_cli.main([]), 0)

        run.assert_called_once_with(["bash", str(script)], check=False)

    def test_tartan_download_variants_share_download_implementation(self):
        expected_variants = {
            "multicamera": {
                "modality": ["image", "meta"],
                "camera_name": [
                    "lcam_front",
                    "rcam_front",
                    "lcam_left",
                    "rcam_left",
                    "lcam_right",
                    "rcam_right",
                    "lcam_back",
                    "rcam_back",
                    "lcam_top",
                    "rcam_top",
                    "lcam_bottom",
                    "rcam_bottom",
                ],
            },
            "multisensor": {
                "modality": ["image", "depth", "imu", "meta"],
                "camera_name": ["lcam_front", "lcam_back"],
            },
        }

        for variant, config in expected_variants.items():
            with self.subTest(variant=variant):
                fake_tartanair = mock.Mock()

                with mock.patch.dict("sys.modules", {"tartanair": fake_tartanair}), mock.patch.object(
                    tartan_download.os,
                    "makedirs",
                ) as makedirs:
                    tartan_download.download_tartan_ground(variant, "/tmp/tartan_ground")

                makedirs.assert_called_once_with("/tmp/tartan_ground", exist_ok=True)
                fake_tartanair.init.assert_called_once_with("/tmp/tartan_ground")
                fake_tartanair.download_ground.assert_called_once_with(
                    env=["OldTownFall"],
                    version=["anymal"],
                    modality=config["modality"],
                    traj=["P2000"],
                    camera_name=config["camera_name"],
                    unzip=True,
                )

    def test_tartan_download_main_passes_variant_and_data_root(self):
        with mock.patch.object(tartan_download, "download_tartan_ground") as download_tartan_ground:
            self.assertEqual(
                tartan_download.main(["--variant", "multicamera", "--data-root", "/tmp/tartan_ground"]),
                0,
            )

        download_tartan_ground.assert_called_once_with("multicamera", "/tmp/tartan_ground")

    def test_tartanground_staging_maps_stereo_pair_to_classic_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_sequence = temp_path / "raw" / "OldTownFall" / "Data_anymal" / "P2000"
            (raw_sequence / "image_lcam_front").mkdir(parents=True)
            (raw_sequence / "image_rcam_front").mkdir()
            (raw_sequence / "image_lcam_front" / "000000_lcam_front.png").write_text("left")
            (raw_sequence / "image_rcam_front" / "000000_rcam_front.png").write_text("right")
            (raw_sequence / "pose_lcam_front.txt").write_text("left pose")
            (raw_sequence / "pose_rcam_front.txt").write_text("right pose")

            staged = tartan_stage.stage_sequences(temp_path / "raw", temp_path / "converted")

            self.assertEqual([temp_path / "converted" / "OldTownFall" / "Data_anymal" / "P2000_front"], staged)
            staged_sequence = staged[0]
            self.assertEqual((staged_sequence / "image_left" / "000000_left.png").read_text(), "left")
            self.assertEqual((staged_sequence / "image_right" / "000000_right.png").read_text(), "right")
            self.assertEqual((staged_sequence / "pose_left.txt").read_text(), "left pose")
            self.assertEqual((staged_sequence / "pose_right.txt").read_text(), "right pose")


if __name__ == "__main__":
    unittest.main()
