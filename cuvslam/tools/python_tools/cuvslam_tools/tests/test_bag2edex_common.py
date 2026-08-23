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

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cuvslam_tools.bag2edex import common
from cuvslam_tools.bag2edex import rosbag_urdf_extraction


class TestBag2EdexCommon(unittest.TestCase):
    def test_sanitize_rosbag2_message_definitions_removes_invalid_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sqlite_path = Path(temp_dir) / "bag.db3"
            with sqlite3.connect(sqlite_path) as connection:
                connection.execute(
                    "CREATE TABLE message_definitions ("
                    "id INTEGER PRIMARY KEY, "
                    "topic_type TEXT NOT NULL, "
                    "encoding TEXT NOT NULL, "
                    "encoded_message_definition TEXT NOT NULL, "
                    "type_description_hash TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO message_definitions VALUES (?, ?, ?, ?, ?)",
                    (1, "tf2_msgs/msg/TFMessage", "ros2msg", "bad definition", ""),
                )
                connection.execute(
                    "INSERT INTO message_definitions VALUES (?, ?, ?, ?, ?)",
                    (2, "custom_msgs/msg/Metadata", "", "", ""),
                )
                connection.commit()

            self.assertEqual(common._sanitize_rosbag2_message_definitions(sqlite_path), 2)

            with sqlite3.connect(sqlite_path) as connection:
                rows = connection.execute("SELECT id FROM message_definitions").fetchall()

        self.assertEqual(rows, [])

    def test_extract_urdf_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "missing" / "urdf_out"
            with mock.patch.object(
                rosbag_urdf_extraction.rosbag_tf_extraction,
                "get_static_transform_manager_from_bag",
                return_value=mock.Mock(nodes=[]),
            ), mock.patch.object(
                rosbag_urdf_extraction,
                "get_urdf_from_tf_manager",
                return_value="<robot name=\"robot\" />",
            ):
                rosbag_urdf_extraction.extract_urdf(
                    temp_path / "bag",
                    "jazzy",
                    output_path,
                )

            self.assertEqual(
                (output_path / "robot.urdf").read_text(encoding="utf-8"),
                "<robot name=\"robot\" />",
            )


if __name__ == "__main__":
    unittest.main()
