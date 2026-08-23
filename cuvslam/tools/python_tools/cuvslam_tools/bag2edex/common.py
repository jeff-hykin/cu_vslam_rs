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

"""Shared ROS bag to EDEX configuration and helper utilities."""

from contextlib import ExitStack, contextmanager
import logging
import pathlib
import shutil
import sqlite3
import tempfile
from typing import Optional

import pydantic
from rosbags import highlevel
from rosbags.typesys import (
    get_types_from_msg,
    get_typestore,
    Stores,
)
from rosbags.typesys.store import Typestore


ROS_TYPESTORES = {
    "empty": Stores.EMPTY,
    "latest": Stores.LATEST,
    "noetic": Stores.ROS1_NOETIC,
    "dashing": Stores.ROS2_DASHING,
    "eloquent": Stores.ROS2_ELOQUENT,
    "foxy": Stores.ROS2_FOXY,
    "galactic": Stores.ROS2_GALACTIC,
    "humble": Stores.ROS2_HUMBLE,
    "iron": Stores.ROS2_IRON,
    "jazzy": Stores.ROS2_JAZZY,
    "kilted": Stores.ROS2_KILTED,
}


class Config(pydantic.BaseModel):
    """Configuration for the bag to edex converter."""

    # Path of the rosbag used for extraction.
    rosbag_path: pathlib.Path
    # Path of the generated edex, urdf, images, videos, etc.
    output_path: pathlib.Path
    # Topics used to get the camera's intrinsics (and extrinsics if frames are not set explicitly).
    camera_info_topics: list[str]
    # Topics used to extract images. Must be the same length as camera_info_topics.
    image_topics: list[str]
    # Topic used to get IMU measurements.
    imu_topic: Optional[str] = None
    # Frames used to acquire the extrinsics. If not set the frames from the messages will be used:
    rig_frame: str
    camera_optical_frames: Optional[list[str]] = None
    imu_frame: Optional[str] = None
    # Number of workers used in image extraction.
    num_workers: int = -1
    # Threshold used for syncing images in the same frame.
    sync_threshold_ns: int = int(0.001 * 10**9)
    # Width and height used to resize the extracted images.
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    output_format: Optional[str] = None
    # ROS distribution used to extract the rosbag.
    ros_distribution: str = "humble"

    @pydantic.model_validator(mode="after")
    def check_fields(self):
        """Preprocess the values and then validate that all members are valid."""
        if not self.rosbag_path.exists():
            raise ValueError(f"Path '{self.rosbag_path}' does not exist")
        if len(self.image_topics) != len(self.camera_info_topics):
            raise ValueError("Need same number of image topics as camera info topics.")
        if self.camera_optical_frames:
            if len(self.camera_optical_frames) != len(self.camera_info_topics):
                raise ValueError(
                    "Need same number of camera optical frames as camera info topics."
                )
        return self


def get_typestore_from_ros_distribution(ros_distribution: str) -> Typestore:
    """Get the typestore from the ROS distribution."""
    if ros_distribution not in ROS_TYPESTORES:
        raise ValueError(f"Unknown ROS distribution: {ros_distribution}")
    return get_typestore(ROS_TYPESTORES[ros_distribution])


def _message_definition_is_valid(
    topic_type: str,
    encoding: str,
    message_definition: str,
    type_description_hash: str,
) -> bool:
    """Return whether rosbags can parse and validate one embedded message definition."""
    if encoding != "ros2msg" or not message_definition or not type_description_hash:
        return False

    try:
        types = get_types_from_msg(message_definition, topic_type)
        store = Typestore()
        store.register(types)
        return type_description_hash == store.hash_rihs01(topic_type)
    except Exception:
        return False


def _sanitize_rosbag2_message_definitions(sqlite_path: pathlib.Path) -> int:
    """Delete invalid embedded ROS2 message definitions from a temporary sqlite bag copy."""
    with sqlite3.connect(sqlite_path) as connection:
        cursor = connection.cursor()
        has_message_definitions = cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='message_definitions'"
        ).fetchone()[0]
        if not has_message_definitions:
            return 0

        invalid_ids = []
        for row_id, topic_type, encoding, message_definition, type_description_hash in cursor.execute(
            "SELECT id, topic_type, encoding, encoded_message_definition, type_description_hash "
            "FROM message_definitions"
        ):
            if not _message_definition_is_valid(
                topic_type,
                encoding,
                message_definition,
                type_description_hash,
            ):
                invalid_ids.append(row_id)

        if invalid_ids:
            cursor.executemany(
                "DELETE FROM message_definitions WHERE id = ?",
                [(row_id,) for row_id in invalid_ids],
            )
            connection.commit()

    return len(invalid_ids)


def _copy_and_sanitize_rosbag(rosbag_path: pathlib.Path, temp_root: pathlib.Path) -> pathlib.Path:
    """Copy a rosbag directory to temp_root and sanitize its sqlite metadata."""
    sanitized_path = temp_root / rosbag_path.name
    shutil.copytree(rosbag_path, sanitized_path)

    n_removed = 0
    for sqlite_path in sanitized_path.glob("*.db3"):
        n_removed += _sanitize_rosbag2_message_definitions(sqlite_path)

    logging.warning(
        "Using a temporary rosbag copy with %d invalid embedded message definition(s) removed. "
        "The source bag is unchanged.",
        n_removed,
    )
    return sanitized_path


@contextmanager
def open_rosbag_reader(rosbag_path: pathlib.Path, ros_distribution: str):
    """Open a rosbag reader, falling back for schema-v4 bags with invalid embedded definitions."""
    typestore = get_typestore_from_ros_distribution(ros_distribution)
    stack = ExitStack()
    try:
        reader = stack.enter_context(
            highlevel.AnyReader(paths=[rosbag_path], default_typestore=typestore)
        )
    except AssertionError as exc:
        stack.close()
        if "Failed to parse" not in str(exc):
            raise
        logging.warning(
            "rosbags could not parse embedded message definitions in '%s': %s",
            rosbag_path,
            exc,
        )
    else:
        with stack:
            yield reader
        return

    with tempfile.TemporaryDirectory(prefix="cuvslam_rosbag_") as temp_dir:
        sanitized_path = _copy_and_sanitize_rosbag(rosbag_path, pathlib.Path(temp_dir))
        with highlevel.AnyReader(
            paths=[sanitized_path],
            default_typestore=typestore,
        ) as reader:
            yield reader


def get_first_message(reader: highlevel.AnyReader, topics: list[str]) -> list[object]:
    """Get the first message of every topic."""
    connections = [c for c in reader.connections if c.topic in topics]
    topic_and_first_msg = {}
    for connection, _, rawdata in reader.messages(connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        topic_and_first_msg[connection.topic] = msg
        if len(topic_and_first_msg) == len(topics):
            break

    # Raise a clear error for any topic that had no messages.
    missing = [t for t in topics if t not in topic_and_first_msg]
    if missing:
        raise ValueError(
            f"get_first_message: no messages found for topic(s) {missing}. "
            "Cannot build camera/IMU metadata. Check that the topic names in "
            "your config match those in the bag (run 'ros2 bag info <bag>' to list topics)."
        )

    # Generate the list in the same order as the input topics.
    return [topic_and_first_msg[topic] for topic in topics]


def log_rosbag_info(reader: highlevel.AnyReader):
    """Log the topics and message types of all message channels in the rosbag."""
    logs = [f"\t- {c.topic}: {c.msgtype}" for c in reader.connections]
    logs = sorted(logs)
    # pylint: disable=logging-not-lazy
    logging.info("Found the following topics in rosbag:\n" + "\n".join(logs))
