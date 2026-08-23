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

import logging
import pathlib
import shutil

from rosbags import highlevel

from . import (
    Config,
    log_rosbag_info,
    open_rosbag_reader,
)


def get_video_path(base_path: pathlib.Path, topic: str) -> pathlib.Path:
    """Get the path to the video with the given topic."""
    # Remove leading slash.
    topic = topic.lstrip("/")
    topic = topic.replace("/", "_")
    return base_path / f"{topic}.h264"


def extract_video(reader: highlevel.AnyReader, topic: str, video_path: pathlib.Path):
    """Extract an image topic from a rosbag and store as an h264 encoded video."""
    # Store topic as an h264 encoded video to disk.
    logging.info(f"Writing h264 video to '{video_path}'.")
    video_path.parent.mkdir(parents=True, exist_ok=True)

    connections = [c for c in reader.connections if c.topic == topic]
    with video_path.open("wb") as file:
        for connection, _, rawdata in reader.messages(connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            file.write(msg.data.tobytes())


def extract_videos(config: Config):
    """Extract only the videos using the config."""
    # Create videos path.
    shutil.rmtree(config.output_path / "videos", ignore_errors=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    with open_rosbag_reader(config.rosbag_path, config.ros_distribution) as reader:
        log_rosbag_info(reader)

        # Extract the videos from available topics.
        topics = [c.topic for c in reader.connections if c.topic in config.image_topics]
        for topic in topics:
            video_path = get_video_path(config.output_path / "videos", topic)
            extract_video(reader, topic, video_path)

    logging.info(f"Finished extracting videos to '{config.output_path}'.")
