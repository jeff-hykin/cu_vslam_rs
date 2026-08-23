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

import json
import logging
import math
import pathlib
import shutil
import sys
from typing import Any

import numpy as np
from rosbags import highlevel
from pytransform3d import transform_manager

from cuvslam_tools.common import edex
from . import (
    Config,
    get_first_message,
    log_rosbag_info,
    open_rosbag_reader,
    rosbag_image_extraction,
    rosbag_tf_extraction,
    rosbag_urdf_extraction,
)

# fmt: off
CV_FROM_ROS = np.array(
    [
        [ 0, -1, 0, 0],
        [ 0,  0, 1, 0],
        [-1,  0, 0, 0],
        [ 0,  0, 0, 1],
    ]
)
ROS_FROM_CV = np.linalg.inv(CV_FROM_ROS)
ROS_CAMERA_OPTICAL_FROM_CV_CAMERA_OPTICAL = np.array(
    [
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ]
)
# fmt: on


def pose_matrix_to_edex_format(pose_matrix: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose matrix to the 3x4 format that edex expects."""
    # Do some safety checks to make sure that the matrix is a valid pose matrix.
    assert pose_matrix.shape == (4, 4)
    assert math.isclose(np.linalg.det(pose_matrix), 1.0)
    assert pose_matrix[3, 3] == 1.0
    return pose_matrix[0:3, :]


def extract_imu_stream(reader: highlevel.AnyReader, config: Config):
    """Extract all imu messages from the bag and store to disk."""
    imu_path = config.output_path / "imu.jsonl"
    logging.info(f"Writing imu data '{imu_path}'.")
    with open(imu_path, "w", encoding="utf-8") as file:
        connections = [c for c in reader.connections if c.topic == config.imu_topic]
        for connection, _, rawdata in reader.messages(connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            imu_data = {
                "timestamp": msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                "AngularVelocityX": msg.angular_velocity.x,
                "AngularVelocityY": msg.angular_velocity.y,
                "AngularVelocityZ": msg.angular_velocity.z,
                "LinearAccelerationX": msg.linear_acceleration.x,
                "LinearAccelerationY": msg.linear_acceleration.y,
                "LinearAccelerationZ": msg.linear_acceleration.z,
            }

            json.dump(imu_data, file)
            file.write("\n")


def get_imu_metadata(
    imu_msg: Any, tf_manager: transform_manager.TransformManager, config: Config
) -> edex.IMU:
    """Create the imu metadata needed for the edex file."""
    if not config.imu_frame:
        config.imu_frame = imu_msg.header.frame_id

    ros_rig_from_ros_imu = tf_manager.get_transform(config.imu_frame, config.rig_frame)
    cv_rig_from_cv_imu = CV_FROM_ROS @ ros_rig_from_ros_imu @ ROS_FROM_CV
    return edex.IMU(
        g=np.array([0.0, -9.81, 0.0], dtype=np.float32),
        measurements="imu.jsonl",
        transform=pose_matrix_to_edex_format(cv_rig_from_cv_imu),
    )


def get_camera_metadata(
    camera_idx: int,
    camera_msg: Any,
    tf_manager: transform_manager.TransformManager,
    config: Config,
) -> edex.Camera:
    """Create the camera metadata needed for the edex file."""
    if config.camera_optical_frames:
        camera_optical_frame = config.camera_optical_frames[camera_idx]
    else:
        camera_optical_frame = camera_msg.header.frame_id

    ros_rig_from_ros_camera_optical = tf_manager.get_transform(
        camera_optical_frame, config.rig_frame
    )
    cv_rig_from_cv_camera_optical = (
        CV_FROM_ROS
        @ ros_rig_from_ros_camera_optical
        @ ROS_CAMERA_OPTICAL_FROM_CV_CAMERA_OPTICAL
    )

    return edex.Camera(
        intrinsics=rosbag_image_extraction.get_camera_intrinsics(camera_msg, config),
        transform=pose_matrix_to_edex_format(cv_rig_from_cv_camera_optical),
    )


def extract_edex_metadata(
    reader: highlevel.AnyReader,
    tf_manager: transform_manager.TransformManager,
    config: Config,
    num_frames: int,
):
    """Create the edex metadata file."""
    camera_info_msgs = get_first_message(reader, config.camera_info_topics)
    cameras_metadata = []
    for idx, msg in enumerate(camera_info_msgs):
        camera_metadata = get_camera_metadata(idx, msg, tf_manager, config)
        cameras_metadata.append(camera_metadata)

    if config.imu_topic:
        imu_msg = get_first_message(reader, [config.imu_topic])[0]
        imu_metadata = get_imu_metadata(imu_msg, tf_manager, config)
    else:
        imu_metadata = None

    sequence_paths = [
        rosbag_image_extraction.get_image_path(pathlib.Path("images"), topic, 0)
        for topic in config.image_topics
    ]

    edex_header = edex.EDEXHeader(
        version="0.9",
        frame_start=0,
        frame_end=num_frames,
        cameras=cameras_metadata,
        imu=imu_metadata,
    )
    edex_body = edex.EDEXBody(
        frame_metadata="frame_metadata.jsonl",
        sequence=sequence_paths,
    )
    edex_metadata = edex.EDEXMetadata(header=edex_header, body=edex_body)

    edex_metadata_path = config.output_path / "edex"
    logging.info(f"Writing edex metadata to '{edex_metadata_path}'.")
    edex_metadata.write(edex_metadata_path)


def extract_edex(config: Config):
    """Extract the entire edex using the config."""
    # Create edex path.
    shutil.rmtree(config.output_path / "images", ignore_errors=True)
    (config.output_path / "edex").unlink(missing_ok=True)
    (config.output_path / "frame_metadata.jsonl").unlink(missing_ok=True)
    (config.output_path / "robot.urdf").unlink(missing_ok=True)
    (config.output_path / "imu.jsonl").unlink(missing_ok=True)
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Extract the URDF from the rosbag.
    tf_manager = rosbag_tf_extraction.get_static_transform_manager_from_bag(
        config.rosbag_path,
        config.ros_distribution,
    )
    logging.info(
        f"Found the following frames in rosbag:\n"
        + "\n".join([f"\t- {node}" for node in tf_manager.nodes])
    )
    if config.rig_frame not in tf_manager.nodes:
        logging.error(
            f"Rig frame '{config.rig_frame}' not found in rosbag. Select one from the list above and set rig_frame in the config."
        )
        sys.exit(1)
    urdf_content = rosbag_urdf_extraction.get_urdf_from_tf_manager("robot", tf_manager)
    (config.output_path / "robot.urdf").write_text(urdf_content)

    with open_rosbag_reader(config.rosbag_path, config.ros_distribution) as reader:
        log_rosbag_info(reader)

        # Do some quick checks that all the required data is present in the rosbag.
        # If not we ignore the inexistent topics.
        bag_topics = [c.topic for c in reader.connections]
        image_topics = []
        camera_info_topics = []
        camera_optical_frames = []
        for idx, (image_topic, info_topic) in enumerate(
            zip(config.image_topics, config.camera_info_topics)
        ):
            if image_topic not in bag_topics:
                logging.warning(
                    f"Could not find topic '{image_topic}' in rosbag. Ignoring it."
                )
            elif info_topic not in bag_topics:
                logging.warning(
                    f"Could not find topic '{info_topic}' in rosbag. Ignoring it."
                )
            else:
                image_topics.append(image_topic)
                camera_info_topics.append(info_topic)
                if config.camera_optical_frames:
                    camera_optical_frames.append(config.camera_optical_frames[idx])

        config.image_topics = image_topics
        config.camera_info_topics = camera_info_topics
        config.camera_optical_frames = camera_optical_frames or None

        # Extract all data and store to disk.
        num_frames = rosbag_image_extraction.run_image_extraction(reader, config)
        extract_edex_metadata(reader, tf_manager, config, num_frames)

        if config.imu_topic:
            extract_imu_stream(reader, config)

    logging.info(f"Finished extracting edex to '{config.output_path}'.")
