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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Constants
DEFAULT_RESOLUTION = (640, 360)
DEFAULT_FPS = 30
DEFAULT_IMU_FREQUENCY = 200

# IMU noise parameters for RealSense
IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04


def opengl_to_opencv_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert from OpenGL coordinate system to OpenCV coordinate system.

    Args:
        rotation: 3x3 rotation matrix in OpenGL coordinates
        translation: 3x1 translation vector in OpenGL coordinates

    Returns:
        Tuple of (rotation_opencv, translation_opencv)
    """
    transform_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
    rotation_opencv = transform_matrix @ rotation @ transform_matrix.T
    translation_opencv = transform_matrix @ translation
    return rotation_opencv, translation_opencv


def transform_to_pose(transform_matrix=None) -> vslam.Pose:
    """Convert a transformation matrix to a vslam.Pose object.

    Args:
        transform_matrix: Either a RealSense transform object or a list of lists
                         representing the transformation matrix

    Returns:
        vslam.Pose object
    """
    if isinstance(transform_matrix, List):
        # Handle list of lists format from YAML
        rotation = np.array([row[:3] for row in transform_matrix])
        translation = np.array([row[3] for row in transform_matrix])
        rotation_opencv, translation_vec = opengl_to_opencv_transform(
            rotation, translation
        )
        rotation_quat = Rotation.from_matrix(rotation_opencv).as_quat()
    elif transform_matrix:
        # Handle RealSense transform object
        rotation_matrix = np.array(transform_matrix.rotation).reshape([3, 3])
        translation_vec = transform_matrix.translation
        rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
        return vslam.Pose(rotation=rotation_quat, translation=translation_vec)
    else:
        # Default identity transform
        rotation_matrix = np.eye(3)
        translation_vec = [0] * 3
        rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()

    return vslam.Pose(rotation=rotation_quat, translation=translation_vec)


def rig_from_imu_pose(rs_transform) -> vslam.Pose:
    """Convert RealSense IMU extrinsics to cuVSLAM Pose.

    Args:
        rs_transform: RealSense extrinsics transform object

    Returns:
        vslam.Pose object
    """
    return vslam.Pose(
        rotation=Rotation.identity().as_quat(),
        translation=rs_transform.translation
    )


def get_rs_camera(
    rs_intrinsics, transform_matrix: Optional[Any] = None
) -> vslam.Camera:
    """Create a Camera object from RealSense intrinsics.

    Args:
        rs_intrinsics: RealSense intrinsics object
        transform_matrix: Optional transformation matrix for camera pose

    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = rs_intrinsics.fx, rs_intrinsics.fy
    cam.principal = rs_intrinsics.ppx, rs_intrinsics.ppy
    cam.size = rs_intrinsics.width, rs_intrinsics.height

    if transform_matrix is not None:
        cam.rig_from_camera = transform_to_pose(transform_matrix)

    return cam


def get_rs_imu(
    imu_extrinsics, frequency: int = DEFAULT_IMU_FREQUENCY
) -> vslam.ImuCalibration:
    """Create an IMU calibration object from RealSense extrinsics.

    Args:
        imu_extrinsics: RealSense IMU extrinsics
        frequency: IMU sampling frequency in Hz

    Returns:
        vslam.ImuCalibration object
    """
    imu = vslam.ImuCalibration()
    imu.rig_from_imu = rig_from_imu_pose(imu_extrinsics)
    imu.gyroscope_noise_density = IMU_GYROSCOPE_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYROSCOPE_RANDOM_WALK
    imu.accelerometer_noise_density = IMU_ACCELEROMETER_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACCELEROMETER_RANDOM_WALK
    imu.frequency = frequency
    return imu


def setup_pipeline(
    serial_number: str,
    resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
    fps: int = DEFAULT_FPS
) -> Tuple[rs.pipeline, rs.config]:
    """Set up and configure a RealSense pipeline.

    Args:
        serial_number: Camera serial number
        resolution: Camera resolution as (width, height)
        fps: Frames per second

    Returns:
        Tuple of (pipeline, config)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(
        rs.stream.infrared, 1, resolution[0], resolution[1], rs.format.y8, fps
    )
    config.enable_stream(
        rs.stream.infrared, 2, resolution[0], resolution[1], rs.format.y8, fps
    )
    return pipeline, config


def get_camera_intrinsics(
    pipeline: rs.pipeline, config: rs.config
) -> Tuple[Any, Any]:
    """Get camera intrinsics from a RealSense pipeline.

    Args:
        pipeline: RealSense pipeline
        config: RealSense config

    Returns:
        Tuple of (left_intrinsics, right_intrinsics)
    """
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    left_intrinsics = frames[0].profile.as_video_stream_profile().intrinsics
    right_intrinsics = frames[1].profile.as_video_stream_profile().intrinsics
    pipeline.stop()
    return left_intrinsics, right_intrinsics


def configure_device(
    pipeline: rs.pipeline, config: rs.config, is_master: bool = False
) -> None:
    """Configure device settings like IR emitter and sync mode.

    Args:
        pipeline: RealSense pipeline
        config: RealSense config
        is_master: Whether this device is the master for synchronization
    """
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    depth_sensor = device.query_sensors()[0]

    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        # First camera is master, others are slave
        sync_mode = 1 if is_master else 2
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)


def get_rs_stereo_rig(
    camera_params: Dict[str, Dict[str, Any]]
) -> vslam.Rig:
    """Create a stereo Rig object from RealSense parameters.

    Args:
        camera_params: Dictionary containing camera parameters

    Returns:
        vslam.Rig object
    """
    rig = vslam.Rig()

    cameras = [get_rs_camera(camera_params['left']['intrinsics'])]

    if 'right' in camera_params:
        cameras.append(
            get_rs_camera(
                camera_params['right']['intrinsics'],
                camera_params['right']['extrinsics']
            )
        )

    rig.cameras = cameras
    return rig


def get_rs_multi_rig(
    camera_params: Dict[str, Dict[str, Dict[str, Any]]]
) -> vslam.Rig:
    """Create a multi-camera Rig object from RealSense parameters.

    Args:
        camera_params: Dictionary containing parameters for multiple cameras

    Returns:
        vslam.Rig object with multiple stereo cameras
    """
    rig = vslam.Rig()
    cameras_list = []

    for i in range(1, len(camera_params) + 1):
        camera_idx = f"camera_{i}"

        # Add left camera
        cameras_list.append(
            get_rs_camera(
                camera_params[camera_idx]['left']['intrinsics'],
                camera_params[camera_idx]['left']['extrinsics']
            )
        )

        # Add right camera
        cameras_list.append(
            get_rs_camera(
                camera_params[camera_idx]['right']['intrinsics'],
                camera_params[camera_idx]['right']['extrinsics']
            )
        )

    rig.cameras = cameras_list
    return rig


def get_rs_vio_rig(
    camera_params: Dict[str, Dict[str, Any]]
) -> vslam.Rig:
    """Create a VIO Rig object with cameras and IMU from RealSense parameters.

    Args:
        camera_params: Dictionary containing camera and IMU parameters

    Returns:
        vslam.Rig object with cameras and IMU
    """
    rig = vslam.Rig()
    rig.cameras = [
        get_rs_camera(camera_params['left']['intrinsics']),
        get_rs_camera(
            camera_params['right']['intrinsics'],
            camera_params['right']['extrinsics']
        )
    ]
    rig.imus = [get_rs_imu(camera_params['imu']['cam_from_imu'])]
    return rig
