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

from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from pyorbbecsdk import Pipeline, OBFrameType, OBFormat

import cuvslam as vslam


def process_ir_frame(ir_frame: Any) -> Optional[np.ndarray]:
    """Process IR frame to grayscale numpy array.

    Args:
        ir_frame: Orbbec IR frame

    Returns:
        Grayscale image as numpy array, or None if processing fails
    """
    if ir_frame is None:
        return None

    ir_frame = ir_frame.as_video_frame()
    ir_data = np.asanyarray(ir_frame.get_data())
    width = ir_frame.get_width()
    height = ir_frame.get_height()
    ir_format = ir_frame.get_format()

    if ir_format == OBFormat.Y8:
        # 8-bit grayscale
        return ir_data.reshape((height, width)).astype(np.uint8)
    elif ir_format == OBFormat.Y16:
        # 16-bit, normalize to 8-bit
        ir_data = np.frombuffer(ir_data, dtype=np.uint16).reshape((height, width))
        return (ir_data / 256).astype(np.uint8)
    elif ir_format == OBFormat.MJPG:
        # Decode JPEG using PIL
        try:
            pil_image = Image.open(BytesIO(ir_data.tobytes()))
            return np.array(pil_image.convert('L'))
        except Exception as e:
            print(f"Warning: Failed to convert IR frame to grayscale: {e}")
            return None
    else:
        print(f"Unsupported IR format: {ir_format}")
        return None


def frame_to_rgb_image(frame: Any) -> Optional[np.ndarray]:
    """Convert Orbbec color frame to RGB numpy array.

    Args:
        frame: Orbbec VideoFrame

    Returns:
        RGB image as numpy array, or None if conversion fails
    """
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        return data.reshape((height, width, 3)).copy()
    elif color_format == OBFormat.BGR:
        image = data.reshape((height, width, 3))
        # Swap BGR to RGB
        return image[:, :, ::-1].copy()
    elif color_format == OBFormat.MJPG:
        # Decode JPEG using PIL
        try:
            pil_image = Image.open(BytesIO(data.tobytes()))
            return np.array(pil_image.convert('RGB'))
        except Exception as e:
            print(f"Warning: Failed to convert color frame to RGB: {e}")
            return None
    elif color_format in (OBFormat.YUYV, OBFormat.UYVY):
        # YUV formats - convert using numpy operations
        yuv = data.reshape((height, width, 2))
        if color_format == OBFormat.YUYV:
            y = yuv[:, :, 0].astype(np.float32)
            u = yuv[:, 0::2, 1].repeat(2, axis=1).astype(np.float32) - 128
            v = yuv[:, 1::2, 1].repeat(2, axis=1).astype(np.float32) - 128
        else:  # UYVY
            y = yuv[:, :, 1].astype(np.float32)
            u = yuv[:, 0::2, 0].repeat(2, axis=1).astype(np.float32) - 128
            v = yuv[:, 1::2, 0].repeat(2, axis=1).astype(np.float32) - 128

        # YUV to RGB conversion
        r = np.clip(y + 1.402 * v, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.344 * u - 0.714 * v, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.772 * u, 0, 255).astype(np.uint8)

        return np.stack([r, g, b], axis=-1)
    else:
        print(f"Unsupported color format: {color_format}")
        return None


def extrinsic_to_pose(extrinsic: Any) -> vslam.Pose:
    """Convert Orbbec extrinsic transform to cuVSLAM pose.

    Args:
        extrinsic: Orbbec extrinsic transformation (rotation matrix + translation)

    Returns:
        vslam.Pose object
    """
    # Orbbec extrinsic contains:
    # - rot: 3x3 rotation matrix as numpy array
    # - transform: translation vector in mm
    rotation_matrix = np.array(extrinsic.rot).reshape(3, 3)
    translation_vec = np.array(extrinsic.transform) / 1000.0  # Convert mm to meters

    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vec)


def get_orbbec_camera(
    intrinsic: Any,
    extrinsic: Optional[Any] = None
) -> vslam.Camera:
    """Create a Camera object from Orbbec camera parameters.

    Orbbec IR images are rectified, so we use pinhole model without distortion.

    Args:
        intrinsic: Orbbec camera intrinsics (fx, fy, cx, cy, width, height)
        extrinsic: Optional extrinsic transformation for camera pose

    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = (intrinsic.fx, intrinsic.fy)
    cam.principal = (intrinsic.cx, intrinsic.cy)
    cam.size = (intrinsic.width, intrinsic.height)

    if extrinsic is not None:
        cam.rig_from_camera = extrinsic_to_pose(extrinsic)

    return cam


def get_stereo_calibration(pipeline: Pipeline) -> Dict[str, Any]:
    """Get stereo calibration data from Orbbec pipeline.

    Args:
        pipeline: Orbbec pipeline with stereo IR streams enabled

    Returns:
        Dictionary containing stereo calibration parameters
    """
    # Wait for frames to get stream profiles
    frames = None
    for _ in range(30):  # Wait up to ~3 seconds
        frames = pipeline.wait_for_frames(100)
        if frames is not None:
            left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
            right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)
            if left_frame is not None and right_frame is not None:
                break

    if frames is None:
        raise RuntimeError("Failed to get frames for calibration")

    left_frame = frames.get_frame(OBFrameType.LEFT_IR_FRAME)
    right_frame = frames.get_frame(OBFrameType.RIGHT_IR_FRAME)

    if left_frame is None or right_frame is None:
        raise RuntimeError("Failed to get stereo IR frames")

    # Get stream profiles
    left_profile = left_frame.as_video_frame().get_stream_profile().as_video_stream_profile()
    right_profile = right_frame.as_video_frame().get_stream_profile().as_video_stream_profile()

    stereo_params = {
        'left_intrinsic': left_profile.get_intrinsic(),
        'right_intrinsic': right_profile.get_intrinsic(),
        'right_extrinsic': right_profile.get_extrinsic_to(left_profile)
    }

    return stereo_params


def get_orbbec_stereo_rig(stereo_params: Dict[str, Any]) -> vslam.Rig:
    """Create a stereo Rig object from Orbbec stereo parameters.

    Args:
        stereo_params: Dictionary containing stereo calibration parameters

    Returns:
        vslam.Rig object
    """
    rig = vslam.Rig()

    # Create left camera (reference camera, no extrinsic)
    left_camera = get_orbbec_camera(stereo_params['left_intrinsic'])

    # Create right camera with extrinsics
    right_camera = get_orbbec_camera(
        stereo_params['right_intrinsic'],
        stereo_params['right_extrinsic']
    )

    # Print camera info
    left_intr = stereo_params['left_intrinsic']
    right_ext = stereo_params['right_extrinsic']
    baseline = np.linalg.norm(np.array(right_ext.transform) / 1000.0)

    print(f"Camera resolution: {left_camera.size}")
    print(f"Left camera - fx: {left_intr.fx:.2f}, fy: {left_intr.fy:.2f}, "
          f"cx: {left_intr.cx:.2f}, cy: {left_intr.cy:.2f}")
    print(f"Stereo baseline: {baseline*1000:.2f} mm")

    rig.cameras = [left_camera, right_camera]
    return rig


def get_rgbd_calibration(pipeline: Pipeline) -> Dict[str, Any]:
    """Get RGBD calibration data from Orbbec pipeline.

    Args:
        pipeline: Orbbec pipeline with color and depth streams enabled

    Returns:
        Dictionary containing RGBD calibration parameters
    """
    # Wait for frames to get stream profiles
    frames = None
    for _ in range(30):  # Wait up to ~3 seconds
        frames = pipeline.wait_for_frames(100)
        if frames is not None:
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is not None and depth_frame is not None:
                break

    if frames is None:
        raise RuntimeError("Failed to get frames for calibration")

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if color_frame is None or depth_frame is None:
        raise RuntimeError("Failed to get color/depth frames")

    # Get stream profiles
    color_profile = color_frame.as_video_frame().get_stream_profile().as_video_stream_profile()

    rgbd_params = {
        'color_intrinsic': color_profile.get_intrinsic(),
    }

    return rgbd_params


def get_orbbec_rgbd_rig(rgbd_params: Dict[str, Any]) -> vslam.Rig:
    """Create an RGBD Rig object from Orbbec RGBD parameters.

    Args:
        rgbd_params: Dictionary containing RGBD calibration parameters

    Returns:
        vslam.Rig object
    """
    rig = vslam.Rig()

    # Create color camera (single camera for RGBD)
    color_camera = get_orbbec_camera(rgbd_params['color_intrinsic'])

    # Print camera info
    color_intr = rgbd_params['color_intrinsic']

    print(f"Camera resolution: {color_camera.size}")
    print(f"Color camera - fx: {color_intr.fx:.2f}, fy: {color_intr.fy:.2f}, "
          f"cx: {color_intr.cx:.2f}, cy: {color_intr.cy:.2f}")

    rig.cameras = [color_camera]
    return rig
