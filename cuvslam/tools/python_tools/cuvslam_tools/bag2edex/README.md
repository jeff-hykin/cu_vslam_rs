# Rosbag Extraction Tool


## Install

See [tools/README.md](/tools/README.md) on how to install the `cuvslam_tools` package.


## Configuration

The config is a YAML file with the following fields:
```yaml
# Topic names for CameraInfo messages.
camera_info_topics:
  - /front_stereo_camera/left/camera_info
  - /front_stereo_camera/right/camera_info
  - /back_stereo_camera/left/camera_info
  - /back_stereo_camera/right/camera_info

# Topic names for image messages, in the same order as camera_info_topics.
image_topics:
  - /front_stereo_camera/left/image_compressed
  - /front_stereo_camera/right/image_compressed
  - /back_stereo_camera/left/image_compressed
  - /back_stereo_camera/right/image_compressed

# Name of the rig frame. If unsure, run `rosbag_extract_urdf` and select from list of frames.
rig_frame: base_link

# Name of the IMU frame. (Optional)
imu_frame: front_stereo_camera_imu

# Parameters to resize and reformat images. (Optional)
output_width: 960
output_height: 600
output_format: png

# Maximum threshold for timestamp synchronization, in nanoseconds. Default is 1ms.
sync_threshold_ns: 1000000

# Number of workers for image extraction, must be >=2. Default is number of cores on system.
num_workers: 8

# ROS distribution. Default is "humble".
ros_distribution: humble
```


## Running Extraction Tools

Note: you may see warning messages about `InvalidDataError` during image extraction. This is expected and does not interfere with the execution of the tool.

Extract an EDEX dataset from a rosbag:
```sh
rosbag_extract_edex \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract images from a rosbag (without camera extrinsics):
```sh
rosbag_extract_images \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract videos from a rosbag (currently only supports H.264 encoded rosbags):
```sh
rosbag_extract_videos \
    --config configs/<config>.yaml \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory
```

Extract a URDF file from a rosbag:
```sh
rosbag_extract_urdf \
    --rosbag_path path/to/rosbag/directory \
    --output_path path/to/edex/directory \
    --ros_distribution humble
```


## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
