# cuvslam-tools

`cuvslam-tools` is the installable Python tools package for cuVSLAM workflows.

It provides command-line tools for:

- Preparing public datasets, including KITTI, EuRoC, TartanGround, and TUM RGB-D.
- Converting ROS 2 bags to EDEX inputs.
- Running one tracking sequence.
- Running dataset reports.
- Running multi-dataset validation.
- Undistorting EDEX images.

## Install

From the repository root:

```bash
cd tools/python_tools
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip
pip install -e .
```

With PDF report support:

```bash
pip install -e ".[pdf]"
```

`cuvslam_tracker`, `cuvslam_reporter`, and `cuvslam_validator` require the `cuvslam` Python binding in the same environment. Dataset preparation, ROS bag conversion, and undistortion should stay usable without importing `cuvslam` when their workflows do not need it.

## Install The cuVSLAM Python Binding

If you have a released wheel that matches your platform, Python version, and CUDA version, install it into the same environment:

```bash
pip install /path/to/cuvslam-*.whl
```

When installing from this source tree, build cuVSLAM first, then install the binding from `python`. From `tools/python_tools`:

```bash
cd ../..
mkdir -p ../build
./build_release.sh

cd tools/python_tools
CUVSLAM_BUILD_DIR=/path/to/build/folder pip install ../../python/
```

`CUVSLAM_BUILD_DIR` must be an absolute path to the cuVSLAM build directory and must contain:

```bash
bin/libcuvslam.so
```

After installing the binding, verify it in the same environment:

```bash
python - <<'PY'
import cuvslam
print(cuvslam.get_version())
PY
```

Reinstall the binding after rebuilding `libcuvslam.so`.

## Commands

| Command | Purpose |
|---|---|
| `prepare_kitti` | Download KITTI odometry archives, convert them to cuVSLAM format, and generate KITTI reporter configs. |
| `prepare_euroc` | Download the EuRoC MAV Machine Hall bundle and convert `MH_01_easy` to EDEX plus TUM-format ground truth. |
| `prepare_tartan` | Download TartanGround data and convert TartanGround stereo pairs or compatible TartanAir-layout sequences to EDEX. |
| `prepare_tum` | Download and lay out the TUM RGB-D `freiburg3_long_office_household` dataset. |
| `cuvslam_tracker` | Run one EDEX sequence or supported video input through cuVSLAM. |
| `cuvslam_reporter` | Run one dataset config and generate report outputs. |
| `cuvslam_validator` | Run multiple reporter configs, combine results, and apply validation checks. |
| `rosbag_extract_edex` | Convert a ROS 2 bag to an EDEX sequence directory. |
| `rosbag_extract_images` | Extract images from a ROS 2 bag. |
| `rosbag_extract_urdf` | Inspect/extract TF and URDF data from a ROS 2 bag. |
| `rosbag_extract_videos` | Extract videos from a ROS 2 bag. |
| `undistort_edex_images` | Undistort images from an EDEX sequence. |

Smoke-check installed commands:

```bash
prepare_kitti --help
prepare_euroc --help
prepare_tartan --help
prepare_tum --help
cuvslam_tracker --help
cuvslam_reporter --help
cuvslam_validator --help
rosbag_extract_edex --help
rosbag_extract_images --help
rosbag_extract_urdf --help
rosbag_extract_videos --help
undistort_edex_images --help
```

## Dataset Preparation

`prepare_kitti` is the installed CLI wrapper for the `cuvslam_tools.dataset_preparation.kitti` workflow. It downloads the KITTI odometry archives when needed, converts them to cuVSLAM format, and writes the reporter config files produced by that workflow, including:

- `kitti-vio_gt.cfg`
- `kitti-slam_gt.cfg`
- `kitti-vio_slam.cfg`
- `kitti-vio_slam_gt.cfg`

Example:

```bash
prepare_kitti \
    --raw-dir /path/to/datasets/kitti/raw \
    --output-dir /path/to/datasets/converted/kitti
```

The converted dataset layout is suitable for tracker and reporter workflows. Pass one of the generated KITTI config files to `cuvslam_reporter --test_config`.

`prepare_euroc` downloads the EuRoC MAV Machine Hall bundle, extracts `MH_01_easy`, and writes converted data under `euroc/MH_01_easy`.

```bash
prepare_euroc \
    --raw-dir /path/to/datasets/euroc/raw \
    --output-dir /path/to/datasets/converted
```

`prepare_tartan` downloads a TartanGround variant, stages each available `lcam_*`/`rcam_*` stereo pair into the classic TartanAir layout expected by the converter, and converts the staged sequences to EDEX.

```bash
prepare_tartan \
    --variant multicamera \
    --raw-dir /path/to/datasets/tartan/raw \
    --output-dir /path/to/datasets/converted
```

Use `--variant multicamera` for EDEX conversion from the 12-camera TartanGround image variant. Both TartanGround variants also download metadata, including `pose_lcam_*` and `pose_rcam_*` files. The multicamera variant converts each complete stereo orientation, for example `P2000_front`, `P2000_left`, and `P2000_right`. The `multisensor` variant is intended for the RGB-D/IMU example data and can be downloaded with `--download-only`.

`prepare_tum` downloads and lays out the TUM RGB-D `freiburg3_long_office_household` dataset and copies the bundled rig calibration into the sequence folder.

```bash
prepare_tum \
    --raw-dir /path/to/datasets/tum/raw \
    --output-dir /path/to/datasets/converted
```

All dataset preparation commands support `--force-download` and `--download-only`.

## Tracking

Run one sequence:

```bash
cuvslam_tracker \
    --dataset /path/to/datasets/converted/kitti/00 \
    --config_path /path/to/datasets/converted/kitti/00/stereo.edex \
    --odometry_mode multicamera \
    --output_dir /tmp/cuvslam_tracker-smoke
```

For ROS bag input, first convert the bag to EDEX with `rosbag_extract_edex`.

## Reporting

Run one dataset config:

```bash
cuvslam_reporter \
    --test_config /path/to/datasets/converted/kitti/kitti-vio_gt.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-reports \
    --odometry_mode multicamera \
    --rectified_stereo_camera true \
    --async_sba false \
    --multicam_mode moderate \
    --use_segments
```

The reporter requires `--test_config` to point to one config file. Relative paths are resolved from the current working directory; `--datasets_root` is only used to locate dataset folders referenced by that config.

## Validation

Run a multi-dataset validation config:

```bash
cuvslam_validator \
    --validation_config significant-prompt-run.cfg \
    --datasets_root /path/to/datasets/converted \
    --output_root /tmp/cuvslam-validation \
    --odometry_mode multicamera \
    --use_segments
```

The validator runs reporter configs, writes a combined summary CSV, and fails when configured metric checks fail.

## ROS Bag To EDEX

Create a YAML config file. See `cuvslam_tools/bag2edex/configs/` for examples.

```yaml
camera_info_topics:
  - /camera/infra1/camera_info
  - /camera/infra2/camera_info

image_topics:
  - /camera/infra1/image_rect_raw
  - /camera/infra2/image_rect_raw

rig_frame: camera_link
imu_topic: /camera/imu
ros_distribution: humble
```

Extract a full EDEX dataset from a ROS 2 bag:

```bash
rosbag_extract_edex \
    --config cuvslam_tools/bag2edex/configs/realsense.yaml \
    --rosbag_path path/to/bag_folder \
    --output_path path/to/edex_folder
```

List available TF frames:

```bash
rosbag_extract_urdf \
    --rosbag_path path/to/bag_folder \
    --output_path /tmp/urdf_out \
    --ros_distribution humble
```

Extract images only:

```bash
rosbag_extract_images \
    --config cuvslam_tools/bag2edex/configs/realsense.yaml \
    --rosbag_path path/to/bag_folder \
    --output_path path/to/output_folder
```

ROS bag extraction can produce:

| File | Contents |
|---|---|
| `edex` | Camera intrinsics, extrinsics, IMU transform, and sequence frame list. |
| `images/<topic>/NNNNN.png` | Extracted camera frames. |
| `frame_metadata.jsonl` | Per-frame metadata with filenames and timestamps. |
| `imu.jsonl` | IMU samples, if `imu_topic` is set. |
| `robot.urdf` | URDF extracted from `/tf_static`. |

Available sensor configs:

| Config | Sensor |
|---|---|
| `cuvslam_tools/bag2edex/configs/realsense.yaml` | RealSense |
| `cuvslam_tools/bag2edex/configs/realsense_imu.yaml` | RealSense stereo plus IMU |
| `cuvslam_tools/bag2edex/configs/nova_hawk.yaml` | NVIDIA Nova plus HAWK stereo |
| `cuvslam_tools/bag2edex/configs/oak6.yaml` | OAK-6 camera |

RGBD tracking expects an EDEX file with `depth_sequence` and `depth_id` entries. The ROS bag extractor currently writes camera images and optional IMU data; it does not synthesize RGBD depth sequences from bags yet.

## Undistort EDEX Images

Undistort one image using the camera intrinsics in an EDEX file:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/stereo.edex \
    /path/to/output.png
```

Use a specific camera from a multi-camera EDEX file:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/stereo.edex \
    /path/to/output.png \
    --camera 1
```

Use a separate output EDEX camera model instead of the default pinhole output model:

```bash
undistort_edex_images \
    /path/to/input.png \
    /path/to/input.edex \
    /path/to/output.png \
    /path/to/output.edex \
    --camera 0
```

Batch-undistort loose images from a folder:

```bash
undistort_edex_images \
    /path/to/images \
    /path/to/stereo.edex \
    /path/to/undistorted_images \
    --batch \
    --pattern "*.png"
```

If `--pattern` is omitted in batch mode, the tool auto-detects common image formats: `jpg`, `jpeg`, `png`, `tga`, and `bmp`.
