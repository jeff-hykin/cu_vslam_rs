#!/usr/bin/env python3
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

"""Convert EuRoC MAV dataset to edex format for cuvslam_api_launcher.

Usage:
    python euroc_to_edex.py /mnt/wdssd/Euroc/MH_01_easy
    python euroc_to_edex.py /mnt/wdssd/Euroc/MH_01_easy --output /tmp/MH_01_edex
    python euroc_to_edex.py /mnt/wdssd/Euroc  # convert all sequences

The script creates an edex directory with:
    stereo.edex          - camera/IMU calibration JSON
    frame_metadata.jsonl - per-frame timestamps and image paths
    IMU.jsonl            - IMU measurements
    images/              - symlinks to original EuRoC images

Coordinate convention:
    The original EuRoC runner passed EuRoC T_BS transforms and raw IMU data
    directly to the cuvslam2 API.  cuvslam_api_launcher reads
    legacy-cuVSLAM-formatted edex and applies LegacyEdexIsometryToOpenCV /
    LegacyEdexImuExtrinsicToOpenCV before feeding the API.  This script stores
    transforms so that after those conversions the API receives the same values.

    Camera:  edex_cam = K * T_BS_cam * K   (similarity transform)
    IMU:     edex_imu = K^-1 * T_BS_imu    (only destination frame changes)
    IMU data: coordinate_system = "opencv"  (identity → raw body-frame data)

    where K = diag(1, -1, -1, 1).
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml


# K converts between OpenCV and legacy cuVSLAM frames: diag(1, -1, -1, 1)
_K = np.diag([1.0, -1.0, -1.0, 1.0])


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_transform(config):
    """Extract 4x4 transform matrix from EuRoC sensor config."""
    for key in config:
        if key.startswith('T_'):
            return np.array(config[key]['data']).reshape(4, 4)
    raise ValueError("No T_* transform found in config")


def cam_to_legacy_edex(T_BS):
    """Convert T_BS to legacy cuVSLAM edex format for cameras.

    LegacyEdexIsometryToOpenCV(result) == T_BS
    LegacyEdexIsometryToOpenCV(X) = K^-1 * X * K = K * X * K  (K is self-inverse)
    So: result = K * T_BS * K
    """
    return _K @ T_BS @ _K


def imu_to_legacy_edex(T_BS):
    """Convert T_BS to legacy cuVSLAM edex format for IMU extrinsic.

    LegacyEdexImuExtrinsicToOpenCV(result) == T_BS
    LegacyEdexImuExtrinsicToOpenCV(X) = K * X
    So: result = K^-1 * T_BS = K * T_BS  (K is self-inverse)
    """
    return _K @ T_BS


def read_cam_csv(csv_path):
    """Read EuRoC camera data.csv, return list of (timestamp_ns, filename)."""
    entries = []
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            entries.append((int(row[0]), row[1].strip()))
    return entries


def read_imu_csv(csv_path):
    """Read EuRoC imu0/data.csv, return list of dicts."""
    entries = []
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue
            entries.append({
                'timestamp': int(row[0]),
                'wx': float(row[1]), 'wy': float(row[2]), 'wz': float(row[3]),
                'ax': float(row[4]), 'ay': float(row[5]), 'az': float(row[6]),
            })
    return entries


def convert_one(euroc_path, output_path):
    """Convert a single EuRoC sequence to edex format."""
    mav0 = os.path.join(euroc_path, 'mav0')
    if not os.path.isdir(mav0):
        print(f"  Skipping {euroc_path}: no mav0/ directory")
        return False

    # Load calibration
    cam0_cfg = load_yaml(os.path.join(mav0, 'cam0', 'sensor.yaml'))
    cam1_cfg = load_yaml(os.path.join(mav0, 'cam1', 'sensor.yaml'))
    imu_cfg = load_yaml(os.path.join(mav0, 'imu0', 'sensor.yaml'))

    # EuRoC T_BS: body-from-sensor transforms.
    # The original EuRoC runner used body as rig frame and passed T_BS directly
    # to the API.
    # We store them in legacy cuVSLAM format so that the launcher's
    # LegacyEdexIsometryToOpenCV / LegacyEdexImuExtrinsicToOpenCV recovers T_BS.
    T_body_cam0 = get_transform(cam0_cfg)
    T_body_cam1 = get_transform(cam1_cfg)
    T_body_imu = get_transform(imu_cfg)

    edex_cam0 = cam_to_legacy_edex(T_body_cam0)
    edex_cam1 = cam_to_legacy_edex(T_body_cam1)
    edex_imu = imu_to_legacy_edex(T_body_imu)

    # Build camera entries
    def make_camera(cfg, edex_transform):
        # EuRoC radial-tangential [k1, k2, p1, p2] → brown5k [k1, k2, p1, p2, k3=0]
        dist = cfg['distortion_coefficients']
        return {
            'transform': edex_transform[:3, :].tolist(),
            'intrinsics': {
                'size': cfg['resolution'],
                'focal': cfg['intrinsics'][:2],
                'principal': cfg['intrinsics'][2:4],
                'distortion_model': 'brown5k',
                'distortion_params': dist[:2] + dist[2:4] + [0.0],
            }
        }

    cam0_entry = make_camera(cam0_cfg, edex_cam0)
    cam1_entry = make_camera(cam1_cfg, edex_cam1)

    # Read camera timestamps
    cam0_data = read_cam_csv(os.path.join(mav0, 'cam0', 'data.csv'))
    cam1_data = read_cam_csv(os.path.join(mav0, 'cam1', 'data.csv'))

    # Find common timestamps (EuRoC stereo pairs share timestamps)
    cam1_ts_set = {ts for ts, _ in cam1_data}
    cam1_by_ts = {ts: fn for ts, fn in cam1_data}
    paired = [(ts, fn0, cam1_by_ts[ts]) for ts, fn0 in cam0_data if ts in cam1_ts_set]
    paired.sort(key=lambda x: x[0])

    if not paired:
        print(f"  Skipping {euroc_path}: no matching stereo pairs")
        return False

    num_frames = len(paired)
    print(f"  {num_frames} stereo pairs, {len(read_imu_csv(os.path.join(mav0, 'imu0', 'data.csv')))} IMU samples")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    img_dir = os.path.join(output_path, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Create symlinks for images and write frame_metadata.jsonl
    with open(os.path.join(output_path, 'frame_metadata.jsonl'), 'w') as fm:
        for i, (ts, fn0, fn1) in enumerate(paired):
            cam0_src = os.path.join(mav0, 'cam0', 'data', fn0)
            cam1_src = os.path.join(mav0, 'cam1', 'data', fn1)
            cam0_dst = f'images/cam0.{i:05d}.png'
            cam1_dst = f'images/cam1.{i:05d}.png'

            # Create symlinks (overwrite if exist)
            for src, dst_rel in [(cam0_src, cam0_dst), (cam1_src, cam1_dst)]:
                dst_abs = os.path.join(output_path, dst_rel)
                if os.path.islink(dst_abs) or os.path.exists(dst_abs):
                    os.remove(dst_abs)
                os.symlink(os.path.abspath(src), dst_abs)

            entry = {
                'frame_id': i,
                'cams': [
                    {'id': 0, 'filename': cam0_dst, 'timestamp': ts},
                    {'id': 1, 'filename': cam1_dst, 'timestamp': ts},
                ]
            }
            json.dump(entry, fm)
            fm.write('\n')

    # Write IMU.jsonl — raw body-frame data, no coordinate conversion
    imu_data = read_imu_csv(os.path.join(mav0, 'imu0', 'data.csv'))
    with open(os.path.join(output_path, 'IMU.jsonl'), 'w') as f:
        for m in imu_data:
            line = {
                'AngularVelocityX': m['wx'],
                'AngularVelocityY': m['wy'],
                'AngularVelocityZ': m['wz'],
                'LinearAccelerationX': m['ax'],
                'LinearAccelerationY': m['ay'],
                'LinearAccelerationZ': m['az'],
                'timestamp': m['timestamp'],
            }
            json.dump(line, f)
            f.write('\n')

    # Write stereo.edex
    # coordinate_system "opencv" → identity change_basis → IMU data passes through raw,
    # matching the original EuRoC runner's direct RegisterImuMeasurement input.
    edex = [
        {
            'version': '0.9',
            'frame_start': 0,
            'frame_end': num_frames - 1,
            'cameras': [cam0_entry, cam1_entry],
            'imu': {
                'transform': edex_imu[:3, :].tolist(),
                'measurements': 'IMU.jsonl',
                'g': [0.0, -9.81, 0.0],
                'coordinate_system': 'opencv',
                'frequency': imu_cfg['rate_hz'],
                'gyro_noise_density': imu_cfg['gyroscope_noise_density'],
                'gyro_random_walk': imu_cfg['gyroscope_random_walk'],
                'accel_noise_density': imu_cfg['accelerometer_noise_density'],
                'accel_random_walk': imu_cfg['accelerometer_random_walk'],
            }
        },
        {
            'frame_metadata': 'frame_metadata.jsonl',
            'sequence': [
                ['images/cam0.00000.png'],
                ['images/cam1.00000.png'],
            ],
            'points2d': {},
            'points3d': {},
            'rig_positions': {},
        }
    ]

    with open(os.path.join(output_path, 'stereo.edex'), 'w') as f:
        json.dump(edex, f, indent=4)

    print(f"  -> {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert EuRoC dataset(s) to edex format')
    parser.add_argument('input', help='Path to a single EuRoC sequence or parent directory')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path. For single sequence, the edex directory. '
                             'For batch, parent directory (default: <input>_edex or <input>)')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)

    # Check if this is a single sequence or a parent directory with multiple sequences
    if os.path.isdir(os.path.join(input_path, 'mav0')):
        # Single sequence
        output = args.output or (input_path + '_edex')
        print(f"Converting {os.path.basename(input_path)}...")
        if not convert_one(input_path, output):
            sys.exit(1)
    else:
        # Batch: find all subdirs with mav0/
        sequences = sorted([
            d for d in os.listdir(input_path)
            if os.path.isdir(os.path.join(input_path, d, 'mav0'))
        ])
        if not sequences:
            print(f"No EuRoC sequences found in {input_path}")
            sys.exit(1)

        output_base = args.output or input_path
        print(f"Found {len(sequences)} sequences: {', '.join(sequences)}")
        for seq in sequences:
            seq_input = os.path.join(input_path, seq)
            seq_output = os.path.join(output_base, seq, 'edex')
            print(f"Converting {seq}...")
            convert_one(seq_input, seq_output)

    print("Done.")


if __name__ == '__main__':
    main()
