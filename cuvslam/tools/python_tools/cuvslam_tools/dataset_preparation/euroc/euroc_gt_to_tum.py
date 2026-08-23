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

"""
Convert EuRoC ground-truth pose CSV to TUM format for comparison with odometry output.

EuRoC state_groundtruth_estimate0/data.csv format:
  #timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,v_RS_R_x,...
  (timestamp in nanoseconds; position in meters; quaternion w,x,y,z)

TUM format (same as odom_pose.txt from cuvslam_api_launcher -print_format=tum):
  timestamp_s tx ty tz qx qy qz qw
  (timestamp in seconds for direct comparison with launcher output)

Usage:
  python euroc_gt_to_tum.py [gt_csv] [output_tum]
  Default: gt_csv = /mnt/wdssd/Euroc/vicon_room1/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv
           output_tum = gt_pose_tum.txt (in same dir as script, or use absolute path)
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert EuRoC GT CSV to TUM format")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to EuRoC sequence root (the directory containing mav0/). "
             "Derives gt_csv and output_tum automatically.",
    )
    parser.add_argument(
        "gt_csv",
        nargs="?",
        default=None,
        help="EuRoC state_groundtruth_estimate0 data.csv path (overrides --dataset)",
    )
    parser.add_argument(
        "output_tum",
        nargs="?",
        default=None,
        help="Output TUM file (overrides --dataset; default: gt_pose_tum.txt next to gt_csv)",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=("seconds", "nanoseconds"),
        default="seconds",
        help="TUM timestamp unit; use 'seconds' to match odom_pose.txt from launcher (default: seconds)",
    )
    args = parser.parse_args()

    if args.gt_csv:
        gt_path = os.path.expanduser(args.gt_csv)
        if args.output_tum:
            out_path = os.path.expanduser(args.output_tum)
        else:
            out_path = os.path.join(os.path.dirname(gt_path), "gt_pose_tum.txt")
    elif args.dataset:
        dataset = os.path.expanduser(args.dataset)
        gt_path = os.path.join(dataset, "mav0", "state_groundtruth_estimate0", "data.csv")
        out_path = args.output_tum or os.path.join(dataset, "gt_pose_tum.txt")
    else:
        parser.error("Provide --dataset /path/to/sequence or a positional gt_csv argument.")

    if not os.path.isfile(gt_path):
        print(f"Error: GT file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    # EuRoC CSV columns:
    # 0=timestamp(ns), 1-3=position, 4-7=quaternion(w,x,y,z),
    # 8-10=velocity, 11-13=gyro_bias(b_w), 14-16=accel_bias(b_a)
    to_seconds = args.timestamp_unit == "seconds"

    # Gyro bias GT output file (next to pose output)
    bias_out_path = os.path.join(os.path.dirname(out_path), "gt_gyro_bias.txt")

    lines_written = 0
    with open(gt_path) as f_in, open(out_path, "w") as f_out, open(bias_out_path, "w") as f_bias:
        f_bias.write("# timestamp_s b_w_x[rad/s] b_w_y[rad/s] b_w_z[rad/s]\n")
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 17:
                continue
            try:
                ts_ns = int(parts[0])
                px = float(parts[1])
                py = float(parts[2])
                pz = float(parts[3])
                qw = float(parts[4])
                qx = float(parts[5])
                qy = float(parts[6])
                qz = float(parts[7])
                bw_x = float(parts[11])
                bw_y = float(parts[12])
                bw_z = float(parts[13])
            except (ValueError, IndexError):
                continue
            if to_seconds:
                ts_out = ts_ns / 1e9
            else:
                ts_out = ts_ns
            # TUM: timestamp tx ty tz qx qy qz qw
            f_out.write(f"{ts_out:.9f} {px:.9f} {py:.9f} {pz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            # Gyro bias GT: timestamp bw_x bw_y bw_z
            f_bias.write(f"{ts_out:.9f} {bw_x:.9f} {bw_y:.9f} {bw_z:.9f}\n")
            lines_written += 1

    print(f"Wrote {lines_written} poses to {out_path} (TUM format, timestamp in {'seconds' if to_seconds else 'nanoseconds'})")
    print(f"Wrote {lines_written} gyro bias GT to {bias_out_path}")
    print(f"Compare: evo_ape tum {out_path} <path/to/poses.txt> -va --align")


if __name__ == "__main__":
    main()
