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

import argparse
import os
from .pipeline import TartanAirPipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert classic TartanAir sequences to EDEX layout.")
    parser.add_argument("--seq_path", required=True)
    parser.add_argument("--save_gt_folder", default="gt")
    parser.add_argument("--save_edex_folder", default="edex")
    args = parser.parse_args(argv)

    seq_path = args.seq_path
    save_gt_folder = args.save_gt_folder
    save_edex_folder = args.save_edex_folder

    if os.path.exists(save_gt_folder):
        print(f"{save_gt_folder} folder already exists")
        return
    if os.path.exists(save_edex_folder):
        print(f"{save_edex_folder} folder already exists")
        return

    seq_folders = []

    tartan_dirs = {"image_left", "image_right"} # set
    tartan_files = {"pose_left.txt", "pose_right.txt"} # set

    for (path, dirs, files) in os.walk(seq_path):
        dirs = set(dirs)
        files = set(files)
        if len(tartan_dirs.intersection(dirs)) == len(tartan_dirs):
            if len(tartan_files.intersection(files)) == len(tartan_files):
                seq_folders.append(path)

    print(seq_folders)

    for seq_folder in seq_folders:
        pipeline = TartanAirPipeline(seq_folder, save_gt_folder, save_edex_folder)
        pipeline()


if __name__ == '__main__':
    main()
