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

import os
import sys
import subprocess
from pathlib import Path

undistort_cmd = os.environ.get('CUVSLAM_UNDISTORT', '~/cuvslam/build/release/bin/undistort')


def batch_undistort(in_folder, in_images_mask, in_edex, in_camera_id, out_folder):
    out_folder_p = Path(out_folder)
    out_folder_p.mkdir(parents=True, exist_ok=True)
    for in_image in Path(in_folder).expanduser().glob(in_images_mask):
        out_image = out_folder_p.joinpath(in_image.name).with_suffix('.png')
        try:
            args = [Path(undistort_cmd).expanduser(), in_image, in_edex, out_image]
            if in_camera_id != '0':
                args += ['-camera', in_camera_id]
            print(f'{in_image} -> {out_image}')
            result = subprocess.run(args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e.stderr}")
            sys.exit(2)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("usage: batch_undistort in_folder in_images_mask in_edex in_camera_id out_folder")
        print("  in_camera_id - starting from 0")
        print("example: batch_undistort ~/raw cam0_right_*.jpg ~/stereo.edex 1 ~/undistorted")
        exit(1)

    batch_undistort(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    exit(0)
