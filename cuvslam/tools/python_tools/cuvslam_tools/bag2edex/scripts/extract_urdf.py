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

"""Extract a URDF from the /tf_static topic in a rosbag.

The generated URDF is minimal and only contains transforms. Physical parameters like mass, inertia
etc. are not contained.
"""

import argparse
import logging
import pathlib

from cuvslam_tools.bag2edex import rosbag_urdf_extraction


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r",
        "--rosbag_path",
        type=pathlib.Path,
        required=True,
        help="Path to the input rosbag directory.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=pathlib.Path,
        required=True,
        help="Path where the URDF is generated.",
    )
    parser.add_argument(
        "-d",
        "--ros_distribution",
        type=str,
        required=False,
        default="humble",
        help="ROS distribution",
    )
    args = parser.parse_args()

    rosbag_urdf_extraction.extract_urdf(
        args.rosbag_path,
        args.ros_distribution,
        args.output_path,
    )


if __name__ == "__main__":
    main()
