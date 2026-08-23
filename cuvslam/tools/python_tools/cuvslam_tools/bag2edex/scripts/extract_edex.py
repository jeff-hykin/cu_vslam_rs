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

"""Extract an EDEX dataset from a rosbag.

The generated EDEX dataset contains images, frame metadata, EDEX metadata, and URDF.
"""

import argparse
import logging
import os
import pathlib

import yaml

from cuvslam_tools.bag2edex import (
    Config,
    extract_edex,
)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=pathlib.Path,
        required=True,
        help="Path to the config file.",
    )
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
        help="Path where the EDEX dataset is generated.",
    )

    args = parser.parse_args()

    assert (
        args.config_path.exists()
    ), f"Config path '{args.config_path}' does not exist."
    yaml_string = args.config_path.read_text()
    yaml_dict = yaml.safe_load(yaml_string)

    # We override some arguments from the CLI.
    if "rosbag_path" in yaml_dict:
        logging.warning("rosbag_path in the config is overridden by --rosbag_path.")
    if "output_path" in yaml_dict:
        logging.warning("output_path in the config is overridden by --output_path.")
    yaml_dict["rosbag_path"] = args.rosbag_path.absolute()
    yaml_dict["output_path"] = args.output_path.absolute()
    config = Config(**yaml_dict)

    extract_edex(config)


if __name__ == "__main__":
    main()
