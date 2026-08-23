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

'''
Convert frame metadata to multicam format

Usage:
    python3 src/tools/edex/convert_frame_metadata_to_multicam_format/convert_frame_metadata_to_multicam_format.py /path/to/cuvslam/datasets/example_dataset/
'''

import os
import json
import shutil
import argparse

def convert_json(json_data):
    """
    input_json_format =
    {
        "frame_id": 0,
        "left_img_filename": "images/cam0.00000.png",
        "left_timestamp": 1679070197387594240,
        "right_img_filename": "images/cam1.00000.png",
        "right_timestamp": 1679070197387594240,
        "type": "frame_metadata"
    }
    output_json_format =
    {
      "frame_id": 0,
      "cams": [
        {"id": 0, "filename": "images/cam0.00000.png", "timestamp": 1679070197387594240},
        {"id": 1, "filename": "images/cam1.00000.png", "timestamp": 1679070197387594240}
      ]
    }
    """
    data = json.loads(json_data)
    frame_id = data["frame_id"]
    left_img_filename = data["left_img_filename"]
    left_timestamp = data["left_timestamp"]
    right_img_filename = data["right_img_filename"]
    right_timestamp = data["right_timestamp"]

    new_data = {
        "frame_id": frame_id,
        "cams": [
            {"id": 0, "filename": left_img_filename, "timestamp": left_timestamp},
            {"id": 1, "filename": right_img_filename, "timestamp": right_timestamp}
        ]
    }

    new_json = json.dumps(new_data, separators=(",", ":"))

    return new_json

def convertion_for_single_dir(input_jsonl_file="frame_metadata.jsonl", input_jsonl_file_stereo="frame_metadata_stereo.jsonl"):
    """
    Replace frame_metadata.jsonl with a new format and
    backup old format frame_metadata.jsonl by creating a copy named frame_metadata_stereo.jsonl
    """

    if os.path.exists(input_jsonl_file_stereo):
        print(f"{input_jsonl_file_stereo} already exists")
    else:
        if os.path.exists(input_jsonl_file):
            shutil.move(input_jsonl_file, input_jsonl_file_stereo)
        else:
            print(f"[ERROR] {input_jsonl_file} does not exist")
            return

    if os.path.exists(input_jsonl_file_stereo):
        input_jsonl = open(input_jsonl_file_stereo, "r")
    else:
        print(f"[ERROR] {input_jsonl_file_stereo} does not exist")
        return
    if os.path.exists(input_jsonl_file):
        os.remove(input_jsonl_file)
    output_jsonl = open(input_jsonl_file, "a")

    for line in input_jsonl:
        output_json_line = convert_json(line)
        output_jsonl.write(output_json_line + "\n")

    input_jsonl.close()
    output_jsonl.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert frame metadata to multicam format')
    parser.add_argument('path', type=str, help='Path to the folder containing frame_metadata.jsonl files')
    args = parser.parse_args()

    # Convert the current directory first
    convertion_for_single_dir()

    # Convert all subdirectories in the specified path
    if os.path.exists(args.path):
        for dirpath, dirnames, filenames in os.walk(args.path):
            for dirname in dirnames:
                seq_path = os.path.join(dirpath, dirname)
                input_jsonl_file = os.path.join(seq_path, "frame_metadata.jsonl")
                input_jsonl_file_stereo = os.path.join(seq_path, "frame_metadata_stereo.jsonl")
                convertion_for_single_dir(input_jsonl_file, input_jsonl_file_stereo)
    else:
        print(f"[ERROR] Path {args.path} does not exist")
