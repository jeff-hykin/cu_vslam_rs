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

# Split long edex sequence to series of shorter ones.
# Useful for parallel reporter execution on a long sequence.
import os
import json
from sys import argv


def edex_split(edex):
    """@edex - edex filename"""
    print("Splitting %s" % edex)
    edex_data = json.load(open(edex))
    header = edex_data[0]
    body = edex_data[1]
    if 'imu' in header:
        header['imu']['measurements'] = os.path.join('./../', header['imu']['measurements'])
    meta_name = body['frame_metadata']
    root = os.path.dirname(edex)
    meta_full_path = os.path.join(root, meta_name)
    gt_full_path = os.path.join(root, 'gt.txt')
    meta = open(meta_full_path)
    gt = open(gt_full_path)
    frames_to_new_folder = 0
    folder_id = 0
    sub_meta = None
    sub_gt = None
    while True:
        meta_str = meta.readline()
        gt_str = gt.readline()
        if not gt_str and not gt_str:
            print('All done with success.')
            break  # end of files with success
        if not gt_str or not gt_str:
            print('Abnormal termination. GT and Meta have different lengths.')
            break

        if frames_to_new_folder == 0:
            sub_name = 'sub%02d' % folder_id
            print('Create %s' % sub_name)
            sub_folder = os.path.join(root, sub_name)
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            if sub_meta:
                sub_meta.close()
            if sub_gt:
                sub_gt.close()
            sub_meta = open(os.path.join(sub_folder, 'frame_metadata.jsonl'), 'w')
            sub_gt = open(os.path.join(sub_folder, 'gt.txt'), 'w')
            with open(os.path.join(sub_folder, 'stereo.edex'), "w") as sub_edex:
                json_str = json.dumps(edex_data, indent=4)
                sub_edex.write(json_str)
            frames_to_new_folder = 1000
            folder_id += 1

        meta_data = json.loads(meta_str)
        meta_data['left_img_filename'] = os.path.join('./../', meta_data['left_img_filename'])
        meta_data['right_img_filename'] = os.path.join('./../', meta_data['right_img_filename'])
        meta_str = json.dumps(meta_data)
        sub_meta.write(meta_str + '\n')
        sub_gt.write(gt_str)
        frames_to_new_folder -= 1


if __name__ == '__main__':
    if len(argv) != 2:
        print('Use:\n edex_split <edex_file>')
        exit(0)

    edex_split(argv[1])
