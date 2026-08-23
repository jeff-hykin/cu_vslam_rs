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

import datetime
import json
import os
import sys
import shutil

BACKUP_FOLDER = '.backup'
DRY_RUN = False


def upgrade_from_08_to_09(f):
    f[0]['version'] = '0.9'
    cam_array = f[0]['cameras']
    for i in range(len(cam_array)):
        intrinsics = cam_array[i]['intrinsics']
        distortion_model = None
        distortion = None

        if 'distortion_model' in intrinsics:
            distortion_model = intrinsics['distortion_model']
        if 'distortion' in intrinsics:
            distortion = intrinsics['distortion']

        if distortion_model == 'pinhole':
            if distortion not in ([0, 0], None):
                raise Exception('Unexpected edex file')
            f[0]['cameras'][i]['intrinsics']['distortion_params'] = []
        elif distortion_model is None:
            if distortion == [0, 0] or distortion is None:
                f[0]['cameras'][i]['intrinsics']['distortion_model'] = 'pinhole'
                f[0]['cameras'][i]['intrinsics']['distortion_params'] = []
            elif len(distortion) == 4:
                f[0]['cameras'][i]['intrinsics']['distortion_model'] = 'fisheye'
                f[0]['cameras'][i]['intrinsics']['distortion_params'] = distortion
            else:
                raise Exception('Unexpected edex file')
        else:
            raise Exception('Unexpected edex file')

        f[0]['cameras'][i]['intrinsics'].pop('distortion', None)  # remove old field
    return f


def make_backup(path, name):
    backup_folder = os.path.join(path, BACKUP_FOLDER)
    source = os.path.join(path, name)
    time_prefix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    backup_name = os.path.join(backup_folder, time_prefix + '_' + name)
    os.makedirs(backup_folder, exist_ok=True)
    shutil.copyfile(source, backup_name)


def validate_edex(f):
    body = f[1]
    fps_present = 'fps' in body
    if 'frame_metadata' in body:
        if fps_present:
            print('Wrong edex file. Fps and metadata are in conflict')
    else:
        if not fps_present:
            print('Wrong edex file. Missing fps field.')


def upgrade_edex(path, name):
    full_path = os.path.join(path, name)
    print('Analysing edex:{}'.format(full_path))
    f = json.load(open(full_path, 'r'))
    validate_edex(f)
    version = f[0]['version']
    if version != '0.9':
        print(' * upgrading from {} version'.format(version))
        if version == '0.8':
            f09 = upgrade_from_08_to_09(f)
            if not DRY_RUN:
                make_backup(path, name)
                with open(full_path, 'w') as out:
                    json.dump(f09, out, indent=4, sort_keys=True)
        else:
            raise Exception('Unsupported version')


def upgrade_all_edex_in_subfolder(folder):
    for root, dirs, files in os.walk(folder):
        if os.path.basename(root) == BACKUP_FOLDER:
            continue
        for name in files:
            _, ext = os.path.splitext(name)
            if ext == '.edex':
                upgrade_edex(os.path.join(folder, root), name)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong command line. Please use:\n{} root_edex_folder'.format(sys.argv[0]))
        sys.exit()
    upgrade_all_edex_in_subfolder(sys.argv[1])
