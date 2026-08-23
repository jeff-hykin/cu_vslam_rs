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

import shutil
import sys
from os.path import basename

from edex_reader import EdexReader
from bag_writer import BagWriter


def convert_edex_to_bag(edex_filename, bag_filename):
    reader = EdexReader(edex_filename)
    writer = BagWriter(bag_filename, reader.intrinsics, reader.transforms, reader.imu)
    i = 0
    if reader.imu:
        for x in reader.imu_measurements:
            writer.write_imu_measurement(x)
    while not reader.end():
        print('{} '.format(i), end='')
        writer.write_frame(*reader.read_frame())
        i = i + 1
    writer.close()
    print('\n ros2 bag play {}'.format(bag_filename))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Use\n {} path/to/file.edex path/to/out_bag_folder".format(basename(sys.argv[0])))
        sys.exit('Wrong parameter number')
    in_edex = sys.argv[1]
    out_bag = sys.argv[2]
    print('in_edex={}'.format(in_edex))
    print('out_bag={}'.format(out_bag))
    shutil.rmtree(out_bag, ignore_errors=True)
    convert_edex_to_bag(in_edex, out_bag)
    sys.exit(0)
