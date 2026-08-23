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

# draw landmarks from output edex
import json
from sys import argv
import matplotlib.pyplot as plt


def edex_landmarks_viewer(edex):
    """@edex - edex filename"""
    print("Open %s" % edex)
    edex_data = json.load(open(edex))
    body = edex_data[1]
    landmarks = body['points3d']
    xs = []
    zs = []
    for landmarks_id in landmarks:
        xyz = landmarks[landmarks_id]
        x = xyz[0]
        y = xyz[1]
        z = -xyz[2]
        if 0.5 < y < 0.8 and 2.5 < z < 4.5:
            xs.append(x)
            zs.append(z)
    plt.scatter(xs, zs, s=2, label=edex[70:])


if __name__ == '__main__':
    if len(argv) == 1:
        print('Use:\n edex_landmarks_viewer <edex1_file> <edex2_file> ..')
        exit(0)

    for edex in argv[1:]:
        edex_landmarks_viewer(edex)
    plt.legend()
    plt.show()
