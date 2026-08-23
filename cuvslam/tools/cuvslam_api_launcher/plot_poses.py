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

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_track_data(filepath):
    data = np.loadtxt(filepath, usecols=(1, 2, 3))  # Assuming x, y, z are the 2nd, 3rd, and 4th columns
    return data

all_tracks = []
track_labels = []

filepaths = sys.argv[1:]

if not filepaths:
    print("Please provide a list of files as arguments. Files should be in format `timestamp x y z [other cols]`")
    sys.exit(1)

for filepath in filepaths:
    try:
        track = read_track_data(filepath)
        all_tracks.append(track)
        track_labels.append(os.path.basename(filepath))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")

if not all_tracks:
    print("No valid tracks found.")
    sys.exit(1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(221, projection='3d')
ax_xy = fig.add_subplot(222)
ax_yz = fig.add_subplot(223)
ax_zx = fig.add_subplot(224)

for track, label in zip(all_tracks, track_labels):
    ax.plot(track[:, 0], track[:, 1], track[:, 2], label=label)
    ax_xy.plot(track[:, 0], track[:, 1], label=label)
    ax_yz.plot(track[:, 1], track[:, 2], label=label)
    ax_zx.plot(track[:, 2], track[:, 0], label=label)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Tracks')
ax.set_aspect('auto', adjustable='datalim')

ax_xy.set_xlabel('X')
ax_xy.set_ylabel('Y')
ax_xy.set_title('XY Projection')
ax_xy.set_aspect('auto', adjustable='datalim')

ax_yz.set_xlabel('Y')
ax_yz.set_ylabel('Z')
ax_yz.set_title('YZ Projection')
ax_yz.set_aspect('auto', adjustable='datalim')

ax_zx.set_xlabel('Z')
ax_zx.set_ylabel('X')
ax_zx.set_title('ZX Projection')
ax_zx.set_aspect('auto', adjustable='datalim')

fig.legend()

plt.show()
