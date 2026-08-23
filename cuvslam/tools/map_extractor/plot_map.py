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
import json
import numpy as np
from scipy.spatial.transform import Rotation

import click

CUVSLAM_FROM_ROS = np.array([
    [0, -1, 0],
    [0,  0, 1],
    [-1, 0, 0]
])

def load_poses(data):
    poses = {}
    for pose in data["poses"]:
        kf_id = pose["keyframe_id"]

        quat = pose["quaternion"]
        R = Rotation.from_quat([
            quat["x"], quat["y"], quat["z"], quat["w"]
        ]).as_matrix()

        R = CUVSLAM_FROM_ROS.T @ R @ CUVSLAM_FROM_ROS

        trans = pose["translation"]
        t = np.array([trans["x"], trans["y"], trans["z"]])
        t = CUVSLAM_FROM_ROS.T @ t

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        poses.update({kf_id: T})
    return poses

def load_edges(data):
    edges = []
    for edge in data["edges"]:
        start_kf_id = edge["start"]
        end_kf_id = edge["end"]
        edges.append((start_kf_id, end_kf_id))
    return edges

@click.command()
@click.argument('map_path')
def main(map_path):
    with open(map_path) as f:
        data = json.loads(f.read())

    poses = load_poses(data)
    edges = load_edges(data)

    points = np.stack([v[:3, 3] for k, v in poses.items()])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for s, e in edges:
        start = poses[s][:3, 3]
        end = poses[e][:3, 3]

        t = np.stack([start, end])
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color = "blue")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color = "red")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

if __name__ == "__main__":
    main()
