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
import json
import numpy as np
from typing import List
from copy import deepcopy
from scipy.spatial.transform import Rotation
from .track import State


class TartanAirReader:
    def __init__(self, pose_left_file):
        self.pose_left_file = pose_left_file

    @staticmethod
    def parse_line(line):
        tx, ty, tz, qx, qy, qz, qw = list(map(float, line.split()))
        t = np.array([ty, tz, tx])
        q = np.array([qy, qz, qx, qw])
        r = Rotation.from_quat(q)
        R = r.as_matrix()
        return R, t

    def __call__(self, *args, **kwargs):
        track = []
        with open(self.pose_left_file) as f:
            for line in f:
                R, t = self.parse_line(line)
                state = State.from_rotation_translation(R, t)
                track.append(state)
        return track


class Normalizer:
    def __call__(self, track: List[State]):
        state = deepcopy(track[0])
        state_inv = state.inv_transform

        new_track = []
        for x in track:
            new_track.append(
                State.from_transform(np.matmul(state_inv, x.transform))
            )
        return new_track


class Rotator:
    def __init__(self, R):
        self.rotation = State.from_rotation_translation(R, np.zeros(3))

    def __call__(self, track: List[State]):
        new_track = [
            State.from_transform(np.matmul(self.rotation.transform, x.transform))
            for x in track
        ]
        return new_track


class Writer:
    def __init__(self, seq_folder):
        self.seq_folder = seq_folder

    def __call__(self, track: List[State]):
        with open(os.path.join(self.seq_folder, "gt.txt"), mode="w") as f:
            for state in track:
                lines = []
                T = state.transform[:3]
                for row in T:
                    s = " ".join(map(lambda x: "{:.6e}".format(x), row.tolist()))
                    lines.append(s)
                f.write(" ".join(lines) + "\n")


class TartanAirImageRename:
    def __init__(self, seq_folder):
        self.seq_folder = seq_folder

    def __call__(self, *args, **kwargs):
        left_folder = os.path.join(self.seq_folder, "image_left")
        right_folder = os.path.join(self.seq_folder, "image_right")

        for image_name in os.listdir(left_folder):
            if "left" in image_name:
                old_name = os.path.join(left_folder, image_name)

                new_image_name = "00.0." + image_name.replace("_left", "")
                new_name = os.path.join(left_folder, new_image_name)
                os.rename(old_name, new_name)

        for image_name in os.listdir(right_folder):
            if "right" in image_name:
                old_name = os.path.join(right_folder, image_name)

                new_image_name = "00.0." + image_name.replace("_right", "")
                new_name = os.path.join(right_folder, new_image_name)
                os.rename(old_name, new_name)
        return None


class FolderRename:
    def __init__(self, seq_folder, old_name, new_name):
        self.seq_folder = seq_folder
        self.old_name = old_name
        self.new_name = new_name

    def __call__(self, *args, **kwargs):
        left_folder = os.path.join(self.seq_folder, self.old_name)
        new_left_folder = os.path.join(self.seq_folder, self.new_name)

        if os.path.exists(left_folder):
            os.rename(left_folder, new_left_folder)
        return None


class TartanAirFillEdex:
    def __init__(self, seq_folder, left_folder_name, right_folder_name):
        self.seq_folder = seq_folder
        self.left_folder_name = left_folder_name
        self.right_folder_name = right_folder_name
        self.template_path = os.path.join(os.path.dirname(__file__), "cfg", "edex_tartanair.json")

    def __call__(self, *args, **kwargs):
        with open(self.template_path, mode="r") as f:
            edex = json.load(f)

        edex[0]["frame_end"] = len(os.listdir(os.path.join(self.seq_folder, self.left_folder_name))) - 1
        edex[1]["sequence"][0][0] = edex[1]["sequence"][0][0].format(
            image_folder=self.left_folder_name,
        )
        edex[1]["sequence"][1][0] = edex[1]["sequence"][1][0].format(
            image_folder=self.right_folder_name
        )
        edex_name = os.path.join(self.seq_folder, "cfg.edex")
        with open(edex_name, mode='w') as f:
            f.write(json.dumps(edex, indent=4))
