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
from abc import ABC
import numpy as np
from .utils import TartanAirReader, Rotator, Normalizer, Writer, TartanAirImageRename, TartanAirFillEdex, FolderRename


class BasicPipeline(ABC):
    def __call__(self):
        x = None
        for op in self.operations:
            x = op(x)


class TartanAirPipeline(BasicPipeline):
    def __init__(self, seq_folder, save_gt_folder, save_edex_folder):
        self.rotate_axis = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        self.pose_left_file = os.path.join(seq_folder, "pose_left.txt")

        os.makedirs(save_gt_folder, exist_ok=True)
        os.makedirs(save_edex_folder, exist_ok=True)

        self.operations = [
            TartanAirReader(self.pose_left_file),
            Normalizer(),
            Writer(seq_folder),
            TartanAirImageRename(seq_folder),
            FolderRename(seq_folder, old_name="image_left", new_name="00"),
            FolderRename(seq_folder, old_name="image_right", new_name="01"),
            TartanAirFillEdex(seq_folder, "00", "01")
        ]
