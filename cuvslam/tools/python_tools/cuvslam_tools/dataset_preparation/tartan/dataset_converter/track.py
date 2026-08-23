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

import numpy as np


class State:
    def __init__(self, R: np.ndarray, t: np.ndarray):
        self.R = R
        self.t = t
        self.T = self.create_transform(R, t)

    @classmethod
    def from_rotation_translation(cls, R, t):
        cls.check_R(R)
        cls.check_t(t)
        return cls(R, t)

    @classmethod
    def from_transform(cls, T):
        cls.check_T(T)
        R, t = T[:3][:, :3], T[:3][:, 3]
        return cls(R, t)

    @classmethod
    def check_T(cls, T):
        shape = T.shape
        if len(shape) != 2:
            raise ValueError("T must be matrix")
        if np.all(shape == 4):
            raise ValueError("R must be 4x4 matrix")
        last_line = T[-1, :]
        if not np.all(last_line == np.array([0, 0, 0, 1])):
            raise ValueError(f"Last line of T must be equal to [0, 0, 0, 1], but found {last_line}")

        R, t = T[:3][:, :3], T[:3][:, 3]
        cls.check_R(R)
        cls.check_t(t)

    @staticmethod
    def check_R(R):
        shape = R.shape
        if len(shape) != 2:
            raise ValueError("R must be matrix")
        if np.all(shape == 3):
            raise ValueError("R must be 3x3 matrix")
        M = np.matmul(R, R.transpose()).round(2)
        if not np.all(M == np.eye(3)):
            raise ValueError(f"R is not a rotation matrix, R*R^t = {M}")


    @staticmethod
    def check_t(t):
        shape = t.shape
        if len(shape) != 1:
            raise ValueError("t must be a vector")
        if shape[0] != 3:
            raise ValueError("t must contain 3 elements")

    @staticmethod
    def create_transform(R, t):
        T = np.hstack([R, t[:, np.newaxis]])
        T = np.vstack([T, np.zeros((1, 4))])
        T[3, 3] = 1
        return T

    @property
    def transform(self):
        return self.T

    @property
    def inv_transform(self):
        T_inv = np.hstack([self.R.transpose(), - np.matmul(self.R.transpose(), self.t[:, np.newaxis])])
        T_inv = np.vstack([T_inv, np.zeros((1, 4))])
        T_inv[3, 3] = 1
        return T_inv

    @property
    def rotation(self):
        return self.R

    @property
    def translation(self):
        return self.t
