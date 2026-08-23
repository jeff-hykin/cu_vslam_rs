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

import math, re
import numpy as np

""" Helper fuctions for Quaternion computations in wxyz order. """

def vector_norm(data):
    """ Return length of the array. """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        return math.sqrt(np.dot(data, data))
    data *= data
    out = np.atleast_1d(np.sum(data))
    np.sqrt(out, out)
    return out

def quaternion_from_euler(pitch, yaw, roll):
    """ Return quaternion from Euler angles. """
    ai = pitch / 2.0
    aj = yaw / 2.0
    ak = roll / 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    quaternion[0] = cj*cc + sj*ss
    quaternion[1] = cj*sc - sj*cs
    quaternion[2] = cj*ss + sj*cc
    quaternion[3] = cj*cs - sj*sc

    return quaternion

def quaternion_conjugate(quaternion):
    """ Return conjugate of quaternion. """
    return np.array((quaternion[0], -quaternion[1],
                        -quaternion[2], -quaternion[3]), dtype=np.float64)

def quaternion_normalize(quaternion):
    """ Return normalized quaternion. """
    return quaternion / np.dot(quaternion, quaternion)

def quaternion_multiply(quaternion1, quaternion0):
    """ Return multiplication of two quaternions.
        out = quaternion1 * quaternion0
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    quaternion2 = np.array(
            (w1*w0 - x1*x0 - y1*y0 - z1*z0,
             w1*x0 + x1*w0 + y1*z0 - z1*y0,
             w1*y0 - x1*z0 + y1*w0 + z1*x0,
             w1*z0 + x1*y0 - y1*x0 + z1*w0),
            dtype=np.float64)

    return quaternion_normalize(quaternion2)

def rotation_to_quaternion(matrix):
    """ Return quaternion from Rotation matrix of size 3x3
        based on https://arc.aiaa.org/doi/10.2514/1.31730"""
    matrix = np.asarray(matrix, dtype=float)

    if (matrix.ndim not in [2, 3] or
        matrix.shape[len(matrix.shape)-2:] != (3, 3)):
        raise ValueError("Expected `matrix` to have shape (3, 3) or "
                         "(N, 3, 3), got {}".format(matrix.shape))

    if matrix.shape == (3, 3):
        cmatrix = matrix[None, :, :]
    else:
        cmatrix = matrix

    num_rotations = cmatrix.shape[0]
    decision = np.zeros(4)

    quat = np.zeros([num_rotations, 4])

    for ind in range(num_rotations):
        decision[0] = cmatrix[ind, 0, 0]
        decision[1] = cmatrix[ind, 1, 1]
        decision[2] = cmatrix[ind, 2, 2]
        decision[3] = cmatrix[ind, 0, 0] + cmatrix[ind, 1, 1] \
                    + cmatrix[ind, 2, 2]
        choice = np.argmax(decision)

        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3

            quat[ind, i] = 1 - decision[3] + 2 * cmatrix[ind, i, i]
            quat[ind, j] = cmatrix[ind, j, i] + cmatrix[ind, i, j]
            quat[ind, k] = cmatrix[ind, k, i] + cmatrix[ind, i, k]
            quat[ind, 3] = cmatrix[ind, k, j] - cmatrix[ind, j, k]
        else:
            quat[ind, 0] = cmatrix[ind, 2, 1] - cmatrix[ind, 1, 2]
            quat[ind, 1] = cmatrix[ind, 0, 2] - cmatrix[ind, 2, 0]
            quat[ind, 2] = cmatrix[ind, 1, 0] - cmatrix[ind, 0, 1]
            quat[ind, 3] = 1 + decision[3]

    norm_val = np.linalg.norm(quat[0])

    return quat/norm_val

def _elementary_basis_index(axis):
    if axis == 'x': return 0
    elif axis == 'y': return 1
    elif axis == 'z': return 2

def _compute_euler_from_quat(quat, seq, extrinsic=False):

    if not extrinsic:
        seq = seq[::-1]

    i = _elementary_basis_index(seq[0])
    j = _elementary_basis_index(seq[1])
    k = _elementary_basis_index(seq[2])

    is_proper = i == k

    if is_proper:
        k = 3 - i - j # get third axis

    # Step 0
    # Check if permutation is even (+1) or odd (-1)
    sign = int((i-j)*(j-k)*(k-i)/2)

    num_rotations = quat.shape[0]
    angles = np.empty((num_rotations, 3))
    eps = 1e-7

    for ind in range(num_rotations):
        _angles = angles[ind, :]

        if is_proper:
            a = quat[ind, 3]
            b = quat[ind, i]
            c = quat[ind, j]
            d = quat[ind, k] * sign
        else:
            a = quat[ind, 3] - quat[ind, j]
            b = quat[ind, i] + quat[ind, k] * sign
            c = quat[ind, j] + quat[ind, 3]
            d = quat[ind, k] * sign - quat[ind, i]

        n2 = a**2 + b**2 + c**2 + d**2

        # Step 3
        # Compute second angle...
        # _angles[1] = 2*np.arccos(np.sqrt((a**2 + b**2) / n2))
        _angles[1] = np.arccos(2*(a**2 + b**2) / n2 - 1)

        # ... and check if equalt to is 0 or pi, causing a singularity
        safe1 = abs(_angles[1]) >= eps
        safe2 = abs(_angles[1] - np.pi) >= eps
        safe = safe1 and safe2

        # Step 4
        # compute first and third angles, according to case
        if safe:
            half_sum = np.arctan2(b, a) # == (alpha+gamma)/2
            half_diff = np.arctan2(-d, c) # == (alpha-gamma)/2

            _angles[0] = half_sum + half_diff
            _angles[2] = half_sum - half_diff

        else:
            # _angles[0] = 0

            if not extrinsic:
                # For intrinsic, set first angle to zero so that after reversal we
                # ensure that third angle is zero
                if not safe:
                    _angles[0] = 0
                if not safe1:
                    half_sum = np.arctan2(b, a)
                    _angles[2] = 2 * half_sum
                if not safe2:
                    half_diff = np.arctan2(-d, c)
                    _angles[2] = -2 * half_diff
            else:
                # For extrinsic, set third angle to zero
                if not safe:
                    _angles[2] = 0
                if not safe1:
                    half_sum = np.arctan2(b, a)
                    _angles[0] = 2 * half_sum
                if not safe2:
                    half_diff = np.arctan2(-d, c)
                    _angles[0] = 2 * half_diff

        for i_ in range(3):
            if _angles[i_] < -np.pi:
                _angles[i_] += 2 * np.pi
            elif _angles[i_] > np.pi:
                _angles[i_] -= 2 * np.pi

        # for Tait-Bryan angles
        if not is_proper:
            _angles[2] *= sign
            _angles[1] -= np.pi / 2

        if not extrinsic:
            # reversal
            _angles[0], _angles[2] = _angles[2], _angles[0]

        # Step 8
        if not safe:
            print("Gimbal lock detected. Setting third angle to zero "
                          "since it is not possible to uniquely determine "
                          "all angles.")

    return angles

def euler_from_quat(quat, seq, degrees=False):
    """ Return Euler angles from quaternion
    based on https://doi.org/10.1371/journal.pone.0276302"""
    if len(seq) != 3:
        raise ValueError("Expected 3 axes, got {}.".format(seq))

    intrinsic = (re.match(r'^[XYZ]{1,3}$', seq) is not None)
    extrinsic = (re.match(r'^[xyz]{1,3}$', seq) is not None)
    if not (intrinsic or extrinsic):
        raise ValueError("Expected axes from `seq` to be from "
                         "['x', 'y', 'z'] or ['X', 'Y', 'Z'], "
                         "got {}".format(seq))

    if any(seq[i] == seq[i+1] for i in range(2)):
        raise ValueError("Expected consecutive axes to be different, "
                         "got {}".format(seq))

    seq = seq.lower()

    if quat.ndim == 1:
        quat = quat[None, :]

    angles = _compute_euler_from_quat(quat, seq, extrinsic)
    angles = np.asarray(angles)

    if degrees:
        angles = np.rad2deg(angles)

    return angles[0]
