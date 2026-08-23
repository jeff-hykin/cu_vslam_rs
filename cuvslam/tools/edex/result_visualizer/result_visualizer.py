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

import argparse
import json
import math
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy
import quat_computations as tr

if int(matplotlib.__version__.split('.')[0]) > 0:
    from mpl_toolkits.mplot3d import Axes3D

labels = ['x', 'y', 'z', 'w']


def plot_translation_3d(plt, ts):
    ax = plt.axes(projection='3d')

    def xs_(t):
        return t[2]

    def ys_(t):
        return t[0]

    def zs_(t):
        return t[1]

    min_x = min(min(xs_(t)) for t in ts)
    max_x = max(max(xs_(t)) for t in ts)
    min_y = min(min(ys_(t)) for t in ts)
    max_y = max(max(ys_(t)) for t in ts)
    min_z = min(min(zs_(t)) for t in ts)
    max_z = max(max(zs_(t)) for t in ts)
    x_d = max_x - min_x
    y_d = max_y - min_y
    z_d = max_z - min_z
    max_d = max(x_d, y_d, z_d)

    plt.xlim(min_x / x_d * max_d, max_x / x_d * max_d)
    plt.ylim(min_y / y_d * max_d, max_y / y_d * max_d)
    ax.set_zlim(min_z / z_d * max_d, max_z / z_d * max_d)
    plt.autoscale(False)

    color_values = list(mcolors.TABLEAU_COLORS.values())
    for i, t in enumerate(ts):
        if i == 0:
            color = 'black'
        else:
            color = color_values[i - 1]

        xs = xs_(t)
        ys = ys_(t)
        zs = zs_(t)
        ax.plot(xs, ys, zs, '*-', color=color)
        ax.text(xs[0], ys[0], zs[0], "S", color=color)
        ax.text(xs[-1], ys[-1], zs[-1], "E", color=color)

    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.legend(range(len(ts)))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("edex_file", nargs='+', help='edex result file')
    args = parser.parse_args()

    translation = [[] for f in args.edex_file]
    rotation = [[] for f in args.edex_file]

    for idx, edex_file in enumerate(args.edex_file):
        with open(edex_file) as fp:
            edex = json.load(fp)

        positions = edex[1]["rig_positions"]
        indexes = sorted([int(i) for i in positions])

        frame_range = list(range(indexes[0], indexes[-1] + 1))
        if indexes != frame_range:
            missed_frames_str = ','.join([str(i) for i in set(frame_range) - set(indexes)])
            print("Found missed frames in %s: [%s]" % (edex_file, missed_frames_str))

        translation[idx] = [[] for t in range(3)]
        rotation[idx] = [[] for r in range(3)]

        for i in indexes:
            pos = positions[str(i)]

            for j, t in enumerate(pos["translation"]):
                translation[idx][j].append(t)

            if len(pos["rotation"]) == 9:
                # if case if we have rotation matrix of size 3x3 in edex file
                rot_matrix_tmp = numpy.array(pos["rotation"])
                rot_matrix_tmp.resize([3, 3])
                quat_tmp = tr.rotation_to_quaternion(rot_matrix_tmp.T)
                pos["rotation"] = tr.euler_from_quat(quat_tmp, "zxy")

            for j, r in enumerate(pos["rotation"]):
                rotation[idx][j].append(r)

    # taking diff w.r.t first edex file for visualizing error
    translation_err = [[] for f in args.edex_file]
    rotation_err = [[] for f in args.edex_file]
    for idx in range(1, len(translation)):
        def dist(i, k):
            return tr.vector_norm((
                translation[i][0][k] - translation[0][0][k],
                translation[i][1][k] - translation[0][1][k],
                translation[i][2][k] - translation[0][2][k]
            ))

        translation_err[idx] = [dist(idx, k) for k in range(len(translation[idx][0]))]

        def angle(i, k):
            def make_q(i, k):
                return tr.quaternion_from_euler(
                    math.radians(rotation[i][0][k]),
                    math.radians(rotation[i][1][k]),
                    math.radians(rotation[i][2][k])
                )

            rotation_q1 = make_q(i, k)
            rotation_q0 = make_q(0, k)

            rotation_q0_conj = tr.quaternion_conjugate(rotation_q0)
            rotation_q2 = tr.quaternion_multiply(rotation_q0_conj, rotation_q1)

            return math.degrees(2 * math.acos(numpy.clip(rotation_q2[0], -1.0, 1.0)))

        rotation_err[idx] = [angle(idx, k) for k in range(len(rotation[idx][0]))]

    # plot absolute translation and rotation
    fig = plt.figure()
    fig.subplots_adjust(right=0.8)
    ax = fig.add_subplot(211)
    ax.set_title("translation (in m)")
    legend_labels = []
    for idx, tt in enumerate(translation):
        legend_labels.extend([label + str(idx) for label in labels[:len(tt)]])
        for t in tt:
            ax.plot(indexes, t)
    ax.legend(legend_labels, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)

    fig.subplots_adjust(hspace=.5)
    ax = fig.add_subplot(212)
    ax.set_title("rotation (degrees)")
    legend_labels = []
    for idx, rr in enumerate(rotation):
        legend_labels.extend([label + str(idx) for label in labels[:len(rr)]])
        for r in rr:
            ax.plot(indexes, r)
    ax.legend(legend_labels, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)

    plt.show()

    plot_translation_3d(plt, translation)

    # plot relative translation and rotation for visualizing error
    fig = plt.figure()
    fig.subplots_adjust(right=0.8)
    ax = fig.add_subplot(211)
    ax.set_title("translation error (in m)")
    for v in translation_err[1:]:
        ax.plot(indexes, v)
    ax.legend(range(1, len(translation_err)), bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)

    fig.subplots_adjust(hspace=.5)
    ax = fig.add_subplot(212)
    ax.set_title("rotation error (degrees)")
    for v in rotation_err[1:]:
        ax.plot(indexes, v)
    ax.legend(range(1, len(rotation_err)), bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)

    plt.show()


if __name__ == '__main__':
    main()
