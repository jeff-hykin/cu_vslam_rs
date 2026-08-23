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
import json
import os
import click

from scipy.spatial.transform import Rotation

from eval_ate_rpe import calculate_ate, calculate_rpe, calculate_rms_ate_rpe, plot_trajectory
from optimize import optimize_R

import matplotlib.pyplot as plt

def load_poses(path):
	out = []
	with open(path) as f:
		for line in f:
			p = np.array(list(map(float, line.split()))).reshape((3,4))
			p = np.vstack([p, [0, 0, 0, 1]])
			out.append(p)
	return out

def pose_inverse(pose):
	R = pose[:3, :3]
	T = pose[:3,  3]

	out = np.eye(4)
	out[:3, :3] = R.transpose()
	out[:3,  3] = - np.matmul(R.transpose(), T)
	return out

def change_CS(pose, M_new_from_old):
	return np.matmul(pose_inverse(M_new_from_old), np.matmul(pose, M_new_from_old))

def draw_sequenses(ax, trajectory, color):
	points = []
	for pose in trajectory:
		T = pose[:3, 3]
		points.append([T[0], T[2]])
	points = np.array(points)

	ax.scatter(points[:, 0], points[:, 1], c=color)

def move_to_origin(trajectory):
	inverse_start_pose = pose_inverse(trajectory[0])

	out = []
	for p in trajectory:
		out.append(np.matmul(inverse_start_pose, p))
	return out

def change_trajectory_CS(trajectory, M):
	out = []
	for p in trajectory:
		out.append(change_CS(p, M))
	return out


@click.command()
@click.argument('path_to_result_trajectory') # e.g. path to reporter result trajectory
@click.argument('path_to_gt_trajectory') # path to gt in our datasets
def main(path_to_result_trajectory, path_to_gt_trajectory):
	et = load_poses(path_to_result_trajectory)
	gt = load_poses(path_to_gt_trajectory)

	gt = move_to_origin(gt)
	et = move_to_origin(et)

	ET_FROM_GT = np.eye(4)
	ET_FROM_GT[:3, :3] = optimize_R(et, gt)
	et = change_trajectory_CS(et, pose_inverse(ET_FROM_GT))

	ate, errors, gt_translations, et_translations, rms_rpe_rotational, rotational_errors, rms_rpe_translational, translational_errors = calculate_rms_ate_rpe(
		gt, et, 1)

	plot_trajectory(gt_translations, et_translations)

	print(f"ATE: {calculate_ate(gt, et)[0]}")
	print(f"RPE: {calculate_rpe(gt, et, 5)[0]}")


if __name__ == '__main__':
	main()
