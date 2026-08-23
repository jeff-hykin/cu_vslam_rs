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

def Skew3(vec3, m = 0):
	return np.array([
		[m,	   -vec3[2],   vec3[1]],
		[vec3[2],		m,  -vec3[0]],
		[-vec3[1], vec3[0],		m]])

def inverse(mat4):
	R = mat4[:3, :3]
	t = mat4[:3, -1]

	T = np.eye(4)
	T[:3, :3] = R.transpose()
	T[:3, -1] = - np.matmul(R.transpose(), t)
	return T

def Exp3(twist3):
	if np.linalg.norm(twist3) < 1e-3 :
		return np.eye(3) + Skew3(twist3)

	theta = np.linalg.norm(twist3)
	c = np.cos(theta)
	s_theta = np.sin(theta) / theta
	n = twist3 / theta

	n = n[:, np.newaxis]

	return Skew3(twist3 * s_theta, c) + (1 - c) * np.matmul(n, n.transpose())

def change(M, T):
	return np.matmul(T, np.matmul(M, inverse(T)))

def evaluate_error(tr1, tr2, R, update = np.zeros(3)):
	out = 0

	R_updated = np.matmul(Exp3(update), R)
	for i in range(len(tr1)):
		t1 = tr1[i]
		t2 = tr2[i]

		e = np.matmul(R_updated, t1[:, np.newaxis]).ravel() - t2

		out += np.dot(e, e)
	return out

def calc_H_r(tr1, tr2, R):
	H = np.zeros((3, 3))
	rhs = np.zeros(3)

	for i in range(len(tr1)):
		t1 = tr1[i]
		t2 = tr2[i]
		e = np.matmul(R, t1[:, np.newaxis]).ravel() - t2

		# print(np.matmul(R, t1[:, np.newaxis]).ravel())
		J = -Skew3(np.matmul(R, t1[:, np.newaxis]).ravel())

		H += np.matmul(J.transpose(), J)

		# print(J.transpose(), e[:, np.newaxis])
		rhs -= np.matmul(J.transpose(), e[:, np.newaxis]).ravel()
	return H, rhs

def optimize_R(poses1, poses2):
	tr1 = [x[:3, 3] for x in poses1]
	tr2 = [x[:3, 3] for x in poses2]

	R = np.eye(3)

	initial_cost = evaluate_error(tr1, tr2, R)
	curr_cost = initial_cost

	lambda_ = 1

	H, rhs = calc_H_r(tr1, tr2, R)

	scaling = H.diagonal()

	for i in range(30):
		aug_H = H + np.diag(lambda_ * scaling)

		U, S, Vh = np.linalg.svd(aug_H, full_matrices=True)
		c = np.dot(U.transpose(), rhs)
		w = np.dot(np.diag(1. / S), c)
		step = np.dot(Vh.conj().transpose(), w)

		cost = evaluate_error(tr1, tr2, R, step)

		# print(f"cost = {cost}, curr_cost = {curr_cost}")

		predicted_relative_reduction = np.dot(step, np.dot(H, step)) / curr_cost + 2 * lambda_ * step.dot(step) / curr_cost;

		rho = (1 - cost / curr_cost) / predicted_relative_reduction

		if rho > 0.25 :
			R = np.matmul(Exp3(step), R)

			if rho > 0.75 :
				lambda_ *= 0.2;

			curr_cost = cost;
			H, rhs = calc_H_r(tr1, tr2, R)
			scaling = H.diagonal()
		else:
			lambda_ *= 2.;


	return R
