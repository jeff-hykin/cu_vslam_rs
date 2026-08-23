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

import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_ate(gt_poses, et_poses):
    # Extract translation vectors from ground truth and estimated poses
    gt_translations = np.array([pose[0:3, 3] for pose in gt_poses])
    et_translations = np.array([pose[0:3, 3] for pose in et_poses])

    # Compute the Absolute Trajectory Error (ATE)
    errors = np.sqrt(np.sum(np.square(gt_translations - et_translations), axis=1))
    ate = np.mean(errors)

    return ate, errors, gt_translations, et_translations


def calculate_rpe(gt_poses, et_poses, delta):
    # Extract rotation matrices and translation vectors from ground truth and estimated poses
    gt_rotations = np.array([pose[0:3, 0:3] for pose in gt_poses])
    et_rotations = np.array([pose[0:3, 0:3] for pose in et_poses])
    gt_translations = np.array([pose[0:3, 3] for pose in gt_poses])
    et_translations = np.array([pose[0:3, 3] for pose in et_poses])

    # Compute the relative pose differences
    relative_rotations = []
    relative_translations = []
    for i in range(delta, len(gt_poses), delta):
        delta_gt_rotation = gt_rotations[i - delta].T.dot(gt_rotations[i])
        delta_et_rotation = et_rotations[i - delta].T.dot(et_rotations[i])
        delta_gt_translation = gt_rotations[i - delta].T.dot(gt_translations[i] - gt_translations[i - delta])
        delta_et_translation = et_rotations[i - delta].T.dot(et_translations[i] - et_translations[i - delta])
        relative_rotations.append(delta_gt_rotation.T.dot(delta_et_rotation))
        relative_translations.append(delta_gt_translation - delta_et_translation)
    relative_rotations = np.array(relative_rotations)
    relative_translations = np.array(relative_translations)

    # Compute the rotational errors as rotation angles
    trace_values = np.clip((np.trace(relative_rotations, axis1=1, axis2=2) - 1) / 2, -1, 1)
    rotational_errors = np.arccos(trace_values)

    # Compute the translational errors as Euclidean distances
    translational_errors = np.linalg.norm(relative_translations, axis=1)

    # Compute the Root Mean Squared Relative Pose Error (RMS RPE) for rotational and translational errors
    rms_rpe_rotational = np.sqrt(np.mean(np.square(rotational_errors)))
    rms_rpe_translational = np.sqrt(np.mean(np.square(translational_errors)))

    return rms_rpe_rotational, rotational_errors, rms_rpe_translational, translational_errors


def load_poses_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    poses = [np.array(list(entry.values()))[0] for entry in data]

    return poses


def calculate_rms_ate_rpe(gt_poses, et_poses, delta):
    # Load ground truth and estimated poses
    # gt_poses = load_poses_from_json(gt_file)
    # et_poses = load_poses_from_json(et_file)

    # Calculate the Absolute Trajectory Error (ATE) and get the errors, ground truth translations, and estimated translations
    ate, errors, gt_translations, et_translations = calculate_ate(gt_poses, et_poses)

    # Calculate the Root Mean Squared Relative Pose Error (RMS RPE) and get the rotational and translational errors
    rms_rpe_rotational, rotational_errors, rms_rpe_translational, translational_errors = calculate_rpe(
        gt_poses, et_poses, delta
    )

    return (
        ate,
        errors,
        gt_translations,
        et_translations,
        rms_rpe_rotational,
        rotational_errors,
        rms_rpe_translational,
        translational_errors,
    )


def save_errors_to_csv(errors, gt_translations, et_translations, frame_ids, output_file):
    data = zip(errors, gt_translations, et_translations, frame_ids)
    header = ["Error", "GT Translation", "ET Translation", "Frame ID"]

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Errors saved to '{output_file}' successfully!")


def plot_trajectory(gt_translations, et_translations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot ground truth trajectory
    gt_x = gt_translations[:, 0]
    gt_y = gt_translations[:, 1]
    gt_z = gt_translations[:, 2]
    ax.plot3D(gt_x, gt_y, gt_z, c="b", label="Ground Truth")

    # Plot estimated trajectory
    et_x = et_translations[:, 0]
    et_y = et_translations[:, 1]
    et_z = et_translations[:, 2]
    ax.plot3D(et_x, et_y, et_z, c="r", label="cuVSLAM")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


# Example usage
# gt_file_path = "gt.json"  # Replace with the path to your ground truth JSON file
# et_file_path = "et.json"  # Replace with the path to your estimated trajectory JSON file
# output_file_path = "ate_errors.csv"  # Replace with the desired path for the output CSV file
# delta = 5  # Replace with the desired delta value

# ate, errors, gt_translations, et_translations, rms_rpe_rotational, rotational_errors, rms_rpe_translational, translational_errors = calculate_rms_ate_rpe(
#     gt_file_path, et_file_path, delta
# )
# frame_ids = range(len(errors))

# save_errors_to_csv(errors, gt_translations, et_translations, frame_ids, output_file_path)
# print("RMS ATE:", ate)
# print("RMS RPE Rotational:", rms_rpe_rotational)
# print("RMS RPE Translational:", rms_rpe_translational)

# plot_trajectory(gt_translations, et_translations)
