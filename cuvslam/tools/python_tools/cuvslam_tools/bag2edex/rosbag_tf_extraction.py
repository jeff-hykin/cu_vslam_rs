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

import collections
import pathlib

import pandas as pd
from pytransform3d import trajectories
from pytransform3d import transform_manager

from . import (
    open_rosbag_reader,
)


def _extract_tf_dataframe_from_bag(
    rosbag_path: pathlib.Path, ros_distribution: str
) -> pd.DataFrame:
    """
    Read a bag and return a pandas dataframe containing the transforms from /tf and /tf_static.
    """
    data = collections.defaultdict(list)
    with open_rosbag_reader(rosbag_path, ros_distribution) as reader:
        connections = [
            x for x in reader.connections if x.topic in ["/tf_static", "/tf"]
        ]
        for connection, _, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            for tf in msg.transforms:
                data["topic"].append(connection.topic)
                data["sec"].append(tf.header.stamp.sec)
                data["nanosec"].append(tf.header.stamp.nanosec)
                data["parent"].append(tf.header.frame_id)
                data["child"].append(tf.child_frame_id)
                data["x"].append(tf.transform.translation.x)
                data["y"].append(tf.transform.translation.y)
                data["z"].append(tf.transform.translation.z)
                data["qw"].append(tf.transform.rotation.w)
                data["qx"].append(tf.transform.rotation.x)
                data["qy"].append(tf.transform.rotation.y)
                data["qz"].append(tf.transform.rotation.z)
    df = pd.DataFrame(data)
    df["time_s"] = df["sec"] + df["nanosec"] / 10**9
    return df


def get_transform_manager_from_bag(
    rosbag_path: pathlib.Path,
    ros_distribution: str,
) -> transform_manager.TemporalTransformManager:
    """Reads a bag and returns a TemporalTransformManager containing all transforms inside."""
    df = _extract_tf_dataframe_from_bag(rosbag_path, ros_distribution)

    tf_manager = transform_manager.TemporalTransformManager()
    for (parent, child), group in df.groupby(["parent", "child"]):
        timestamps = group["time_s"].to_numpy()
        pqs = group[["x", "y", "z", "qw", "qx", "qy", "qz"]].to_numpy()
        tf_series = transform_manager.NumpyTimeseriesTransform(timestamps, pqs)
        tf_manager.add_transform(child, parent, tf_series)

    return tf_manager


def get_static_transform_manager_from_bag(
    rosbag_path: pathlib.Path,
    ros_distribution: str,
) -> transform_manager.TransformManager:
    """Reads a bag and returns a TransformManager containing the static transforms only."""
    df = _extract_tf_dataframe_from_bag(rosbag_path, ros_distribution)
    # Filter out any non-static tfs.
    df_static = df[df["topic"] == "/tf_static"].reset_index()
    transforms = trajectories.transforms_from_pqs(
        df_static[["x", "y", "z", "qw", "qx", "qy", "qz"]]
    )
    tf_manager = transform_manager.TransformManager()
    for index, row in df_static.iterrows():
        tf_manager.add_transform(row["child"], row["parent"], transforms[index])
    return tf_manager
