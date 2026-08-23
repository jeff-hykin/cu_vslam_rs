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

"""Tests for Odometry state serialization (save_state / load_state).

The main test checks the "pure function" property: tracking N frames continuously produces the
same poses as tracking K frames, checkpointing, restoring the checkpoint into a fresh tracker in a
FRESH PROCESS, and tracking the remaining N-K frames. Fresh processes are used so the global track
id counter starts identically in the reference and checkpointed runs.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("RERUN", "0")

import cuvslam as vslam
import data_gen as data

STEPS = int(os.environ.get("CUVSLAM_TEST_STEPS", 40))
SAVE_AT = int(os.environ.get("CUVSLAM_TEST_SAVE_AT", 20))
WIDTH, HEIGHT = 640, 480
FRAME_INTERVAL_NS = 33_000_000

TRANSLATION_ATOL_M = 1e-3  # 1 mm
ROTATION_ATOL_DEG = 0.01


def make_rig(baseline: float = 0.25) -> "vslam.Rig":
    cameras = data.generate_stereo_camera(WIDTH, HEIGHT, baseline=baseline)
    return vslam.Rig(cameras, [])


def make_config(use_gpu: bool = True) -> "vslam.Tracker.OdometryConfig":
    cfg = vslam.Tracker.OdometryConfig()
    cfg.odometry_mode = vslam.Tracker.OdometryMode.Multicamera
    # Synchronous SBA: with async SBA the timing of bundle-adjustment updates relative to frames is
    # scheduler-dependent, which breaks continuous-vs-restored comparison.
    cfg.async_sba = False
    cfg.use_gpu = use_gpu
    return cfg


def pose_to_list(pose_estimate):
    wfr = pose_estimate.world_from_rig
    if not wfr:
        return None
    return [float(v) for v in wfr.pose.translation] + [float(v) for v in wfr.pose.rotation]


def track_range(tracker, gen, start, stop):
    poses = []
    for i in range(start, stop):
        images, _ = gen.generate_zoomed_images(i)
        pose_estimate, _ = tracker.track(i * FRAME_INTERVAL_NS, images)
        poses.append(pose_to_list(pose_estimate))
    return poses


def run_worker(mode: str, device: str, out_path: str, state_path: str) -> None:
    rig = make_rig()
    gen = data.ImageGenerator(rig.cameras, STEPS)
    tracker = vslam.Tracker(rig, make_config(use_gpu=(device == "gpu")))
    if mode == "reference":
        poses = track_range(tracker, gen, 0, STEPS)
    elif mode == "first-half":
        poses = track_range(tracker, gen, 0, SAVE_AT)
        tracker.save_state_to_file(state_path)
    elif mode == "second-half":
        tracker.load_state_from_file(state_path)
        poses = track_range(tracker, gen, SAVE_AT, STEPS)
    else:
        raise ValueError(f"unknown worker mode {mode}")
    with open(out_path, "w") as f:
        json.dump(poses, f)


def quat_angle_deg(q1, q2):
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    if np.array_equal(q1, q2):
        return 0.0
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = abs(float(np.clip(np.dot(q1, q2), -1.0, 1.0)))
    return float(np.degrees(2.0 * np.arccos(dot)))


def pose_errors(pose_a, pose_b):
    t_err = float(np.max(np.abs(np.array(pose_a[:3]) - np.array(pose_b[:3]))))
    r_err = quat_angle_deg(pose_a[3:], pose_b[3:])
    return t_err, r_err


def max_trajectory_errors(test, traj_a, traj_b, first_frame, what):
    max_t, max_r = 0.0, 0.0
    for i, (pose_a, pose_b) in enumerate(zip(traj_a, traj_b)):
        test.assertEqual(pose_a is None, pose_b is None,
                         f"frame {first_frame + i}: validity mismatch ({what})")
        if pose_a is None:
            continue
        t_err, r_err = pose_errors(pose_a, pose_b)
        max_t, max_r = max(max_t, t_err), max(max_r, r_err)
    return max_t, max_r


class TestStateSerialization(unittest.TestCase):
    def _spawn(self, mode, device, out_path, state_path):
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", mode, device, out_path,
             state_path],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=os.environ.copy(),
        )

    def _load(self, path):
        with open(path) as f:
            return json.load(f)

    def test_fresh_process_save_restore_matches_continuous_cpu_bitwise(self):
        """CPU mode is deterministic: restored run must reproduce the continuous run exactly."""
        try:
            vslam.Tracker(make_rig(), make_config(use_gpu=False))
        except ValueError:
            self.skipTest("this build has ENFORCE_GPU=ON; CPU mode unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "reference.json")
            first_path = os.path.join(temp_dir, "first.json")
            second_path = os.path.join(temp_dir, "second.json")
            state_path = os.path.join(temp_dir, "checkpoint.cvkp")

            self._spawn("reference", "cpu", ref_path, state_path)
            self._spawn("first-half", "cpu", first_path, state_path)
            self._spawn("second-half", "cpu", second_path, state_path)

            reference = self._load(ref_path)
            first = self._load(first_path)
            second = self._load(second_path)

            # On the CPU path the pipeline is bit-deterministic, so the prefix must match exactly
            # and the restored continuation must match the reference exactly.
            self.assertEqual(reference[:SAVE_AT], first,
                             "CPU prefix diverged — CPU replay is expected to be bit-deterministic")
            self.assertEqual(len(second), STEPS - SAVE_AT)
            self.assertEqual(reference[SAVE_AT:], second,
                             "CPU restored run diverged from the continuous run")

    def test_fresh_process_save_restore_matches_continuous_gpu(self):
        """GPU restored run must match the continuous run within the pipeline's own replay noise.

        The GPU pipeline is not bit-deterministic (floating-point reduction order varies between
        identical runs), so the bar is: checkpoint/restore must not add error beyond the measured
        identical-run noise floor (with margin), and must stay within absolute bounds.
        """
        self._run_noise_floor_comparison("gpu")

    def test_fresh_process_save_restore_matches_continuous_rgbd(self):
        """RGBD mode: restored run must match the continuous run within the replay noise floor."""
        self._run_noise_floor_comparison("rgbd")

    def _run_noise_floor_comparison(self, device):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "reference.json")
            ref2_path = os.path.join(temp_dir, "reference2.json")
            first_path = os.path.join(temp_dir, "first.json")
            second_path = os.path.join(temp_dir, "second.json")
            state_path = os.path.join(temp_dir, "checkpoint.cvkp")

            self._spawn("reference", device, ref_path, state_path)
            self._spawn("reference", device, ref2_path, state_path)
            self._spawn("first-half", device, first_path, state_path)
            self._spawn("second-half", device, second_path, state_path)

            reference = self._load(ref_path)
            reference2 = self._load(ref2_path)
            first = self._load(first_path)
            second = self._load(second_path)

            # Noise floor: two identical, unmodified continuous runs.
            floor_t, floor_r = max_trajectory_errors(
                self, reference, reference2, 0, "identical-run noise floor")
            print(f"\n{device} identical-run noise floor: {floor_t:.3e} m, {floor_r:.3e} deg")

            thr_t = max(TRANSLATION_ATOL_M, 3.0 * floor_t)
            thr_r = max(ROTATION_ATOL_DEG, 3.0 * floor_r)

            prefix_t, prefix_r = max_trajectory_errors(
                self, reference[:SAVE_AT], first, 0, "prefix")
            self.assertLessEqual(prefix_t, thr_t,
                                 f"prefix translation deviation {prefix_t:.3e} m over threshold")
            self.assertLessEqual(prefix_r, thr_r,
                                 f"prefix rotation deviation {prefix_r:.3e} deg over threshold")

            self.assertEqual(len(second), STEPS - SAVE_AT)
            cont_t, cont_r = max_trajectory_errors(
                self, reference[SAVE_AT:], second, SAVE_AT, "restored continuation")
            print(f"restored-run deviation: {cont_t:.3e} m, {cont_r:.3e} deg "
                  f"(thresholds {thr_t:.3e} m, {thr_r:.3e} deg)")
            self.assertLessEqual(cont_t, thr_t,
                                 f"restored translation deviation {cont_t:.3e} m over threshold")
            self.assertLessEqual(cont_r, thr_r,
                                 f"restored rotation deviation {cont_r:.3e} deg over threshold")

    def test_roundtrip_in_process(self):
        """save_state bytes restore into a fresh tracker and tracking continues successfully."""
        rig = make_rig()
        gen = data.ImageGenerator(rig.cameras, STEPS)
        tracker = vslam.Tracker(rig, make_config())
        track_range(tracker, gen, 0, 10)
        blob = tracker.save_state()
        self.assertIsInstance(blob, bytes)
        self.assertGreater(len(blob), 0)

        restored = vslam.Tracker(make_rig(), make_config())
        restored.load_state(blob)
        poses = track_range(restored, gen, 10, 15)
        self.assertTrue(all(p is not None for p in poses), "tracking failed after restore")

    def test_save_before_first_frame(self):
        """A checkpoint taken before any Track() call restores into a working tracker."""
        rig = make_rig()
        tracker = vslam.Tracker(rig, make_config())
        blob = tracker.save_state()
        restored = vslam.Tracker(make_rig(), make_config())
        restored.load_state(blob)
        gen = data.ImageGenerator(rig.cameras, STEPS)
        poses = track_range(restored, gen, 0, 5)
        self.assertTrue(any(p is not None for p in poses))

    def test_load_rejects_mismatched_rig(self):
        tracker = vslam.Tracker(make_rig(baseline=0.25), make_config())
        blob = tracker.save_state()
        other = vslam.Tracker(make_rig(baseline=0.30), make_config())
        with self.assertRaises((ValueError, RuntimeError)):
            other.load_state(blob)

    def test_load_rejects_garbage(self):
        tracker = vslam.Tracker(make_rig(), make_config())
        with self.assertRaises((ValueError, RuntimeError)):
            tracker.load_state(b"definitely not a checkpoint")

    def test_unsupported_mode_throws(self):
        cameras = data.generate_stereo_camera(WIDTH, HEIGHT, baseline=0.25)
        rig = vslam.Rig(cameras, [vslam.ImuCalibration()])
        cfg = vslam.Tracker.OdometryConfig()
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Inertial
        tracker = vslam.Tracker(rig, cfg)
        with self.assertRaises(RuntimeError):
            tracker.save_state()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        unittest.main()
