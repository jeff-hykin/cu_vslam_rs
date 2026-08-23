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

"""Processing decorators used to transform tracker input streams."""

from typing import Optional, Sequence

import numpy as np

from cuvslam_tools.tracker.dataset_reader import Processing


class ProcessingDecorator:
    """Base decorator for the Processing protocol. Subclasses override only what they touch."""

    def __init__(self, inner: Processing):
        """Wrap another Processing implementation."""
        self.inner = inner

    def process_images(self, *args, **kwargs):
        """Forward image frames to the wrapped processor."""
        self.inner.process_images(*args, **kwargs)

    def process_imu(self, *args, **kwargs):
        """Forward IMU measurements to the wrapped processor."""
        self.inner.process_imu(*args, **kwargs)

    def get_camera_pose(self, frame_id: int):
        """Return a camera pose from the wrapped processor."""
        return self.inner.get_camera_pose(frame_id)

    def set_frame_metadata(self, *args, **kwargs):
        """Forward frame metadata to the wrapped processor."""
        self.inner.set_frame_metadata(*args, **kwargs)


class BlackoutFilter(ProcessingDecorator):
    """Periodically zeros out images/masks before forwarding (sensor-failure simulation).

    Every `period` frames, the next `duration` frames are blacked out. The pattern runs
    against the emitted frame counter so it spans loop wraps in repeat/shuttle mode.
    """

    def __init__(self, inner: Processing, period: int, duration: int):
        """Configure blackout period and duration before forwarding frames."""
        super().__init__(inner)
        if period <= 0 or duration < 0:
            raise ValueError("period must be > 0 and duration must be >= 0")
        self.period = period
        self.duration = duration

    def process_images(self, frame_id: int, timestamps: Sequence[int],
                       images: Sequence, masks: Sequence,
                       depths: Optional[Sequence] = None):
        """Zero selected frames before forwarding them to the wrapped processor."""
        if (frame_id % self.period) < self.duration:
            images = [np.zeros_like(im) for im in images]
            masks = [np.zeros_like(m) if m.size else m for m in masks]
            if depths is not None:
                depths = [np.zeros_like(d) if d.size else d for d in depths]
        self.inner.process_images(frame_id, timestamps, images, masks, depths)
