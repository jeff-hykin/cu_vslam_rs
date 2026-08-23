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

"""Stage TartanAir-compatible sequences for the bundled dataset converter."""

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


CLASSIC_IMAGE_DIRS = ("image_left", "image_right")
CLASSIC_POSE_FILES = ("pose_left.txt", "pose_right.txt")
TARTANGROUND_CAMERA_SUFFIXES = ("front", "left", "right", "back", "top", "bottom")


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copytree_with_links(source: Path, destination: Path) -> None:
    def copy_function(src: str, dst: str) -> str:
        _link_or_copy_file(Path(src), Path(dst))
        return dst

    shutil.copytree(source, destination, copy_function=copy_function)


def _classic_sequences(seq_path: Path) -> list[Path]:
    sequences = []
    for image_left in seq_path.rglob("image_left"):
        candidate = image_left.parent
        if all((candidate / folder).is_dir() for folder in CLASSIC_IMAGE_DIRS) and all(
            (candidate / file_name).is_file() for file_name in CLASSIC_POSE_FILES
        ):
            sequences.append(candidate)
    return sorted(sequences)


def _tartanground_sequence_dirs(seq_path: Path) -> Iterable[Path]:
    for pose in seq_path.rglob("pose_lcam_*.txt"):
        yield pose.parent


def _available_tartanground_pairs(sequence_dir: Path) -> list[str]:
    pairs = []
    for suffix in TARTANGROUND_CAMERA_SUFFIXES:
        required_paths = (
            sequence_dir / f"image_lcam_{suffix}",
            sequence_dir / f"image_rcam_{suffix}",
            sequence_dir / f"pose_lcam_{suffix}.txt",
            sequence_dir / f"pose_rcam_{suffix}.txt",
        )
        if all(path.exists() for path in required_paths):
            pairs.append(suffix)
    return pairs


def _classic_image_name(source_name: str, camera_name: str, classic_side: str) -> str:
    source = Path(source_name)
    camera_suffix = f"_{camera_name}"
    frame_id = source.stem
    if frame_id.endswith(camera_suffix):
        frame_id = frame_id[: -len(camera_suffix)]
    return f"{frame_id}_{classic_side}{source.suffix}"


def _stage_tartanground_image_dir(source: Path, destination: Path, camera_name: str, classic_side: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for image in sorted(source.iterdir()):
        if image.is_file():
            _link_or_copy_file(image, destination / _classic_image_name(image.name, camera_name, classic_side))


def _stage_tartanground_sequence(seq_path: Path, sequence_dir: Path, output_dir: Path, suffix: str) -> Path:
    relative_sequence = sequence_dir.relative_to(seq_path)
    staged_sequence = output_dir / relative_sequence.parent / f"{relative_sequence.name}_{suffix}"

    left_camera = f"lcam_{suffix}"
    right_camera = f"rcam_{suffix}"
    _stage_tartanground_image_dir(
        sequence_dir / f"image_{left_camera}",
        staged_sequence / "image_left",
        left_camera,
        "left",
    )
    _stage_tartanground_image_dir(
        sequence_dir / f"image_{right_camera}",
        staged_sequence / "image_right",
        right_camera,
        "right",
    )
    shutil.copy2(sequence_dir / f"pose_{left_camera}.txt", staged_sequence / "pose_left.txt")
    shutil.copy2(sequence_dir / f"pose_{right_camera}.txt", staged_sequence / "pose_right.txt")
    return staged_sequence


def stage_sequences(seq_path: Path, output_dir: Path) -> list[Path]:
    """Stage classic or TartanGround sequences under output_dir for conversion."""
    seq_path = seq_path.resolve()
    output_dir = output_dir.resolve()

    classic_sequences = _classic_sequences(seq_path)
    if classic_sequences:
        _copytree_with_links(seq_path, output_dir)
        return [output_dir / sequence.relative_to(seq_path) for sequence in classic_sequences]

    staged_sequences = []
    for sequence_dir in sorted(set(_tartanground_sequence_dirs(seq_path))):
        for suffix in _available_tartanground_pairs(sequence_dir):
            staged_sequences.append(_stage_tartanground_sequence(seq_path, sequence_dir, output_dir, suffix))

    return staged_sequences


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage TartanAir-compatible sequences for EDEX conversion.")
    parser.add_argument("--seq-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    staged_sequences = stage_sequences(args.seq_path, args.output_dir)
    if not staged_sequences:
        print(f"error: no convertible TartanAir or TartanGround stereo sequences found under {args.seq_path}.")
        print("       Need classic image_left/image_right + pose_left.txt/pose_right.txt,")
        print("       or TartanGround image_lcam_*/image_rcam_* + pose_lcam_*/pose_rcam_* pairs.")
        return 1

    print("Staged sequences:")
    for sequence in staged_sequences:
        print(f"  - {sequence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
