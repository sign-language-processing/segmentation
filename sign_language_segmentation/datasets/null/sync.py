"""generate null pose files and an annotations cache.

Usage:
    uv run python -m sign_language_segmentation.datasets.null.sync --max_frames 1024 --sample_count 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

from sign_language_segmentation.datasets.common import CACHE_DIR

_DATASET_CACHE_DIR = CACHE_DIR / "null"
_DEFAULT_OUTPUT = _DATASET_CACHE_DIR / "annotations_cache.json"
_DEFAULT_POSES_DIR = _DATASET_CACHE_DIR / "poses"

_POSE_POINTS = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
_HAND_POINTS = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


def _sample_frame_counts(max_frames: int, sample_count: int) -> tuple[int, ...]:
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return tuple(max(2, round(max_frames * index / sample_count)) for index in range(1, sample_count + 1))


def _pose_header() -> PoseHeader:
    return PoseHeader(
        version=0.1,
        dimensions=PoseHeaderDimensions(width=1, height=1, depth=1),
        components=[
            PoseHeaderComponent("POSE_LANDMARKS", _POSE_POINTS, [], [], "XYZC"),
            PoseHeaderComponent("LEFT_HAND_LANDMARKS", _HAND_POINTS, [], [], "XYZC"),
            PoseHeaderComponent("RIGHT_HAND_LANDMARKS", _HAND_POINTS, [], [], "XYZC"),
        ],
    )


def _make_blank_pose(num_frames: int, fps: float) -> Pose:
    header = _pose_header()
    shape = (num_frames, 1, header.total_points(), 3)
    data = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape[:-1], dtype=np.float32)
    body = NumPyPoseBody(fps=int(round(fps)), data=data, confidence=confidence)
    return Pose(header=header, body=body)


def _load_static_frame(source_pose_path: Path, frame_index: int) -> Pose:
    if not source_pose_path.exists():
        raise FileNotFoundError(f"static source pose not found: {source_pose_path}")
    if frame_index < 0:
        raise ValueError("static source frame index must be non-negative")
    with open(source_pose_path, "rb") as f:
        return Pose.read(f, start_frame=frame_index, end_frame=frame_index + 1)


def _make_static_pose(source_frame: Pose, num_frames: int, fps: float) -> Pose:
    frame_data = source_frame.body.data[:1]
    frame_confidence = source_frame.body.confidence[:1]
    data = ma.concatenate([frame_data] * num_frames, axis=0).astype(np.float32)
    confidence = np.concatenate([frame_confidence] * num_frames, axis=0).astype(np.float32)
    body = NumPyPoseBody(fps=int(round(fps)), data=data, confidence=confidence)
    return Pose(header=source_frame.header, body=body)


def build_cache(
    output_path: Path,
    poses_dir: Path,
    max_frames: int,
    sample_count: int,
    fps: float,
    force: bool,
    static_source_pose: Path | None = None,
    static_source_frame: int = 0,
) -> dict:
    if fps < 24:
        raise ValueError("fps must be at least 24")
    frame_counts = _sample_frame_counts(max_frames=max_frames, sample_count=sample_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    static_frame = _load_static_frame(
        source_pose_path=static_source_pose,
        frame_index=static_source_frame,
    ) if static_source_pose is not None else None

    videos: dict[str, dict] = {}
    kinds = ("blank", "static") if static_frame is not None else ("blank",)
    for index, total_frames in enumerate(frame_counts, start=1):
        kind = kinds[(index - 1) % len(kinds)]
        video_id = f"null_{kind}_{index:02d}_{total_frames:06d}frames_{int(round(fps)):03d}fps"
        pose_path = poses_dir / f"{video_id}.pose"
        if force or not pose_path.exists():
            if kind == "blank":
                pose = _make_blank_pose(num_frames=total_frames + 1, fps=fps)
            else:
                pose = _make_static_pose(source_frame=static_frame, num_frames=total_frames + 1, fps=fps)
            with open(pose_path, "wb") as f:
                pose.write(f)
        videos[video_id] = {
            "pose_path": str(pose_path),
            "kind": kind,
            "fps": fps,
            "total_frames": total_frames,
            "signs": [],
            "sentences": [],
        }

    cache = {"videos": videos}
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate null segmentation pose files")
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUTPUT), help="output annotations cache path")
    parser.add_argument("--poses_dir", type=str, default=str(_DEFAULT_POSES_DIR), help="directory for generated pose files")
    parser.add_argument("--max_frames", type=int, default=1024, help="maximum frame count for the null samples")
    parser.add_argument("--sample_count", type=int, default=10, help="number of null samples from short to max_frames")
    parser.add_argument("--fps", type=float, default=30.0, help="fps for generated pose files, minimum 24")
    parser.add_argument("--static_source_pose", type=str, default=None, help="pose file to duplicate for static clips")
    parser.add_argument("--static_source_frame", type=int, default=0, help="frame index to duplicate from static source pose")
    parser.add_argument("--force", action="store_true", default=False, help="overwrite existing pose files")
    args = parser.parse_args()

    cache = build_cache(
        output_path=Path(args.output),
        poses_dir=Path(args.poses_dir),
        max_frames=args.max_frames,
        sample_count=args.sample_count,
        fps=args.fps,
        force=args.force,
        static_source_pose=Path(args.static_source_pose) if args.static_source_pose else None,
        static_source_frame=args.static_source_frame,
    )
    print(f"Wrote {len(cache['videos'])} null clips to {args.output}")


if __name__ == "__main__":
    main()
