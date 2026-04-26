from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from sign_language_segmentation.datasets.common import CACHE_DIR, BaseSegmentationDataset, Split, assign_split


class SignTubeSegmentationDataset(BaseSegmentationDataset):
    """dataset backed by annotations from the SignTube PostgreSQL database.

    Reads a local annotations_cache.json produced by sync.py, resolves pose files,
    and produces the same output format as other segmentation datasets.
    Splits are assigned deterministically by hashing the video ID.
    """

    dataset_name = "signtube"

    def __init__(
        self,
        annotations_path: str,
        split: Split = Split.TRAIN,
        num_frames: int = 1024,
        velocity: bool = True,
        fps_aug: bool = True,
        frame_dropout: float = 0.15,
        body_part_dropout: float = 0.1,
        split_seed: int = 42,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout
        self.split_seed = split_seed
        poses_dir = CACHE_DIR / self.dataset_name / "poses"

        self._init_split_tracking()
        self.items = []

        with open(annotations_path) as f:
            cache = json.load(f)

        if "videos" not in cache:
            raise ValueError(f"Corrupted annotations cache at {annotations_path}: missing 'videos' key")

        for video_id, video_data in cache["videos"].items():
            pose_path = Path(video_data.get("pose_path") or poses_dir / f"{video_id}.pose")
            if not pose_path.exists():
                fallback_pose_path = poses_dir / f"{video_id}.pose"
                if fallback_pose_path.exists():
                    pose_path = fallback_pose_path
            if not pose_path.exists():
                continue

            if video_data.get("total_frames", 0) < 2:
                continue

            video_split = assign_split(video_id, split_seed=split_seed, dev_ratio=dev_ratio, test_ratio=test_ratio)
            self._track_and_filter(
                video_id,
                video_split,
                {
                    "id": video_id,
                    "pose_path": pose_path,
                    "fps": video_data["fps"],
                    "total_frames": video_data["total_frames"],
                    "glosses": video_data["signs"],
                    "sentences": video_data.get("sentences", []),
                },
            )

        print(
            f"SignTubeSegmentationDataset({split}): "
            f"{len(self.items)} videos "
            f"(train={len(self._all_split_ids[Split.TRAIN])}, "
            f"dev={len(self._all_split_ids[Split.DEV])}, "
            f"test={len(self._all_split_ids[Split.TEST])})"
        )

    @classmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs) -> SignTubeSegmentationDataset:
        signtube_path = CACHE_DIR / cls.dataset_name / "annotations_cache.json"
        if not signtube_path.exists():
            raise FileNotFoundError(f"annotations cache not found at {signtube_path} — run the sync script first")
        return cls(
            annotations_path=str(signtube_path),
            split=split,
            **augment_kwargs,
        )
