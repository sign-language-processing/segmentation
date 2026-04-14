from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import os

from sign_language_segmentation.datasets.common import BaseSegmentationDataset, Split


class AnnotationPlatformSegmentationDataset(BaseSegmentationDataset):
    """dataset backed by annotations from the annotation platform (Convex DB).

    Reads a local annotations_cache.json produced by sync.py, resolves pose files,
    and produces the same output format as DGSSegmentationDataset.
    """

    def __init__(
        self,
        annotations_path: str,
        poses_dir: str,
        split: Split = Split.TRAIN,
        num_frames: int = 1024,
        velocity: bool = True,
        fps_aug: bool = True,
        frame_dropout: float = 0.15,
        body_part_dropout: float = 0.1,
        split_seed: int = 42,
        dev_ratio: float = 0.1,
        test_ratio: float = 0.1,
        quality_percentile: float = 1.0,
    ):
        self.poses_dir = poses_dir
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout
        self.split_seed = split_seed
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.quality_percentile = quality_percentile

        with open(annotations_path) as f:
            cache = json.load(f)

        # build list of valid videos with sign annotations
        all_videos: list[dict] = []
        for video_id, video_data in cache.get("videos", {}).items():
            if not video_data.get("signs"):
                continue

            if video_data.get("total_frames", 0) < 2:
                continue

            pose_hash = video_data.get("pose_hash")
            if not pose_hash:
                continue

            pose_path = os.path.join(poses_dir, f"{pose_hash}.pose")
            if not os.path.exists(pose_path):
                continue

            all_videos.append({
                "id": video_id,
                "pose_path": pose_path,
                "fps": video_data["fps"],
                "total_frames": video_data["total_frames"],
                "glosses": video_data["signs"],
                "sentences": video_data.get("phrases", []),
                "quality_score": video_data.get("quality_score", 0.0),
            })

        # quality filtering: keep top X%
        if quality_percentile < 1.0 and all_videos:
            all_videos.sort(key=lambda v: v["quality_score"], reverse=True)
            keep_count = max(1, int(len(all_videos) * quality_percentile))
            all_videos = all_videos[:keep_count]

        # deterministic split by video ID
        train_threshold = int((1.0 - dev_ratio - test_ratio) * 1000)
        dev_threshold = int((1.0 - test_ratio) * 1000)

        self.items: list[dict] = []
        self._all_split_ids: dict[str, list[str]] = {
            Split.TRAIN: [],
            Split.DEV: [],
            Split.TEST: [],
        }

        for video in all_videos:
            bucket = _split_bucket(video_id=video["id"], seed=split_seed)
            if bucket < train_threshold:
                video_split = Split.TRAIN
            elif bucket < dev_threshold:
                video_split = Split.DEV
            else:
                video_split = Split.TEST

            self._all_split_ids[video_split].append(video["id"])
            if video_split == split:
                self.items.append(video)

        print(
            f"AnnotationPlatformSegmentationDataset({split}): "
            f"{len(self.items)} videos "
            f"(train={len(self._all_split_ids[Split.TRAIN])}, "
            f"dev={len(self._all_split_ids[Split.DEV])}, "
            f"test={len(self._all_split_ids[Split.TEST])})"
        )

    @classmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs) -> AnnotationPlatformSegmentationDataset:
        if not getattr(args, "annotations_path", None):
            raise ValueError("--annotations_path required for platform dataset")
        return cls(
            annotations_path=args.annotations_path,
            poses_dir=args.poses,
            split=split,
            quality_percentile=getattr(args, "quality_percentile", 1.0),
            **augment_kwargs,
        )

    def get_split_manifest(self) -> dict:
        """return manifest of video IDs per split for reproducibility tracking."""
        return {
            "dataset": "annotation_platform",
            "split_seed": self.split_seed,
            "quality_percentile": self.quality_percentile,
            "splits": {
                split.value: sorted(ids) for split, ids in self._all_split_ids.items()
            },
        }

def _split_bucket(video_id: str, seed: int) -> int:
    """deterministic hash-based split assignment. returns 0-999."""
    h = hashlib.sha256(f"{video_id}_{seed}".encode()).hexdigest()
    return int(h, 16) % 1000
