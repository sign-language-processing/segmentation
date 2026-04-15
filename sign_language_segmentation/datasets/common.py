from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
import hashlib
from pathlib import Path
import random
from enum import StrEnum

import numpy as np
import torch
from pose_format import Pose
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sign_language_segmentation.utils.bio import BIO, create_bio, create_bio_from_times
from sign_language_segmentation.utils.pose import compute_velocity, preprocess_pose

# project-level cache directory — each dataset stores its annotations_cache.json under .cache/{dataset_name}/
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


# -- dataset registry ----------------------------------------------------------

DATASET_REGISTRY: dict[str, type[BaseSegmentationDataset]] = {}


def register_dataset(name: str, cls: type[BaseSegmentationDataset]) -> None:
    """register a dataset class under *name* so build_datasets can find it."""
    DATASET_REGISTRY[name] = cls


def ensure_datasets_registered() -> None:
    """lazily import known dataset packages to trigger registration."""
    if DATASET_REGISTRY:
        return
    # importing the modules registers the datasets
    import sign_language_segmentation.datasets.dgs  # noqa: F401
    import sign_language_segmentation.datasets.annotation_platform  # noqa: F401
    import sign_language_segmentation.datasets.signtube  # noqa: F401


def build_datasets(names: str, split: Split, args: Namespace, **augment_kwargs) -> Dataset:
    """build one or more datasets from a comma-separated *names* string.

    Each name must be registered via register_dataset(). Dataset-specific
    args are pulled from *args* inside each class's from_args classmethod.
    augment_kwargs (num_frames, velocity, fps_aug, ...) are forwarded as-is.
    """
    ensure_datasets_registered()

    dataset_names = sorted(DATASET_REGISTRY.keys()) if names == "all" else [n.strip() for n in names.split(",")]
    datasets: list[Dataset] = []
    for name in dataset_names:
        if name not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")
        cls = DATASET_REGISTRY[name]
        datasets.append(cls.from_args(split=split, args=args, **augment_kwargs))

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def get_dataloader(
    split: Split,
    dataset_names: str,
    args: Namespace,
    batch_size: int | None = None,
    num_frames: int | None = None,
    persistent_workers: bool = True,
    **augment_overrides,
) -> DataLoader:
    """build a DataLoader for one or more datasets.

    augment kwargs default to values from *args* but can be overridden
    via **augment_overrides (e.g. fps_aug=False for evaluation).
    """
    augment_kwargs = dict(
        num_frames=num_frames if num_frames is not None else getattr(args, "num_frames", 999999),
        velocity=getattr(args, "velocity", True),
        fps_aug=getattr(args, "fps_aug", True),
        frame_dropout=getattr(args, "frame_dropout", 0.0),
        body_part_dropout=getattr(args, "body_part_dropout", 0.0) if split == Split.TRAIN else 0.0,
    )
    augment_kwargs.update(augment_overrides)
    dataset = build_datasets(names=dataset_names, split=split, args=args, **augment_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size or getattr(args, "batch_size", 1),
        shuffle=(split == Split.TRAIN),
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=persistent_workers,
        prefetch_factor=4 if persistent_workers else 2,
        pin_memory=True,
    )


def md5sum(file_path: str) -> str:
    """compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def split_bucket(video_id: str, seed: int) -> int:
    """deterministic hash-based split assignment. returns 0-999."""
    h = hashlib.sha256(f"{video_id}_{seed}".encode()).hexdigest()
    return int(h, 16) % 1000


def assign_split(
    video_id: str,
    split_seed: int,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Split:
    """assign a split to a video ID based on deterministic hashing."""
    train_threshold = int((1.0 - dev_ratio - test_ratio) * 1000)
    dev_threshold = int((1.0 - test_ratio) * 1000)
    bucket = split_bucket(video_id=video_id, seed=split_seed)
    if bucket < train_threshold:
        return Split.TRAIN
    if bucket < dev_threshold:
        return Split.DEV
    return Split.TEST


class BaseSegmentationDataset(Dataset, ABC):
    """base class for segmentation datasets.

    Subclasses must populate self.items in __init__ — a list of dicts with keys:
    id, pose_path, fps, total_frames, glosses (sign spans), sentences (phrase spans).
    All span times must be in milliseconds.

    Split tracking via _init_split_tracking / _track_and_filter / get_split_manifest.
    Default get_split_manifest uses split_seed (for hash-based splits); subclasses
    with fixed splits (e.g. DGS) can override get_split_manifest.
    """

    # dataset name used in manifests — subclasses should set this
    dataset_name: str = "base"

    items: list[dict]
    split: Split
    num_frames: int
    velocity: bool
    fps_aug: bool
    frame_dropout: float
    body_part_dropout: float
    split_seed: int
    _all_split_ids: dict[str, list[str]]

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...

    def _init_split_tracking(self) -> None:
        """initialize split tracking — call at the start of subclass __init__."""
        self._all_split_ids = {
            Split.TRAIN: [],
            Split.DEV: [],
            Split.TEST: [],
        }

    def _track_and_filter(self, video_id: str, video_split: Split, item: dict) -> None:
        """track a video's split assignment and append to self.items if it matches."""
        self._all_split_ids[video_split].append(video_id)
        if video_split == self.split:
            self.items.append(item)

    @classmethod
    @abstractmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs) -> BaseSegmentationDataset:
        """construct from a parsed argument namespace + shared augment kwargs."""
        ...

    def get_split_manifest(self) -> dict:
        """return manifest of video IDs per split for reproducibility tracking."""
        return {
            "dataset": self.dataset_name,
            "split_seed": self.split_seed,
            "splits": {s.value: sorted(ids) for s, ids in self._all_split_ids.items()},
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return load_and_augment(
            pose_path=item["pose_path"],
            fps=item["fps"],
            total_frames=item["total_frames"],
            signs=item["glosses"],
            sentences=item["sentences"],
            split=self.split,
            num_frames=self.num_frames,
            velocity=self.velocity,
            fps_aug=self.fps_aug,
            frame_dropout=self.frame_dropout,
            body_part_dropout=self.body_part_dropout,
        )


def load_and_augment(
    pose_path: str,
    fps: float,
    total_frames: int,
    signs: list[dict[str, float]],
    sentences: list[dict[str, float]],
    split: Split,
    num_frames: int,
    velocity: bool,
    fps_aug: bool,
    frame_dropout: float,
    body_part_dropout: float,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """shared pose loading, augmentation, and BIO label generation.

    signs/sentences: lists of {start, end} spans in milliseconds.
    Returns {pose, timestamps, bio: {sign, sentence}}.
    """
    # determine effective target fps (stochastic during training if fps_aug)
    effective_target_fps: float | None = None
    tempo_fps: float | None = None
    if fps_aug and split == Split.TRAIN:
        if random.random() < 0.05:
            # tempo stretch: rescale RoPE positions only, no frame removal
            tempo_fps = random.choice([24.0, 30.0, 60.0])
        else:
            effective_target_fps = random.uniform(25, 50)

    # load proportionally more native frames when downsampling
    load_num_frames = num_frames
    if effective_target_fps and fps > effective_target_fps * 1.05:
        load_num_frames = min(total_frames, round(num_frames * fps / effective_target_fps))

    if split == Split.TRAIN and total_frames > load_num_frames:
        start = random.randint(0, total_frames - load_num_frames)
        end = start + load_num_frames
    else:
        start = 0
        end = total_frames

    with open(pose_path, "rb") as f:
        pose = Pose.read(f, start_frame=start, end_frame=end)

    pose = preprocess_pose(pose)
    actual_frames = len(pose.body.data)
    pose_data = pose.body.data.filled(0)[:, 0, :, :3].astype(np.float32)

    # downsample to target fps if needed
    effective_fps = fps
    if effective_target_fps and fps > effective_target_fps * 1.05:
        target_len = max(1, round(actual_frames * effective_target_fps / fps))
        src_indices = (
            np.round(np.arange(target_len) * (actual_frames - 1) / max(1, target_len - 1))
            .astype(int)
            .clip(0, actual_frames - 1)
        )
        pose_data = pose_data[src_indices]
        frame_times = src_indices.astype(np.float32) / fps
        frame_times_ms = frame_times * 1000
        effective_fps = effective_target_fps
        actual_frames = target_len
    else:
        if fps_aug:
            rope_fps = tempo_fps if tempo_fps is not None else fps
            frame_times = np.arange(actual_frames, dtype=np.float32) / rope_fps
            frame_times_ms = frame_times * 1000
        else:
            frame_times = np.arange(actual_frames, dtype=np.float32) / fps
            frame_times_ms = frame_times * 1000

    # body part dropout: zero entire hands independently (training only)
    if body_part_dropout > 0.0 and split == Split.TRAIN:
        if random.random() < body_part_dropout:
            pose_data[:, 8:29, :] = 0
        if random.random() < body_part_dropout:
            pose_data[:, 29:50, :] = 0

    # random frame dropout: drop 0-frame_dropout% of middle frames (training only)
    if frame_dropout > 0.0 and split == Split.TRAIN and actual_frames > 2:
        drop_rate = random.uniform(0.0, frame_dropout)
        n_drop = int((actual_frames - 2) * drop_rate)
        if n_drop > 0:
            drop_indices = set(random.sample(range(1, actual_frames - 1), n_drop))
            keep_mask = np.array([i not in drop_indices for i in range(actual_frames)])
            pose_data = pose_data[keep_mask]
            frame_times = frame_times[keep_mask]
            if fps_aug:
                frame_times_ms = frame_times_ms[keep_mask]
            actual_frames = len(pose_data)

    if velocity:
        vel = compute_velocity(pose_data, frame_times)
        pose_data = np.concatenate([pose_data, vel], axis=-1)

    # clip annotation spans to the current time window
    start_ms = start / fps * 1000
    end_ms = end / fps * 1000

    window_signs = [
        {"start": max(0, g["start"] - start_ms), "end": min(end_ms - start_ms, g["end"] - start_ms)}
        for g in signs
        if g["end"] > start_ms and g["start"] < end_ms
    ]
    window_sentences = [
        {"start": max(0, s["start"] - start_ms), "end": min(end_ms - start_ms, s["end"] - start_ms)}
        for s in sentences
        if s["end"] > start_ms and s["start"] < end_ms
    ]

    if fps_aug:
        sign_bio = create_bio_from_times(window_signs, frame_times_ms)
        sentence_bio = create_bio_from_times(window_sentences, frame_times_ms)
    else:
        sign_bio = create_bio(window_signs, actual_frames, effective_fps)
        sentence_bio = create_bio(window_sentences, actual_frames, effective_fps)

    return {
        "pose": torch.from_numpy(pose_data),
        "timestamps": torch.from_numpy(frame_times),
        "bio": {
            "sign": torch.from_numpy(sign_bio).long(),
            "sentence": torch.from_numpy(sentence_bio).long(),
        },
    }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """pad sequences to same length in batch."""
    max_len = max(item["pose"].shape[0] for item in batch)
    pose_dim = batch[0]["pose"].shape[1:]

    padded_poses: list[torch.Tensor] = []
    padded_sign_bio: list[torch.Tensor] = []
    padded_sentence_bio: list[torch.Tensor] = []
    padded_timestamps: list[torch.Tensor] = []
    has_timestamps = "timestamps" in batch[0]

    for item in batch:
        seq_len = item["pose"].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            padded_poses.append(torch.cat([item["pose"], torch.zeros(pad_len, *pose_dim)]))
            padded_sign_bio.append(
                torch.cat([item["bio"]["sign"], torch.full((pad_len,), BIO["UNK"], dtype=torch.long)])
            )
            padded_sentence_bio.append(
                torch.cat([item["bio"]["sentence"], torch.full((pad_len,), BIO["UNK"], dtype=torch.long)])
            )
            if has_timestamps:
                padded_timestamps.append(torch.cat([item["timestamps"], torch.zeros(pad_len)]))
        else:
            padded_poses.append(item["pose"])
            padded_sign_bio.append(item["bio"]["sign"])
            padded_sentence_bio.append(item["bio"]["sentence"])
            if has_timestamps:
                padded_timestamps.append(item["timestamps"])

    lengths = torch.tensor([item["pose"].shape[0] for item in batch], dtype=torch.long)
    result: dict[str, torch.Tensor] = {
        "pose": torch.stack(padded_poses),
        "lengths": lengths,
        "bio": {
            "sign": torch.stack(padded_sign_bio),
            "sentence": torch.stack(padded_sentence_bio),
        },
    }
    if has_timestamps:
        result["timestamps"] = torch.stack(padded_timestamps)
    return result
