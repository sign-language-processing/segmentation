import hashlib
import json
import math
import os
import random
from typing import List, Literal, Optional

import numpy as np
import torch
from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences
from torch.utils.data import Dataset

from sign_language_segmentation.data.utils import preprocess_pose, BIO

Split = Literal["train", "dev", "test"]

EXCLUDED_IDS = {"1289910", "1245887", "1289868", "1246064", "1584617"}


def md5sum(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_joke(corpus_dir: str, doc_id: str) -> bool:
    cmdi_path = os.path.join(corpus_dir, "videos", doc_id, "data.cmdi")
    if not os.path.exists(cmdi_path):
        return False
    with open(cmdi_path, "r") as f:
        return "<cmdp:Task>Joke</cmdp:Task>" in f.read()


def create_bio(annotations, num_frames, fps):
    bio = np.full((num_frames,), fill_value=BIO["O"], dtype=np.uint8)
    for a in annotations:
        start_frame = min(math.floor(a["start"] * fps / 1000), num_frames - 1)
        end_frame = min(math.ceil(a["end"] * fps / 1000), num_frames - 1)
        bio[start_frame] = BIO["B"]
        bio[start_frame + 1:end_frame + 1] = BIO["I"]
    return bio


def create_bio_from_times(annotations: list, frame_times_ms: np.ndarray) -> np.ndarray:
    """BIO labels using actual frame timestamps in ms — handles non-uniform frame spacing."""
    num_frames = len(frame_times_ms)
    bio = np.full(num_frames, fill_value=BIO["O"], dtype=np.uint8)
    for a in annotations:
        start_idx = int(np.searchsorted(frame_times_ms, a["start"], side="left"))
        end_idx = int(np.searchsorted(frame_times_ms, a["end"], side="right")) - 1
        start_idx = min(start_idx, num_frames - 1)
        end_idx = min(end_idx, num_frames - 1)
        if start_idx <= end_idx:
            bio[start_idx] = BIO["B"]
            if start_idx + 1 <= end_idx:
                bio[start_idx + 1:end_idx + 1] = BIO["I"]
    return bio


class DGSSegmentationDataset(Dataset):
    """Reads DGS corpus poses on the fly with efficient partial reading."""

    def __init__(self,
                 corpus_dir: str,
                 poses_dir: str,
                 split: Split = "train",
                 num_frames: int = 1024,
                 normalize: bool = True,
                 pose_dims: int = 3,
                 velocity: bool = False,
                 no_face: bool = False,
                 target_fps: Optional[float] = None,
                 fps_aug: bool = False,
                 frame_dropout: float = 0.0,
                 body_part_dropout: float = 0.0,
                 splits_path: Optional[str] = None,
                 cache_path: Optional[str] = None):
        self.corpus_dir = corpus_dir
        self.poses_dir = poses_dir
        self.split = split
        self.num_frames = num_frames
        self.normalize = normalize
        self.pose_dims = pose_dims
        self.velocity = velocity
        self.no_face = no_face
        self.target_fps = target_fps
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout

        if splits_path is None:
            splits_path = os.path.join(os.path.dirname(__file__), "splits.json")
        with open(splits_path) as f:
            splits = json.load(f)

        dev_ids = set(splits["dev"])
        test_ids = set(splits["test"])

        if cache_path is None:
            cache_path = os.path.join(corpus_dir, ".segmentation_cache.json")

        self.items: List[dict] = []
        cache = self._load_cache(cache_path)
        cache_dirty = False

        videos_dir = os.path.join(corpus_dir, "videos")
        doc_ids = sorted(d for d in os.listdir(videos_dir)
                         if os.path.isdir(os.path.join(videos_dir, d)))

        for doc_id in doc_ids:
            if doc_id in EXCLUDED_IDS or is_joke(corpus_dir, doc_id):
                continue

            if split == "dev" and doc_id not in dev_ids:
                continue
            if split == "test" and doc_id not in test_ids:
                continue
            if split == "train" and (doc_id in dev_ids or doc_id in test_ids):
                continue

            eaf_path = os.path.join(videos_dir, doc_id, "data.eaf")
            if not os.path.exists(eaf_path) or os.path.getsize(eaf_path) < 10000:
                continue

            sentences = list(get_elan_sentences(eaf_path))
            if not sentences:
                continue

            for person in ["a", "b"]:
                video_path = os.path.join(videos_dir, doc_id, f"video_{person}.mp4")
                if not os.path.exists(video_path):
                    continue

                cache_key = f"{doc_id}_{person}"

                if cache_key in cache:
                    video_hash = cache[cache_key]["hash"]
                    fps = cache[cache_key]["fps"]
                    total_frames = cache[cache_key]["total_frames"]
                else:
                    video_hash = md5sum(video_path)
                    pose_path = os.path.join(poses_dir, f"{video_hash}.pose")
                    if not os.path.exists(pose_path):
                        continue
                    with open(pose_path, "rb") as f:
                        meta_pose = Pose.read(f, pose_body=EmptyPoseBody)
                    fps = meta_pose.body.fps
                    total_frames = len(meta_pose.body.data)
                    cache[cache_key] = {"hash": video_hash, "fps": fps, "total_frames": total_frames}
                    cache_dirty = True

                pose_path = os.path.join(poses_dir, f"{video_hash}.pose")
                if not os.path.exists(pose_path):
                    continue

                person_sentences = [s for s in sentences
                                    if s["participant"].lower() == person and len(s["glosses"]) > 0]
                if not person_sentences:
                    continue

                all_glosses = [g for s in person_sentences for g in s["glosses"]]
                sentence_spans = [{"start": s["start"], "end": s["end"]} for s in person_sentences]

                self.items.append({
                    "id": cache_key,
                    "pose_path": pose_path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "glosses": all_glosses,
                    "sentences": sentence_spans,
                })

        if cache_dirty:
            self._save_cache(cache_path, cache)

        print(f"DGSSegmentationDataset({split}): {len(self.items)} videos")

    def _load_cache(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_cache(self, path: str, cache: dict):
        try:
            with open(path, "w") as f:
                json.dump(cache, f)
        except OSError:
            pass

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        total_frames = item["total_frames"]
        fps = item["fps"]

        # Determine effective target fps (stochastic during training if fps_aug)
        effective_target_fps = self.target_fps
        tempo_fps = None  # RoPE-only tempo stretch, no frame removal
        if self.fps_aug and self.split == "train":
            if random.random() < 0.05:
                # Tempo stretch: rescale RoPE positions only, no frame removal.
                # 24/30 = stretch timeline (appears slower), 60 = compress (appears faster).
                tempo_fps = random.choice([24.0, 30.0, 60.0])
            else:
                effective_target_fps = random.uniform(25, 50)

        # Load proportionally more native frames when downsampling
        load_num_frames = self.num_frames
        if effective_target_fps and fps > effective_target_fps * 1.05:
            load_num_frames = min(total_frames, round(self.num_frames * fps / effective_target_fps))

        if self.split == "train" and total_frames > load_num_frames:
            start = random.randint(0, total_frames - load_num_frames)
            end = start + load_num_frames
        else:
            start = 0
            end = total_frames

        with open(item["pose_path"], "rb") as f:
            pose = Pose.read(f, start_frame=start, end_frame=end)

        pose = preprocess_pose(pose, normalize=self.normalize, no_face=self.no_face)
        num_frames = len(pose.body.data)
        pose_data = pose.body.data.filled(0)[:, 0, :, :self.pose_dims].astype(np.float32)

        # Downsample to target fps if needed
        effective_fps = fps
        if effective_target_fps and fps > effective_target_fps * 1.05:
            target_len = max(1, round(num_frames * effective_target_fps / fps))
            src_indices = np.round(
                np.arange(target_len) * (num_frames - 1) / max(1, target_len - 1)
            ).astype(int).clip(0, num_frames - 1)
            pose_data = pose_data[src_indices]
            # 50fps-equivalent positions: fps-invariant and correct RoPE scale
            REFERENCE_FPS = 50.0
            frame_times = src_indices.astype(np.float32) * (REFERENCE_FPS / fps)
            frame_times_ms = src_indices.astype(np.float32) / fps * 1000
            effective_fps = effective_target_fps
            num_frames = target_len
        else:
            if self.fps_aug:
                REFERENCE_FPS = 50.0
                # tempo_fps: RoPE-only stretch/compress (no frames removed).
                # 24/30fps → positions spread out (slower view); 60fps → compressed (faster view).
                rope_fps = tempo_fps if tempo_fps is not None else fps
                frame_times = np.arange(num_frames, dtype=np.float32) * (REFERENCE_FPS / rope_fps)
                # frame_times_ms must use rope_fps so BIO labels align with RoPE positions.
                # e.g. at rope_fps=24: frame i is at i/24s, so a 500ms sign starts at frame 12
                # — consistent with RoPE seeing positions at 24fps spacing.
                frame_times_ms = np.arange(num_frames, dtype=np.float32) / rope_fps * 1000
            else:
                # Frame-index (backward compat for non-fps_aug experiments)
                frame_times = np.arange(num_frames, dtype=np.float32)

        # Body part dropout: zero entire hands independently (training only).
        # With no_face=True: joints 0-7 = pose body, 8-28 = left hand, 29-49 = right hand.
        # Applied before velocity so that zeroed joints produce zero velocity too.
        if self.body_part_dropout > 0.0 and self.split == "train":
            if random.random() < self.body_part_dropout:
                pose_data[:, 8:29, :] = 0   # left hand (all dims)
            if random.random() < self.body_part_dropout:
                pose_data[:, 29:50, :] = 0  # right hand (all dims)

        # Random frame dropout: drop 0–frame_dropout% of middle frames (training only)
        if self.frame_dropout > 0.0 and self.split == "train" and num_frames > 2:
            drop_rate = random.uniform(0.0, self.frame_dropout)
            n_drop = int((num_frames - 2) * drop_rate)
            if n_drop > 0:
                drop_indices = set(random.sample(range(1, num_frames - 1), n_drop))
                keep_mask = np.array([i not in drop_indices for i in range(num_frames)])
                pose_data = pose_data[keep_mask]
                frame_times = frame_times[keep_mask]
                if self.fps_aug:
                    frame_times_ms = frame_times_ms[keep_mask]
                num_frames = len(pose_data)

        # Velocity: fps_aug uses 50fps-equivalent dt for fps-invariant magnitude;
        # non-fps_aug uses raw frame-to-frame diff.
        if self.velocity:
            if num_frames > 1:
                if self.fps_aug:
                    dt = np.diff(frame_times)  # (T-1,) in 50fps-equiv units
                    vel_inner = np.diff(pose_data, axis=0) / dt[:, None, None]
                    vel = np.concatenate([np.zeros_like(pose_data[:1]), vel_inner], axis=0)
                else:
                    vel = np.diff(pose_data, axis=0, prepend=pose_data[:1])
            else:
                vel = np.zeros_like(pose_data)
            pose_data = np.concatenate([pose_data, vel], axis=-1)

        start_ms = start / fps * 1000
        end_ms = end / fps * 1000

        window_glosses = [{"start": max(0, g["start"] - start_ms),
                           "end": min(end_ms - start_ms, g["end"] - start_ms)}
                          for g in item["glosses"]
                          if g["end"] > start_ms and g["start"] < end_ms]

        window_sentences = [{"start": max(0, s["start"] - start_ms),
                             "end": min(end_ms - start_ms, s["end"] - start_ms)}
                            for s in item["sentences"]
                            if s["end"] > start_ms and s["start"] < end_ms]

        # Use timestamp-based BIO creation (handles non-uniform spacing after dropout)
        if self.fps_aug:
            sign_bio = create_bio_from_times(window_glosses, frame_times_ms)
            sentence_bio = create_bio_from_times(window_sentences, frame_times_ms)
        else:
            sign_bio = create_bio(window_glosses, num_frames, effective_fps)
            sentence_bio = create_bio(window_sentences, num_frames, effective_fps)

        return {
            "pose": torch.from_numpy(pose_data),
            "timestamps": torch.from_numpy(frame_times),
            "bio": {
                "sign": torch.from_numpy(sign_bio).long(),
                "sentence": torch.from_numpy(sentence_bio).long(),
            },
        }


def collate_fn(batch):
    """Pad sequences to same length in batch."""
    max_len = max(item["pose"].shape[0] for item in batch)
    pose_dim = batch[0]["pose"].shape[1:]

    padded_poses = []
    padded_sign_bio = []
    padded_sentence_bio = []
    padded_timestamps = []
    has_timestamps = "timestamps" in batch[0]

    for item in batch:
        seq_len = item["pose"].shape[0]
        pad_len = max_len - seq_len

        if pad_len > 0:
            padded_poses.append(torch.cat([item["pose"],
                                           torch.zeros(pad_len, *pose_dim)]))
            padded_sign_bio.append(torch.cat([item["bio"]["sign"],
                                              torch.full((pad_len,), BIO["UNK"], dtype=torch.long)]))
            padded_sentence_bio.append(torch.cat([item["bio"]["sentence"],
                                                  torch.full((pad_len,), BIO["UNK"], dtype=torch.long)]))
            if has_timestamps:
                padded_timestamps.append(torch.cat([item["timestamps"],
                                                    torch.zeros(pad_len)]))
        else:
            padded_poses.append(item["pose"])
            padded_sign_bio.append(item["bio"]["sign"])
            padded_sentence_bio.append(item["bio"]["sentence"])
            if has_timestamps:
                padded_timestamps.append(item["timestamps"])

    lengths = torch.tensor([item["pose"].shape[0] for item in batch], dtype=torch.long)
    result = {
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
