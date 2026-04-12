import json
import os

from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences

from sign_language_segmentation.datasets.common import BaseSegmentationDataset, Split, md5sum

EXCLUDED_IDS: set[str] = {"1289910", "1245887", "1289868", "1246064", "1584617"}


def is_joke(corpus_dir: str, doc_id: str) -> bool:
    cmdi_path = os.path.join(corpus_dir, "videos", doc_id, "data.cmdi")
    if not os.path.exists(cmdi_path):
        return False
    with open(cmdi_path, "r") as f:
        return "<cmdp:Task>Joke</cmdp:Task>" in f.read()


class DGSSegmentationDataset(BaseSegmentationDataset):
    """reads DGS corpus poses on the fly with efficient partial reading."""

    def __init__(
        self,
        corpus_dir: str,
        poses_dir: str,
        split: Split = Split.TRAIN,
        num_frames: int = 1024,
        velocity: bool = True,
        target_fps: float | None = None,
        fps_aug: bool = True,
        frame_dropout: float = 0.15,
        body_part_dropout: float = 0.1,
        splits_path: str | None = None,
        cache_path: str | None = None,
    ):
        self.corpus_dir = corpus_dir
        self.poses_dir = poses_dir
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
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

        self.items: list[dict] = []
        cache = self._load_cache(cache_path)
        cache_dirty = False

        videos_dir = os.path.join(corpus_dir, "videos")
        doc_ids = sorted(
            d for d in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir, d))
        )

        for doc_id in doc_ids:
            if doc_id in EXCLUDED_IDS or is_joke(corpus_dir, doc_id):
                continue

            if split == Split.DEV and doc_id not in dev_ids:
                continue
            if split == Split.TEST and doc_id not in test_ids:
                continue
            if split == Split.TRAIN and (doc_id in dev_ids or doc_id in test_ids):
                continue

            eaf_path = os.path.join(videos_dir, doc_id, "data.eaf")
            if not os.path.exists(eaf_path) or os.path.getsize(eaf_path) < 10000:
                continue

            sentences = list(get_elan_sentences(eaf_path))
            if not sentences:
                continue

            for person in ("a", "b"):
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

                person_sentences = [
                    s for s in sentences
                    if s["participant"].lower() == person and len(s["glosses"]) > 0
                ]
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

    def _save_cache(self, path: str, cache: dict) -> None:
        try:
            with open(path, "w") as f:
                json.dump(cache, f)
        except OSError:
            pass

    def get_split_manifest(self) -> dict:
        """return manifest of video IDs per split for reproducibility tracking."""
        return {
            "dataset": "dgs",
            "split": self.split.value,
            "video_ids": [item["id"] for item in self.items],
        }

