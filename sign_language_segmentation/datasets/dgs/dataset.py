from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody

# shim: sign_language_datasets imports get_dl_dirname which was renamed in tfds >=4.9
import tensorflow_datasets.core.download.resource as _tfds_resource
if not hasattr(_tfds_resource, "get_dl_dirname"):
    _tfds_resource.get_dl_dirname = _tfds_resource.get_dl_fname

from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences

from sign_language_segmentation.datasets.common import CACHE_DIR, BaseSegmentationDataset, Split, md5sum

EXCLUDED_IDS: set[str] = {"1289910", "1245887", "1289868", "1246064", "1584617"}


def is_joke(corpus_dir: Path, doc_id: str) -> bool:
    cmdi_path = corpus_dir / "videos" / doc_id / "data.cmdi"
    if not cmdi_path.exists():
        return False
    return "<cmdp:Task>Joke</cmdp:Task>" in cmdi_path.read_text()


class DGSSegmentationDataset(BaseSegmentationDataset):
    """reads DGS corpus poses on the fly with efficient partial reading."""

    dataset_name = "dgs"

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
        self.corpus_dir = Path(corpus_dir)
        self.poses_dir = Path(poses_dir)
        self.split = split
        self.num_frames = num_frames
        self.velocity = velocity
        self.target_fps = target_fps
        self.fps_aug = fps_aug
        self.frame_dropout = frame_dropout
        self.body_part_dropout = body_part_dropout

        if splits_path is None:
            splits_path = str(Path(__file__).parent / "splits.json")
        self.splits_path = splits_path
        splits = json.loads(Path(splits_path).read_text())
        dev_ids = set(splits["dev"])
        test_ids = set(splits["test"])

        self._init_split_tracking()
        self.items = []

        if cache_path is None:
            dgs_cache_dir = CACHE_DIR / "dgs"
            dgs_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(dgs_cache_dir / "segmentation_cache.json")

        cache_file = Path(cache_path) if cache_path else self.corpus_dir / ".segmentation_cache.json"
        cache = self._load_cache(cache_file)
        cache_dirty = False

        videos_dir = self.corpus_dir / "videos"
        doc_ids = sorted(
            d.name for d in videos_dir.iterdir() if d.is_dir()
        )

        for doc_id in doc_ids:
            if doc_id in EXCLUDED_IDS or is_joke(self.corpus_dir, doc_id):
                continue

            # fixed split from splits.json (research-standard split)
            if doc_id in dev_ids:
                doc_split = Split.DEV
            elif doc_id in test_ids:
                doc_split = Split.TEST
            else:
                doc_split = Split.TRAIN

            eaf_path = videos_dir / doc_id / "data.eaf"
            if not eaf_path.exists() or eaf_path.stat().st_size < 10000:
                continue

            sentences = list(get_elan_sentences(str(eaf_path)))
            if not sentences:
                continue

            for person in ("a", "b"):
                video_path = videos_dir / doc_id / f"video_{person}.mp4"
                if not video_path.exists():
                    continue

                cache_key = f"{doc_id}_{person}"

                if cache_key in cache:
                    video_hash = cache[cache_key]["hash"]
                    fps = cache[cache_key]["fps"]
                    total_frames = cache[cache_key]["total_frames"]
                else:
                    video_hash = md5sum(str(video_path))
                    pose_path = self.poses_dir / f"{video_hash}.pose"
                    if not pose_path.exists():
                        continue
                    with open(pose_path, "rb") as f:
                        meta_pose = Pose.read(f, pose_body=EmptyPoseBody)
                    fps = meta_pose.body.fps
                    total_frames = len(meta_pose.body.data)
                    cache[cache_key] = {"hash": video_hash, "fps": fps, "total_frames": total_frames}
                    cache_dirty = True

                pose_path = self.poses_dir / f"{video_hash}.pose"
                if not pose_path.exists():
                    continue

                person_sentences = [
                    s for s in sentences
                    if s["participant"].lower() == person and len(s["glosses"]) > 0
                ]
                if not person_sentences:
                    continue

                all_glosses = [g for s in person_sentences for g in s["glosses"]]
                sentence_spans = [{"start": s["start"], "end": s["end"]} for s in person_sentences]

                self._track_and_filter(cache_key, doc_split, {
                    "id": cache_key,
                    "pose_path": str(pose_path),
                    "fps": fps,
                    "total_frames": total_frames,
                    "glosses": all_glosses,
                    "sentences": sentence_spans,
                })

        if cache_dirty:
            self._save_cache(cache_file, cache)

        print(
            f"DGSSegmentationDataset({split}): "
            f"{len(self.items)} videos "
            f"(train={len(self._all_split_ids[Split.TRAIN])}, "
            f"dev={len(self._all_split_ids[Split.DEV])}, "
            f"test={len(self._all_split_ids[Split.TEST])})"
        )

    def get_split_manifest(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "splits_path": self.splits_path,
            "splits": {
                s.value: sorted(ids) for s, ids in self._all_split_ids.items()
            },
        }

    def _load_cache(self, path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_cache(self, path: Path, cache: dict) -> None:
        try:
            path.write_text(json.dumps(cache))
        except OSError:
            pass

    @classmethod
    def from_args(cls, split: Split, args: Namespace, **augment_kwargs) -> DGSSegmentationDataset:
        return cls(
            corpus_dir=args.corpus,
            poses_dir=args.poses,
            split=split,
            target_fps=getattr(args, "target_fps", None),
            **augment_kwargs,
        )
