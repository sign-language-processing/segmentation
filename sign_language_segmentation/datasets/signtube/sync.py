"""sync annotations from the SignTube PostgreSQL database.

Queries the captions table for sign/sentence annotations, resolves pose files,
and writes a local JSON cache that SignTubeSegmentationDataset can read.

Usage:
    python -m sign_language_segmentation.datasets.signtube.sync \
        --output .cache/signtube/annotations_cache.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
from collections import Counter
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody
from tqdm import tqdm

from sign_language_segmentation.datasets.common import CACHE_DIR

load_dotenv()

_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_CACHE_DIR = CACHE_DIR / "signtube"
_DEFAULT_CACHE = _DATASET_CACHE_DIR / "annotations_cache.json"
_QUERY_PATH = _PACKAGE_DIR / "captions.sql"

# NAS-mounted pose files, keyed by md5 of the source video (see _NAS_VIDEO_LIST)
_NAS_POSES_DIR = Path("/mnt/nas/GCS/sign-mediapipe-holistic-poses")
_NAS_VIDEO_LIST = Path("/mnt/nas/transformations/videos/video_list.csv")


def _get_connection():
    for var in ("DB_NAME", "DB_USER", "DB_PASS", "DB_HOST"):
        if os.environ.get(var) is None:
            raise RuntimeError(f"Missing {var} environment variable")
    return psycopg.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        host=os.environ["DB_HOST"],
    )


def _fetch_annotations() -> dict[str, list[dict]]:
    """query DB and group captions by video ID."""
    query = _QUERY_PATH.read_text()

    print("Fetching annotations from SignTube DB...")
    try:
        with _get_connection() as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(query)
                rows = cur.fetchall()
    except psycopg.Error as e:
        raise RuntimeError(f"Failed to fetch annotations from SignTube DB: {e}") from e

    videos: dict[str, list[dict]] = {}
    for row in rows:
        videos.setdefault(row["videoId"], []).append(row)

    print(f"Fetched {len(rows)} annotations across {len(videos)} videos")
    return videos


def _is_sign_annotation(row: dict) -> bool:
    return row["language"] in ("Sgnw", "hns")


def _build_signtube_md5_lookup() -> dict[str, str]:
    """map signtube video IDs to md5 hashes via the NAS video list."""
    lookup: dict[str, str] = {}
    with open(_NAS_VIDEO_LIST, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if not name.startswith("sign-tube/"):
                continue
            lookup[Path(name).stem] = row["md5Hash"]
    return lookup


def _build_cache(videos: dict[str, list[dict]]) -> dict:
    """build annotations cache from DB rows + pose metadata (poses resolved on NAS)."""
    cache: dict[str, dict] = {}
    skipped: Counter[str] = Counter()

    print(f"Building signtube md5 lookup from {_NAS_VIDEO_LIST}...")
    md5_lookup = _build_signtube_md5_lookup()
    print(f"Loaded {len(md5_lookup)} signtube md5 entries")

    for video_id, video_annotations in tqdm(videos.items()):
        md5 = md5_lookup.get(video_id)
        if md5 is None:
            skipped["no_md5"] += 1
            continue

        pose_path = _NAS_POSES_DIR / f"{md5}.pose"
        if not pose_path.exists():
            skipped["pose_missing"] += 1
            continue

        try:
            meta = Pose.read(pose_path.read_bytes(), pose_body=EmptyPoseBody)
        except (struct.error, EOFError, ValueError) as e:
            # pose_format fails modes: struct.error on binary corruption, EOFError on
            # truncation, ValueError on malformed headers/dims — skip those and let
            # real bugs (AttributeError, ImportError, etc.) surface instead of a bare Exception.
            print(f"skipping {pose_path}: {type(e).__name__}: {e}")
            skipped["pose_unreadable"] += 1
            continue

        fps = float(meta.body.fps)
        total_frames = len(meta.body.data)

        if total_frames < 2:
            skipped["frames_lt_2"] += 1
            continue

        sign_annotations = [r for r in video_annotations if _is_sign_annotation(r)]
        sentence_annotations = [r for r in video_annotations if not _is_sign_annotation(r)]

        signs = [{"start": float(r["start"]), "end": float(r["end"])} for r in sign_annotations]
        sentences = [{"start": float(r["start"]), "end": float(r["end"])} for r in sentence_annotations]

        cache[video_id] = {
            "pose_path": str(pose_path),
            "fps": fps,
            "total_frames": total_frames,
            "signs": signs,
            "sentences": sentences,
        }

    total_skipped = sum(skipped.values())
    print(f"Built cache: {len(cache)} videos ({total_skipped} skipped)")
    for reason, count in skipped.most_common():
        print(f"  {count} skipped: {reason}")
    return {"videos": cache}


def main():
    parser = argparse.ArgumentParser(description="Sync SignTube annotations to local cache")
    parser.add_argument("--output", type=str, default=str(_DEFAULT_CACHE), help="output annotations cache path")
    parser.add_argument("--force", action="store_true", default=False, help="overwrite existing cache file")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Cache already exists at {args.output}, skipping (use --force to overwrite)")
        return

    videos = _fetch_annotations()
    cache = _build_cache(videos=videos)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Wrote cache to {args.output}")


if __name__ == "__main__":
    main()
