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
from collections import Counter
from fractions import Fraction
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

from sign_language_segmentation.datasets.common import CACHE_DIR

load_dotenv()

_PACKAGE_DIR = Path(__file__).resolve().parent
_DATASET_CACHE_DIR = CACHE_DIR / "signtube"
_DEFAULT_CACHE = _DATASET_CACHE_DIR / "annotations_cache.json"
_QUERY_PATH = _PACKAGE_DIR / "captions.sql"

# nas-mounted pose files, keyed by md5 of the source video (see _NAS_VIDEO_LIST)
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


def _parse_video_metadata(row: dict[str, str]) -> dict[str, str | float | int] | None:
    md5 = row["md5Hash"]
    if not md5:
        return None

    duration_text = row["duration"]
    frame_rate_text = row["avg_frame_rate"]
    if not duration_text or not frame_rate_text or frame_rate_text == "0/0":
        return None

    try:
        duration = float(duration_text)
        fps = float(Fraction(frame_rate_text))
    except (ValueError, ZeroDivisionError):
        return None

    if duration < 0 or fps <= 0:
        return None

    return {"md5": md5, "fps": fps, "total_frames": round(duration * fps)}


def _build_signtube_video_lookup() -> dict[str, dict[str, str | float | int]]:
    """map signtube video IDs to NAS video metadata."""
    lookup: dict[str, dict[str, str | float | int]] = {}
    with open(_NAS_VIDEO_LIST, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            if not name.startswith("sign-tube/"):
                continue
            metadata = _parse_video_metadata(row=row)
            if metadata is None:
                continue
            lookup[Path(name).stem] = metadata
    return lookup


def _build_cache(videos: dict[str, list[dict]]) -> dict:
    """build annotations cache from DB rows + video metadata (poses resolved on NAS)."""
    cache: dict[str, dict] = {}
    skipped: Counter[str] = Counter()

    print(f"Building signtube video metadata lookup from {_NAS_VIDEO_LIST}...")
    video_lookup = _build_signtube_video_lookup()
    print(f"Loaded {len(video_lookup)} signtube video metadata entries")

    for video_id, video_annotations in tqdm(videos.items()):
        video_metadata = video_lookup.get(video_id)
        if video_metadata is None:
            skipped["no_video_metadata"] += 1
            continue

        pose_path = _NAS_POSES_DIR / f"{video_metadata['md5']}.pose"
        if not pose_path.exists():
            skipped["pose_missing"] += 1
            continue

        fps = float(video_metadata["fps"])
        total_frames = int(video_metadata["total_frames"])

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
