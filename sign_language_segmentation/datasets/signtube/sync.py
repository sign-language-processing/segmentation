"""sync annotations from the SignTube PostgreSQL database.

Queries the captions table for sign/sentence annotations, resolves pose files,
and writes a local JSON cache that SignTubeSegmentationDataset can read.

Usage:
    python -m sign_language_segmentation.datasets.signtube.sync \
        --output .cache/signtube/annotations_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gcsfs
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
    with open(_QUERY_PATH) as f:
        query = f.read()

    print("Fetching annotations from SignTube DB...")
    conn = _get_connection()
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    conn.close()

    videos: dict[str, list[dict]] = {}
    for row in rows:
        videos.setdefault(row["videoId"], []).append(row)

    print(f"Fetched {len(rows)} annotations across {len(videos)} videos")
    return videos


def _is_sign_annotation(row: dict) -> bool:
    if row["language"] in ("Sgnw", "hns"):
        return True
    if row["language"] == "gloss" and " " not in row["text"]:
        return True
    return False


def _build_cache(videos: dict[str, list[dict]]) -> dict:
    """build annotations cache from DB rows + pose metadata."""
    cache: dict[str, dict] = {}
    skipped = 0

    poses_dir_path = _DATASET_CACHE_DIR / "poses"
    poses_dir_path.mkdir(parents=True, exist_ok=True)

    for video_id, video_annotations in tqdm(videos.items()):
        pose_path = poses_dir_path / f"{video_id}.pose"

        if not pose_path.exists():
            # download from GCS bucket if not available locally
            bucket_url = f"gs://sign-mt-poses/{video_id}.pose"
            try:
                with gcsfs.GCSFileSystem().open(bucket_url, "rb") as f:
                    pose_path.write_bytes(f.read())
            except FileNotFoundError:
                skipped += 1
                continue

        meta = Pose.read(pose_path.read_bytes(), pose_body=EmptyPoseBody)

        fps = float(meta.body.fps)
        total_frames = len(meta.body.data)

        if total_frames < 2:
            skipped += 1
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

    print(f"Built cache: {len(cache)} videos ({skipped} skipped)")
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
