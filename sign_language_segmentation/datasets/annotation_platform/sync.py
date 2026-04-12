"""sync annotations from the Convex annotation platform and optionally score them.

Usage:
    # sync annotations
    python -m sign_language_segmentation.datasets.annotation_platform.sync \
        --convex_url https://amiable-labrador-506.convex.cloud \
        --project_ids m971s5xkknqsfhgrnjr2rdy83n80hmwx \
        --poses_dir /mnt/nas/GCS/sign-mediapipe-holistic-poses \
        --gcs_root /mnt/nas/GCS \
        --output annotations_cache.json

    # score annotations against current model
    python -m sign_language_segmentation.datasets.annotation_platform.sync \
        --score --model_path dist/2026/best.ckpt \
        --cache annotations_cache.json \
        --poses_dir /mnt/nas/GCS/sign-mediapipe-holistic-poses
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
import httpx
from pose_format import Pose
from pose_format.pose_body import EmptyPoseBody

from sign_language_segmentation.datasets.common import md5sum


def _build_auth_header(token: str) -> str:
    """build the Authorization header value for a Convex token.

    Deploy keys (dev:/prod:) use 'Convex' scheme with a synthetic identity
    so that ctx.auth.getUserIdentity() returns non-null.
    User JWTs use 'Bearer' scheme.
    """
    if token.startswith(("dev:", "prod:")):
        import base64
        identity = base64.b64encode(
            json.dumps({"subject": "sync-script", "issuer": "admin",
                         "tokenIdentifier": "admin|sync-script"}).encode()
        ).decode()
        return f"Convex {token}:{identity}"
    return f"Bearer {token}"


def convex_query(url: str, path: str, args: dict | None = None, token: str | None = None) -> dict:
    """call a Convex query function."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = _build_auth_header(token)
    body = {"path": path, "args": args or {}, "format": "json"}
    resp = httpx.post(f"{url}/api/query", json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") == "error":
        raise RuntimeError(f"Convex error on {path}: {result.get('errorMessage', 'unknown')}")
    return result["value"]


def fetch_ontology_class_map(convex_url: str, ontology_id: str, token: str | None = None) -> dict[str, str]:
    """fetch ontology and build objectClassId -> type mapping (sign, phrase, or skip)."""
    ontology = convex_query(url=convex_url, path="ontologies:get", args={"id": ontology_id}, token=token)
    class_map: dict[str, str] = {}
    for obj_class in ontology.get("objectClasses", []):
        class_id = obj_class["_id"]
        annotation_type = obj_class.get("annotationType", "")
        class_type = obj_class.get("type", "")
        if annotation_type == "time_aligned":
            if "sign" in class_type:
                class_map[class_id] = "sign"
            elif "phrase" in class_type or "spoken" in class_type:
                class_map[class_id] = "phrase"
    return class_map


def fetch_project_annotations(
    convex_url: str,
    project_id: str,
    token: str | None = None,
) -> tuple[dict[str, list[dict]], list[str]]:
    """fetch all annotations for a project, grouped by externalItemId.

    Returns (annotations_by_video, dataset_ids).
    """
    # projects:get returns ontology, linked datasets, and task counts — no auth needed
    project = convex_query(url=convex_url, path="projects:get", args={"id": project_id}, token=token)
    if not project:
        raise ValueError(f"Project {project_id} not found")

    ontology_id = project["ontologyId"]
    dataset_ids = [d["datasetId"] for d in project.get("datasets", [])]
    class_map = fetch_ontology_class_map(convex_url=convex_url, ontology_id=ontology_id, token=token)

    # fetch tasks, filter completed
    tasks = convex_query(url=convex_url, path="tasks:list", args={"projectId": project_id}, token=token)
    completed_tasks = [t for t in tasks if t.get("status") in ("submitted", "approved")]

    # collect unique video IDs from completed tasks
    completed_item_ids = {t["itemId"] for t in completed_tasks if "itemId" in t}

    print(f"  Project {project.get('name', project_id)}: "
          f"{len(completed_tasks)}/{len(tasks)} completed tasks, "
          f"{len(completed_item_ids)} unique videos")

    # fetch annotations per video using annotations:listByItem (no auth, one call per video)
    annotations_by_video: dict[str, list[dict]] = {}
    for item_id in completed_item_ids:
        item_annotations = convex_query(
            url=convex_url, path="annotations:listByItem",
            args={"externalItemId": item_id}, token=token,
        )
        for ann in item_annotations:
            if "startTime" not in ann or "endTime" not in ann:
                continue

            class_type = class_map.get(ann.get("objectClassId", ""))
            if class_type is None:
                continue

            if item_id not in annotations_by_video:
                annotations_by_video[item_id] = []

            annotations_by_video[item_id].append({
                "type": class_type,
                "start": ann["startTime"] * 1000,  # seconds -> ms
                "end": ann["endTime"] * 1000,
            })

    return annotations_by_video, dataset_ids


def resolve_video_paths(
    convex_url: str,
    dataset_id: str,
    video_ids: set[str],
    gcs_root: str,
    poses_dir: str,
    token: str | None = None,
    video_map_cache_path: str | None = None,
) -> dict[str, dict]:
    """resolve externalItemId -> {pose_hash, fps, total_frames} via datasetItems API + local files.

    Caches resolved mappings in video_map_cache_path to avoid re-hashing.
    """
    # load existing cache
    video_map: dict[str, dict] = {}
    if video_map_cache_path and os.path.exists(video_map_cache_path):
        with open(video_map_cache_path) as f:
            video_map = json.load(f)

    # find which video IDs still need resolution
    unresolved = video_ids - set(video_map.keys())
    if not unresolved:
        return {vid: video_map[vid] for vid in video_ids if vid in video_map}

    # fetch dataset items to get video URLs
    print(f"  Fetching dataset items for {len(unresolved)} unresolved videos...")
    items_by_id: dict[str, dict] = {}
    try:
        result = convex_query(
            url=convex_url, path="datasetItems:list",
            args={"datasetId": dataset_id, "limit": 10000}, token=token,
        )
    except RuntimeError as e:
        if "Unauthorized" in str(e):
            print("  WARNING: datasetItems requires auth. Set CONVEX_AUTH_TOKEN env var.")
            return {vid: video_map[vid] for vid in video_ids if vid in video_map}
        raise

    items = result.get("items", []) if isinstance(result, dict) else result
    for item in items:
        item_id = item.get("itemId") or item.get("externalId") or item.get("_id")
        if item_id in unresolved:
            items_by_id[item_id] = item

    print(f"  Fetched {len(items)} dataset items, matched {len(items_by_id)}/{len(unresolved)} videos")

    # resolve video URL -> local path -> md5 -> pose file
    cache_dirty = False
    for video_id in unresolved:
        item = items_by_id.get(video_id)
        if not item:
            continue

        # extract video path from item values
        values = item.get("values", item.get("data", item))
        video_url = values.get("video_url") or values.get("video") or ""
        if not video_url:
            continue

        # map GCS URL to local path
        local_path = _gcs_url_to_local(video_url, gcs_root=gcs_root)
        if not local_path or not os.path.exists(local_path):
            print(f"  WARNING: video file not found for {video_id}: {local_path}")
            continue

        video_hash = md5sum(local_path)
        pose_path = os.path.join(poses_dir, f"{video_hash}.pose")
        if not os.path.exists(pose_path):
            print(f"  WARNING: pose file not found for {video_id}: {pose_path}")
            continue

        # read pose metadata
        with open(pose_path, "rb") as f:
            meta_pose = Pose.read(f, pose_body=EmptyPoseBody)

        video_map[video_id] = {
            "pose_hash": video_hash,
            "fps": float(meta_pose.body.fps),
            "total_frames": int(len(meta_pose.body.data)),
        }
        cache_dirty = True

    # save cache
    if cache_dirty and video_map_cache_path:
        with open(video_map_cache_path, "w") as f:
            json.dump(video_map, f, indent=2)

    return {vid: video_map[vid] for vid in video_ids if vid in video_map}


def _gcs_url_to_local(url: str, gcs_root: str) -> str | None:
    """convert a GCS URL or bucket path to a local filesystem path under gcs_root."""
    if url.startswith("gs://"):
        # gs://bucket-name/path/to/file -> {gcs_root}/bucket-name/path/to/file
        path = url[5:]
        return os.path.join(gcs_root, path)
    if url.startswith("https://storage.googleapis.com/"):
        # https://storage.googleapis.com/bucket-name/path -> {gcs_root}/bucket-name/path
        parsed = urlparse(url)
        path = parsed.path.lstrip("/")
        return os.path.join(gcs_root, path)
    if url.startswith("http"):
        # other URL, try to extract path
        parsed = urlparse(url)
        return os.path.join(gcs_root, parsed.path.lstrip("/"))
    # assume it's already a local path
    return url


def sync(
    convex_url: str,
    project_ids: list[str],
    poses_dir: str,
    gcs_root: str,
    output_path: str,
    token: str | None = None,
) -> None:
    """fetch annotations from Convex and write annotations_cache.json."""
    print(f"Syncing annotations from {convex_url}")
    print(f"Projects: {project_ids}")

    all_annotations: dict[str, list[dict]] = {}
    dataset_ids: set[str] = set()

    for project_id in project_ids:
        annotations_by_video, project_dataset_ids = fetch_project_annotations(
            convex_url=convex_url, project_id=project_id, token=token,
        )
        for video_id, anns in annotations_by_video.items():
            all_annotations.setdefault(video_id, []).extend(anns)
        dataset_ids.update(project_dataset_ids)

    print(f"\nFound {len(all_annotations)} videos with time-aligned annotations")

    # resolve video paths
    video_map_cache = os.path.join(os.path.dirname(output_path), "video_map_cache.json")
    video_map: dict[str, dict] = {}
    for dataset_id in dataset_ids:
        resolved = resolve_video_paths(
            convex_url=convex_url, dataset_id=dataset_id,
            video_ids=set(all_annotations.keys()), gcs_root=gcs_root,
            poses_dir=poses_dir, token=token,
            video_map_cache_path=video_map_cache,
        )
        video_map.update(resolved)

    # build output
    videos: dict[str, dict] = {}
    for video_id, anns in all_annotations.items():
        if video_id not in video_map:
            print(f"  Skipping {video_id}: could not resolve pose file")
            continue

        meta = video_map[video_id]
        signs = [{"start": a["start"], "end": a["end"]} for a in anns if a["type"] == "sign"]
        phrases = [{"start": a["start"], "end": a["end"]} for a in anns if a["type"] == "phrase"]

        videos[video_id] = {
            "pose_hash": meta["pose_hash"],
            "fps": meta["fps"],
            "total_frames": meta["total_frames"],
            "signs": signs,
            "phrases": phrases,
        }

    cache = {
        "synced_at": datetime.now(tz=timezone.utc).isoformat(),
        "project_ids": project_ids,
        "videos": videos,
    }

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"\nWrote {len(videos)} videos to {output_path}")


def score(
    cache_path: str,
    model_path: str,
    poses_dir: str,
    device: str = "cpu",
) -> None:
    """score cached annotations against model predictions, writing quality_score per video."""
    import torch
    from pose_format import Pose as PoseRead

    from sign_language_segmentation.bin import load_model, run_inference
    from sign_language_segmentation.metrics import likeliest_probs_to_segments, segment_IoU

    with open(cache_path) as f:
        cache = json.load(f)

    model = load_model(model_path, device=device)
    scored = 0

    for video_id, video_data in cache.get("videos", {}).items():
        pose_hash = video_data.get("pose_hash")
        if not pose_hash:
            continue

        pose_path = os.path.join(poses_dir, f"{pose_hash}.pose")
        if not os.path.exists(pose_path):
            continue

        signs = video_data.get("signs", [])
        if not signs:
            continue

        # run inference
        with open(pose_path, "rb") as f:
            pose = PoseRead.read(f)

        with torch.inference_mode():
            log_probs = run_inference(model=model, pose=pose, device=device)

        fps = pose.body.fps
        total_frames = len(pose.body.data)

        # model segments (frame indices)
        model_sign_segments = likeliest_probs_to_segments(log_probs["sign"][0].cpu())

        # human segments (convert ms -> frame indices)
        human_sign_segments = []
        for s in signs:
            start_frame = int(s["start"] / 1000 * fps)
            end_frame = int(s["end"] / 1000 * fps)
            human_sign_segments.append({"start": start_frame, "end": min(end_frame, total_frames - 1)})

        # compute IoU
        iou = segment_IoU(
            segments=model_sign_segments,
            segments_gold=human_sign_segments,
            max_len=total_frames,
        )
        video_data["quality_score"] = round(float(iou), 4)
        scored += 1

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Scored {scored} videos, wrote to {cache_path}")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Sync annotations from Convex annotation platform")

    # sync args
    parser.add_argument("--convex_url", type=str, default=os.environ.get("CONVEX_URL", ""))
    parser.add_argument("--project_ids", type=str, nargs="+", help="Convex project IDs to sync")
    parser.add_argument("--poses_dir", type=str, default="/mnt/nas/GCS/sign-mediapipe-holistic-poses")
    parser.add_argument("--gcs_root", type=str, default="/mnt/nas/GCS")
    parser.add_argument("--output", type=str, default="annotations_cache.json")

    # score args
    parser.add_argument("--score", action="store_true", help="Score annotations against model", default=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--cache", type=str, default=None, help="Path to existing cache for scoring")
    parser.add_argument("--device", type=str, default="gpu")

    args = parser.parse_args()

    token = os.environ.get("CONVEX_AUTH_TOKEN")

    if args.score:
        if not args.cache:
            parser.error("--cache required for scoring")
        if not args.model_path:
            parser.error("--model_path required for scoring")
        score(
            cache_path=args.cache,
            model_path=args.model_path,
            poses_dir=args.poses_dir,
            device=args.device,
        )
    else:
        if not args.project_ids:
            parser.error("--project_ids required for syncing")
        sync(
            convex_url=args.convex_url,
            project_ids=args.project_ids,
            poses_dir=args.poses_dir,
            gcs_root=args.gcs_root,
            output_path=args.output,
            token=token,
        )


if __name__ == "__main__":
    main()
