#!/usr/bin/env python
"""Inference entry point for 2026 sign language segmentation models.

Loads a PyTorch Lightning checkpoint (.ckpt), runs inference on a .pose file,
and writes an ELAN (.eaf) annotation file with SIGN and SENTENCE tiers.
"""
import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pympi
import torch
from pose_format import Pose
from safetensors.torch import load_file as load_safetensors

from sign_language_segmentation.utils.pose import preprocess_pose, compute_velocity
from sign_language_segmentation.metrics import likeliest_probs_to_segments, filter_segments
from sign_language_segmentation.model.model import PoseTaggingModel

_BAKED_IN_DIR = Path(__file__).resolve().parent / "dist" / "2026"


def resolve_model_path() -> str:
    """Resolve model directory path.

    Priority: MODEL_PATH env > HF_MODEL_REPO env > baked-in package default.
    """
    # 1. explicit local path
    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        return explicit

    # 2. huggingface hub download
    hf_repo = os.environ.get("HF_MODEL_REPO")
    if hf_repo:
        return _download_from_hf(hf_repo)

    # 3. baked-in default
    return str(_BAKED_IN_DIR)


def _download_from_hf(repo_id: str) -> str:
    """Download model from HuggingFace Hub. Returns local cache directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF_MODEL_REPO. "
            "Install with: pip install sign-language-segmentation[hf]"
        )
    revision = os.environ.get("HF_MODEL_REVISION")
    if not revision:
        raise ValueError("HF_MODEL_REVISION must be set when using HF_MODEL_REPO")
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model.safetensors", "config.json"],
    )


def _load_from_safetensors(model_dir: str, device: str) -> PoseTaggingModel:
    """Load model from safetensors + config.json directory."""
    model_dir_path = Path(model_dir)
    with open(model_dir_path / "config.json") as f:
        config = json.load(f)
    # config.json stores tuples as lists — convert pose_dims back
    if "pose_dims" in config:
        config["pose_dims"] = tuple(config["pose_dims"])
    model = PoseTaggingModel(**config)
    state_dict = load_safetensors(filename=str(model_dir_path / "model.safetensors"), device=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_model(model_dir: str, device: str = "cpu", revision: str = "") -> PoseTaggingModel:
    # revision is part of the cache key only — callers pass HF_MODEL_REVISION so a mid-process
    # env change invalidates the cache entry instead of silently returning a stale model.
    model_dir_path = Path(model_dir)
    # prefer safetensors if available, fall back to .ckpt
    if (model_dir_path / "model.safetensors").exists():
        return _load_from_safetensors(model_dir=str(model_dir), device=device)
    # backward compat: load .ckpt directly (model_dir might be a file path)
    ckpt_path = model_dir_path if model_dir_path.suffix == ".ckpt" else model_dir_path / "best.ckpt"
    model = PoseTaggingModel.load_from_checkpoint(checkpoint_path=str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def run_inference(model: PoseTaggingModel, pose: Pose, device: str) -> dict:
    """Preprocess pose and run model inference. Returns log-prob tensors."""
    processed = preprocess_pose(pose)
    pose_data = processed.body.data.filled(0)[:, 0, :, :3].astype("float32")  # (T, joints, 3)

    fps = pose.body.fps
    T = len(pose_data)
    frame_times = np.arange(T, dtype="float32") / fps

    if getattr(model.hparams, 'velocity', True):
        vel = compute_velocity(pose_data, frame_times)
        pose_data = np.concatenate([pose_data, vel], axis=-1)

    timestamps = torch.from_numpy(frame_times).unsqueeze(0).to(device)
    pose_tensor = torch.from_numpy(pose_data).unsqueeze(0).to(device)  # (1, T, joints, dims)

    return model(pose_tensor, timestamps=timestamps)


def segment_pose(pose: Pose, model_dir: str = None, device: str = "cpu",
                 min_frames: int = 3, merge_gap: int = 0):
    """Segment a pose into signs and sentences.

    Returns:
        eaf: pympi.Elan.Eaf with SIGN and SENTENCE tiers
        tiers: dict mapping tier name to list of {start, end} segment dicts
    """
    model_dir = model_dir or resolve_model_path()
    revision = os.environ.get("HF_MODEL_REVISION", "")
    model = load_model(model_dir=model_dir, device=device, revision=revision)

    log_probs = run_inference(model=model, pose=pose, device=device)

    fps = pose.body.fps
    seg_fn = likeliest_probs_to_segments
    tiers = {
        "SIGN": filter_segments(seg_fn(log_probs["sign"][0].cpu()), min_frames=min_frames, merge_gap=merge_gap),
        "SENTENCE": filter_segments(seg_fn(log_probs["sentence"][0].cpu()), min_frames=min_frames, merge_gap=merge_gap),
    }

    eaf = pympi.Elan.Eaf(author="sign-language-processing/segmentation")
    for tier_id, segments in tiers.items():
        eaf.add_tier(tier_id)
        for seg in segments:
            start_ms = int(seg["start"] / fps * 1000)
            end_ms = int(seg["end"] / fps * 1000)
            eaf.add_annotation(tier_id, start_ms, end_ms)

    return eaf, tiers


def save_pose_segments(tiers: dict, tier_id: str, input_file_path: Path) -> None:
    """Save cropped .pose files for each segment in the given tier."""
    with input_file_path.open("rb") as f:
        pose = Pose.read(f)

    for i, segment in enumerate(tiers[tier_id]):
        out_path = input_file_path.parent / f"{input_file_path.stem}_{tier_id}_{i}.pose"
        start_frame = int(segment["start"])
        end_frame = int(segment["end"])
        cropped_pose = Pose(header=pose.header, body=pose.body[start_frame:end_frame + 1])

        print(f"Saving cropped pose with start {start_frame} and end {end_frame} to {out_path}")
        with out_path.open("wb") as f:
            cropped_pose.write(f)


def get_args():
    parser = argparse.ArgumentParser(description="Sign language segmentation inference (2026 model)")
    parser.add_argument("--pose", required=True, type=Path, help="input .pose file")
    parser.add_argument("--elan", required=True, type=str, help="output .eaf file path")
    parser.add_argument("--model", default=None, type=str,
                        help="path to model directory (safetensors) or .ckpt file")
    parser.add_argument("--video", default=None, type=str, help="video file to link in ELAN")
    parser.add_argument("--subtitles", default=None, type=str, help="path to .srt subtitle file")
    parser.add_argument("--no-pose-link", action="store_true", help="do not link pose file in ELAN")
    parser.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    parser.add_argument("--min_frames", type=int, default=3,
                        help="drop segments shorter than N frames (default: 3, ~60ms at 50fps)")
    parser.add_argument("--merge_gap", type=int, default=0,
                        help="merge segments with gaps ≤ N frames (default: 0 = off)")
    parser.add_argument("--save-segments", type=str, choices=["SENTENCE", "SIGN"],
                        help="save cropped .pose files for each segment in this tier")
    return parser.parse_args()


def main():
    args = get_args()

    model_dir = args.model or resolve_model_path()
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model not found: {model_dir}\n"
            "Set HF_MODEL_REPO env var, pass --model <path>, "
            "or place model files at dist/2026/."
        )

    print(f"Loading pose: {args.pose}")
    with open(args.pose, "rb") as f:
        pose = Pose.read(f)

    print(f"Loading model: {model_dir}")
    print("Running inference...")
    eaf, tiers = segment_pose(pose, model_dir=model_dir, device=args.device,
                               min_frames=args.min_frames, merge_gap=args.merge_gap)

    sign_count = len(tiers["SIGN"])
    sentence_count = len(tiers["SENTENCE"])
    print(f"Found {sign_count} signs, {sentence_count} sentences")

    if args.video is not None:
        mimetype = "video/mp4" if args.video.endswith(".mp4") else None
        eaf.add_linked_file(args.video, mimetype=mimetype)
    if not args.no_pose_link:
        eaf.add_linked_file(str(args.pose), mimetype="application/pose")

    if args.subtitles and os.path.exists(args.subtitles):
        import srt
        eaf.add_tier("SUBTITLE")
        with open(args.subtitles, "r", encoding="utf-8-sig") as f:
            for subtitle in srt.parse(f):
                start = subtitle.start.total_seconds()
                end = subtitle.end.total_seconds()
                eaf.add_annotation("SUBTITLE", int(start * 1000), int(end * 1000), subtitle.content)

    if args.save_segments:
        print(f"Saving {args.save_segments} cropped .pose files")
        save_pose_segments(tiers, tier_id=args.save_segments, input_file_path=args.pose)

    print(f"Saving ELAN file: {args.elan}")
    eaf.to_file(args.elan)


if __name__ == "__main__":
    main()
