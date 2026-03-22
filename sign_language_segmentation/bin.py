#!/usr/bin/env python
"""Inference entry point for 2026 sign language segmentation models.

Loads a PyTorch Lightning checkpoint (.ckpt), runs inference on a .pose file,
and writes an ELAN (.eaf) annotation file with SIGN and SENTENCE tiers.

For 2023 models (torch.jit .pth format) use sign_language_segmentation/old/bin.py.
"""
import argparse
import os
from pathlib import Path

import pympi
import torch
from pose_format import Pose
from pose_format.utils.generic import pose_normalization_info, pose_hide_legs, reduce_holistic

from sign_language_segmentation.data.utils import preprocess_pose
from sign_language_segmentation.metrics import probs_to_segments
from sign_language_segmentation.model.model import PoseTaggingModel


def _default_model_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "dist", "2026", "best.ckpt")


def get_args():
    parser = argparse.ArgumentParser(description="Sign language segmentation inference (2026 model)")
    parser.add_argument("--pose", required=True, type=Path, help="input .pose file")
    parser.add_argument("--elan", required=True, type=str, help="output .eaf file path")
    parser.add_argument("--model", default=None, type=str,
                        help="path to .ckpt checkpoint (default: dist/2026/best.ckpt)")
    parser.add_argument("--video", default=None, type=str, help="video file to link in ELAN")
    parser.add_argument("--no-pose-link", action="store_true", help="do not link pose file in ELAN")
    parser.add_argument("--device", default="cpu", help="inference device (default: cpu)")
    return parser.parse_args()


def load_pose(pose_path: Path) -> Pose:
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())
    return pose


def run_inference(model: PoseTaggingModel, pose: Pose, device: str) -> dict:
    """Preprocess pose and run model inference. Returns log-prob tensors."""
    processed = preprocess_pose(pose, normalize=True, no_face=getattr(model.hparams, 'no_face', False))

    import numpy as np
    pose_data = processed.body.data.filled(0)[:, 0, :, :].astype("float32")  # (T, joints, 3)

    # Velocity (matching training if model was trained with velocity)
    if getattr(model.hparams, 'velocity', False):
        vel = np.diff(pose_data, axis=0, prepend=pose_data[:1])
        pose_data = np.concatenate([pose_data, vel], axis=-1)

    # FPS-based timestamps for RoPE (if model was trained with fps_aug)
    fps = pose.body.fps
    T = len(pose_data)
    if getattr(model.hparams, 'fps_aug', False):
        REFERENCE_FPS = 50.0
        timestamps = torch.from_numpy(
            (np.arange(T, dtype="float32") * REFERENCE_FPS / fps)
        ).unsqueeze(0).to(device)
    else:
        timestamps = None

    pose_tensor = torch.from_numpy(pose_data).unsqueeze(0).to(device)  # (1, T, joints, dims)

    model.eval()
    with torch.no_grad():
        return model(pose_tensor, timestamps=timestamps)


def segments_to_eaf(tiers: dict, fps: float, pose_path: Path,
                    video: str, no_pose_link: bool) -> pympi.Elan.Eaf:
    eaf = pympi.Elan.Eaf(author="sign-language-processing/segmentation")

    if video is not None:
        mimetype = "video/mp4" if video.endswith(".mp4") else None
        eaf.add_linked_file(video, mimetype=mimetype)
    if not no_pose_link:
        eaf.add_linked_file(str(pose_path), mimetype="application/pose")

    for tier_id, segments in tiers.items():
        eaf.add_tier(tier_id)
        for seg in segments:
            start_ms = int(seg["start"] / fps * 1000)
            end_ms = int(seg["end"] / fps * 1000)
            eaf.add_annotation(tier_id, start_ms, end_ms)

    return eaf


def main():
    args = get_args()

    model_path = args.model or _default_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download a checkpoint and place it at dist/2026/best.ckpt, "
            "or pass --model <path>."
        )

    print(f"Loading model: {model_path}")
    model = PoseTaggingModel.load_from_checkpoint(model_path, map_location=args.device)
    model = model.to(args.device)

    print(f"Loading pose: {args.pose}")
    pose = load_pose(args.pose)

    print("Running inference...")
    log_probs = run_inference(model, pose, args.device)

    fps = pose.body.fps
    sign_segments = probs_to_segments(log_probs["sign"][0].cpu())
    sentence_segments = probs_to_segments(log_probs["sentence"][0].cpu())

    print(f"Found {len(sign_segments)} signs, {len(sentence_segments)} sentences")

    eaf = segments_to_eaf(
        tiers={"SIGN": sign_segments, "SENTENCE": sentence_segments},
        fps=fps,
        pose_path=args.pose,
        video=args.video,
        no_pose_link=args.no_pose_link,
    )

    print(f"Saving ELAN file: {args.elan}")
    eaf.to_file(args.elan)


if __name__ == "__main__":
    main()
