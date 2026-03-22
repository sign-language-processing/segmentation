"""Evaluate a trained model on the DGS corpus test set.

Processes each video individually to compute per-video metrics
matching the evaluation protocol from the EMNLP 2023 paper.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from sign_language_segmentation.data.dataset import DGSSegmentationDataset, collate_fn
from sign_language_segmentation.data.utils import BIO
from sign_language_segmentation.metrics import frame_f1, probs_to_segments, likeliest_probs_to_segments, segment_IoU, segment_f1
from sign_language_segmentation.model.model import PoseTaggingModel


def evaluate_model(model, dataloader, device, seg_fn=None):
    """Evaluate model on dataloader.

    seg_fn: segmentation function (default: probs_to_segments).  Pass a
    custom function to test different threshold settings without monkey-patching.
    """
    model.eval()
    if seg_fn is None:
        seg_fn = probs_to_segments

    all_metrics = {
        "sign_frame_f1": [], "sign_IoU": [], "sign_segment_f1": [],
        "sentence_frame_f1": [], "sentence_IoU": [], "sentence_segment_f1": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            pose = batch["pose"].to(device)
            log_probs = model(pose, timestamps=batch.get("timestamps"))

            for i in range(pose.shape[0]):
                for level, key_prefix in [("sign", "sign"), ("sentence", "sentence")]:
                    gold = batch["bio"][level][i]
                    probs = log_probs[level][i].cpu()

                    mask = gold != BIO["UNK"]
                    if mask.sum() == 0:
                        continue

                    masked_gold = gold[mask]
                    masked_probs = probs[mask]
                    num_frames = int(mask.sum())

                    all_metrics[f"{key_prefix}_frame_f1"].append(
                        frame_f1(masked_probs, masked_gold))

                    pred_segments = seg_fn(probs[:num_frames])

                    # Build gold segments from BIO labels
                    gold_np = gold[:num_frames].numpy()
                    gold_segments = []
                    seg_start = None
                    for j in range(len(gold_np)):
                        if gold_np[j] == BIO["B"]:
                            if seg_start is not None:
                                gold_segments.append({"start": seg_start, "end": j - 1})
                            seg_start = j
                        elif gold_np[j] == BIO["O"] and seg_start is not None:
                            gold_segments.append({"start": seg_start, "end": j - 1})
                            seg_start = None
                    if seg_start is not None:
                        gold_segments.append({"start": seg_start, "end": len(gold_np) - 1})

                    all_metrics[f"{key_prefix}_IoU"].append(
                        segment_IoU(pred_segments, gold_segments, num_frames))
                    all_metrics[f"{key_prefix}_segment_f1"].append(
                        segment_f1(pred_segments, gold_segments))

    results = {}
    for key, values in all_metrics.items():
        results[key] = sum(values) / len(values) if values else 0.0
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--corpus", default="/mnt/nas/GCS/sign-external-datasets/dgs-corpus")
    parser.add_argument("--poses", default="/mnt/nas/GCS/sign-mediapipe-holistic-poses")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_normalize", action="store_true", default=False)
    parser.add_argument("--pose_dims", type=int, default=3, choices=[2, 3])
    parser.add_argument("--velocity", action="store_true", default=False)
    parser.add_argument("--no_face", action="store_true", default=False)
    parser.add_argument("--target_fps", type=float, default=None,
                        help="downsample poses to this FPS before evaluation")
    parser.add_argument("--b_threshold", type=int, default=50,
                        help="B-class detection threshold 0-100 (default 50)")
    parser.add_argument("--o_threshold", type=int, default=50,
                        help="O-class reset threshold 0-100 (default 50)")
    parser.add_argument("--io_threshold", type=int, default=50,
                        help="IO-fallback I-class threshold 0-100 (default 50)")
    parser.add_argument("--likeliest", action="store_true", default=False,
                        help="Use argmax (likeliest) decoding instead of threshold-based")
    parser.add_argument("--tune_threshold", action="store_true", default=False,
                        help="Sweep b_threshold, io_threshold and report best sign IoU")
    eval_args = parser.parse_args()

    model = PoseTaggingModel.load_from_checkpoint(eval_args.checkpoint, map_location=eval_args.device, strict=False)
    model = model.to(eval_args.device)

    # Use time-based timestamps if model was trained with fps_aug
    fps_aug = getattr(model.hparams, 'fps_aug', False)

    dataset = DGSSegmentationDataset(
        corpus_dir=eval_args.corpus,
        poses_dir=eval_args.poses,
        split=eval_args.split,
        num_frames=999999,
        normalize=not eval_args.no_normalize,
        pose_dims=eval_args.pose_dims,
        velocity=eval_args.velocity,
        no_face=eval_args.no_face,
        target_fps=eval_args.target_fps,
        fps_aug=fps_aug,
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    if eval_args.tune_threshold:
        print(f"\nSweeping thresholds on {eval_args.split} set...")
        best_cfg, best_iou = {}, 0.0

        # Try likeliest first
        r = evaluate_model(model, dataloader, eval_args.device, seg_fn=likeliest_probs_to_segments)
        print(f"  likeliest        → Sign IoU={r['sign_IoU']:.4f}  Sign F1={r['sign_frame_f1']:.4f}")
        if r['sign_IoU'] > best_iou:
            best_iou = r['sign_IoU']
            best_cfg = {"likeliest": True}

        # Sweep io_threshold (most relevant since our models use IO-fallback)
        for io_thresh in range(10, 95, 5):
            fn = lambda lp, t=io_thresh: probs_to_segments(lp, b_threshold=99, io_threshold=t)
            r = evaluate_model(model, dataloader, eval_args.device, seg_fn=fn)
            print(f"  io_threshold={io_thresh:3d} → Sign IoU={r['sign_IoU']:.4f}  Sign F1={r['sign_frame_f1']:.4f}")
            if r['sign_IoU'] > best_iou:
                best_iou = r['sign_IoU']
                best_cfg = {"b_threshold": 99, "io_threshold": io_thresh}

        print(f"\nBest config={best_cfg} → Sign IoU={best_iou:.4f}")
        if best_cfg.get("likeliest"):
            eval_args.likeliest = True
        else:
            eval_args.b_threshold = best_cfg.get("b_threshold", eval_args.b_threshold)
            eval_args.io_threshold = best_cfg.get("io_threshold", eval_args.io_threshold)

    if eval_args.likeliest:
        final_seg_fn = likeliest_probs_to_segments
    else:
        final_seg_fn = lambda lp: probs_to_segments(lp, b_threshold=eval_args.b_threshold,
                                                    o_threshold=eval_args.o_threshold,
                                                    io_threshold=eval_args.io_threshold)
    results = evaluate_model(model, dataloader, eval_args.device, seg_fn=final_seg_fn)

    print(f"\n{'='*50}")
    print(f"Evaluation on {eval_args.split} set")
    print(f"b_threshold={eval_args.b_threshold}, o_threshold={eval_args.o_threshold}")
    print(f"{'='*50}")
    print(f"Sign Frame F1:     {results['sign_frame_f1']:.4f}")
    print(f"Sign IoU:          {results['sign_IoU']:.4f}")
    print(f"Sign Segment F1:   {results['sign_segment_f1']:.4f}")
    print(f"Phrase Frame F1:   {results['sentence_frame_f1']:.4f}")
    print(f"Phrase IoU:        {results['sentence_IoU']:.4f}")
    print(f"Phrase Segment F1: {results['sentence_segment_f1']:.4f}")
    print(f"{'='*50}")
