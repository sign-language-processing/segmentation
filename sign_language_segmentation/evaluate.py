"""Evaluate a trained model on the DGS corpus test set.

Processes each video individually to compute per-video metrics
matching the evaluation protocol from the EMNLP 2023 paper.
"""

import argparse

import torch

from sign_language_segmentation.datasets.common import Split, get_dataloader
from sign_language_segmentation.utils.bio import BIO
from sign_language_segmentation.metrics import (
    frame_f1,
    likeliest_probs_to_segments,
    segment_IoU,
    segment_f1,
    bio_labels_to_segments,
    filter_segments,
)
from sign_language_segmentation.model.model import PoseTaggingModel


def evaluate_model(model, dataloader, device, seg_fn=None):
    """Evaluate model on dataloader.

    seg_fn: segmentation function (default: likeliest_probs_to_segments).
    """
    model.eval()
    if seg_fn is None:
        seg_fn = likeliest_probs_to_segments

    all_metrics = {
        "sign_frame_f1": [],
        "sign_IoU": [],
        "sign_segment_f1": [],
        "sentence_frame_f1": [],
        "sentence_IoU": [],
        "sentence_segment_f1": [],
        "hm_IoU": [],
    }

    with torch.no_grad():
        for batch in dataloader:
            pose = batch["pose"].to(device)
            log_probs = model(pose, timestamps=batch.get("timestamps"))

            for i in range(pose.shape[0]):
                item_ious = {}
                for level, key_prefix in [("sign", "sign"), ("sentence", "sentence")]:
                    gold = batch["bio"][level][i]
                    probs = log_probs[level][i].cpu()

                    mask = gold != BIO["UNK"]
                    if mask.sum() == 0:
                        continue

                    num_frames = int(mask.sum())
                    masked_gold = gold[mask]
                    masked_probs = probs[mask]

                    all_metrics[f"{key_prefix}_frame_f1"].append(frame_f1(masked_probs, masked_gold))

                    pred_segments = seg_fn(probs[:num_frames])
                    gold_segments = bio_labels_to_segments(gold[:num_frames])

                    iou = segment_IoU(pred_segments, gold_segments, num_frames)
                    all_metrics[f"{key_prefix}_IoU"].append(iou)
                    item_ious[key_prefix] = iou

                    all_metrics[f"{key_prefix}_segment_f1"].append(segment_f1(pred_segments, gold_segments))

                # per-item HM IoU (matches training validation_step)
                s, p = item_ious.get("sign", 0), item_ious.get("sentence", 0)
                if s > 0 and p > 0:
                    all_metrics["hm_IoU"].append(2 * s * p / (s + p))

    results = {}
    for key, values in all_metrics.items():
        results[key] = sum(values) / len(values) if values else 0.0
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--datasets", type=str, default="dgs", help="comma-separated dataset names (e.g. dgs,platform)")
    parser.add_argument("--corpus", default="/mnt/nas/GCS/sign-external-datasets/dgs-corpus")
    parser.add_argument("--poses", default="/mnt/nas/GCS/sign-mediapipe-holistic-poses")
    parser.add_argument(
        "--signtube_annotations_path",
        type=str,
        default="sign_language_segmentation/datasets/signtube/annotations_cache.json",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_fps", type=float, default=None, help="downsample poses to this FPS before evaluation")
    parser.add_argument(
        "--chunk_multiplier",
        type=float,
        default=1.0,
        help="scale inference chunk size by this factor (e.g. 2.0 for 2x context)",
    )
    parser.add_argument(
        "--min_frames", type=int, default=0, help="drop predicted segments shorter than this many frames (0=off)"
    )
    parser.add_argument(
        "--merge_gap", type=int, default=0, help="merge predicted segments separated by ≤ this many frames (0=off)"
    )
    parser.add_argument(
        "--quality_percentile",
        type=float,
        default=1.0,
        help="keep top X%% of videos by quality score (default: 1.0 = all)",
    )
    eval_args = parser.parse_args()

    model = PoseTaggingModel.load_from_checkpoint(eval_args.checkpoint, map_location=eval_args.device, strict=False)
    model = model.to(eval_args.device)

    if eval_args.chunk_multiplier != 1.0:
        model.hparams.num_frames = int(model.hparams.num_frames * eval_args.chunk_multiplier)

    fps_aug = getattr(model.hparams, "fps_aug", False)
    velocity = getattr(model.hparams, "velocity", True)

    dataloader = get_dataloader(
        split=Split(eval_args.split),
        dataset_names=eval_args.datasets,
        args=eval_args,
        batch_size=1,
        num_frames=999999,
        persistent_workers=False,
        fps_aug=fps_aug,
        velocity=velocity,
    )

    def seg_fn(lp):
        segs = likeliest_probs_to_segments(lp)
        if eval_args.min_frames > 0 or eval_args.merge_gap > 0:
            segs = filter_segments(segs, min_frames=eval_args.min_frames, merge_gap=eval_args.merge_gap)
        return segs

    results = evaluate_model(model, dataloader, eval_args.device, seg_fn=seg_fn)

    print(f"\n{'=' * 50}")
    print(f"Evaluation on {eval_args.datasets} {eval_args.split} set")
    print(f"{'=' * 50}")
    print(f"Sign Frame F1:     {results['sign_frame_f1']:.4f}")
    print(f"Sign IoU:          {results['sign_IoU']:.4f}")
    print(f"Sign Segment F1:   {results['sign_segment_f1']:.4f}")
    print(f"Phrase Frame F1:   {results['sentence_frame_f1']:.4f}")
    print(f"Phrase IoU:        {results['sentence_IoU']:.4f}")
    print(f"Phrase Segment F1: {results['sentence_segment_f1']:.4f}")
    print(f"HM IoU:            {results['hm_IoU']:.4f}")
    print(f"{'=' * 50}")
