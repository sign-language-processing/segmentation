from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score

from sign_language_segmentation.data.utils import BIO


def bio_labels_to_segments(gold: torch.Tensor) -> List[dict]:
    """Convert a BIO label tensor to a list of {start, end} segment dicts."""
    gold_np = gold.cpu().numpy()
    segments = []
    seg_start = None
    for j, label in enumerate(gold_np):
        if label == BIO["B"]:
            if seg_start is not None:
                segments.append({"start": seg_start, "end": j - 1})
            seg_start = j
        elif label == BIO["O"] and seg_start is not None:
            segments.append({"start": seg_start, "end": j - 1})
            seg_start = None
    if seg_start is not None:
        segments.append({"start": seg_start, "end": len(gold_np) - 1})
    return segments


def frame_f1(probs: torch.Tensor, gold: torch.Tensor) -> float:
    return f1_score(gold.cpu().numpy(), probs.argmax(dim=1).cpu().numpy(), average='macro', zero_division=0)


def likeliest_probs_to_segments(log_probs: torch.Tensor) -> List[dict]:
    """Argmax decoding: group contiguous B/I frames into segments."""
    preds = log_probs.cpu().numpy().argmax(axis=1)  # (T,)
    segments = []
    seg_start = None
    for i, p in enumerate(preds):
        if p in (BIO["B"], BIO["I"]):
            if seg_start is None:
                seg_start = i
        else:  # O or UNK
            if seg_start is not None:
                segments.append({"start": seg_start, "end": i - 1})
                seg_start = None
    if seg_start is not None:
        segments.append({"start": seg_start, "end": len(preds) - 1})
    return segments


def filter_segments(segments: List[dict], min_frames: int = 3,
                    merge_gap: int = 0) -> List[dict]:
    """Post-processing: remove very short segments and optionally merge close ones.

    Args:
        segments:   list of {start, end} dicts (sorted by start).
        min_frames: drop segments shorter than this (removes single-frame noise).
        merge_gap:  merge adjacent segments separated by ≤ this many frames.
                    0 = no merging (default).  Set to e.g. 3 to bridge tiny O gaps.

    Returns:
        Filtered/merged segment list.
    """
    if not segments:
        return segments

    # Merge close segments first
    if merge_gap > 0:
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            if seg["start"] - merged[-1]["end"] - 1 <= merge_gap:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(dict(seg))
        segments = merged

    # Drop short segments
    return [s for s in segments if s["end"] - s["start"] + 1 >= min_frames]


def segment_IoU(segments: List[dict], segments_gold: List[dict], max_len: int) -> float:
    pred_v = np.zeros(max_len)
    for s in segments:
        pred_v[s['start']:s['end'] + 1] = 1

    gold_v = np.zeros(max_len)
    for s in segments_gold:
        gold_v[s['start']:s['end'] + 1] = 1

    intersection = np.logical_and(pred_v, gold_v).sum()
    union = np.logical_or(pred_v, gold_v).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def _segment_recall(segments: List[dict], segments_gold: List[dict]) -> float:
    if len(segments_gold) == 0:
        return 1.0 if len(segments) == 0 else 0.0

    allowed_shift = 17  # ~0.68s at 25fps
    hit = 0
    for sg in segments_gold:
        start = sg["start"] - allowed_shift
        end = sg["end"] + allowed_shift
        if any(range(max(s['start'], start), min(s['end'], end) + 1) for s in segments
               if s['start'] <= end and s['end'] >= start):
            hit += 1
    return hit / len(segments_gold)


def segment_f1(segments: List[dict], segments_gold: List[dict]) -> float:
    if len(segments_gold) == 0 or len(segments) == 0:
        return 1.0 if len(segments) == len(segments_gold) else 0.0

    precision = _segment_recall(segments_gold, segments)
    recall = _segment_recall(segments, segments_gold)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
