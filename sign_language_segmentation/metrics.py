from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score

from sign_language_segmentation.data.utils import BIO


def frame_f1(probs: torch.Tensor, gold: torch.Tensor) -> float:
    return f1_score(gold.cpu().numpy(), probs.argmax(dim=1).cpu().numpy(), average='macro', zero_division=0)


def probs_to_segments(log_probs: torch.Tensor, b_threshold=50, o_threshold=50, io_threshold=50) -> List[dict]:
    probs = np.round(np.exp(log_probs.cpu().numpy()) * 100)

    if np.all(probs[:, BIO["B"]] < b_threshold):
        return _io_probs_to_segments(probs, io_threshold=io_threshold)

    segments = []
    segment_start = None
    did_pass_start = False

    for idx in range(len(probs)):
        b = float(probs[idx, BIO["B"]])
        o = float(probs[idx, BIO["O"]])

        if segment_start is None:
            if b > b_threshold:
                segment_start = idx
                did_pass_start = False
        else:
            if did_pass_start:
                if b > b_threshold or o > o_threshold:
                    segments.append({"start": segment_start, "end": idx - 1})
                    segment_start = idx if b > b_threshold else None
                    did_pass_start = False
            elif b < b_threshold:
                did_pass_start = True

    if segment_start is not None:
        segments.append({"start": segment_start, "end": len(probs) - 1})

    return segments


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


def _io_probs_to_segments(probs: np.ndarray, io_threshold=50) -> List[dict]:
    segments = []
    i = 0
    while i < len(probs):
        if probs[i, BIO["I"]] > io_threshold:
            end = len(probs) - 1
            for j in range(i + 1, len(probs)):
                if probs[j, BIO["I"]] < io_threshold:
                    end = j - 1
                    break
            segments.append({"start": i, "end": end})
            i = end + 1
        else:
            i += 1
    return segments


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
        gold_range = range(start, end + 1)
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
