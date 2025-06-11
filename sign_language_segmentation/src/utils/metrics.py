from typing import List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import linear_sum_assignment


def frame_accuracy(probs: torch.Tensor, gold: torch.Tensor) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return float(torch.sum(gold == probs.argmax(dim=1)) / gold.shape[0])


def frame_f1(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return f1_score(gold.numpy(), probs.argmax(dim=1).numpy(), **kwargs)


def frame_precision(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return precision_score(gold.numpy(), probs.argmax(dim=1).numpy(), zero_division=np.nan, **kwargs)


def frame_recall(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return recall_score(gold.numpy(), probs.argmax(dim=1).numpy(), zero_division=np.nan, **kwargs)


def frame_roc_auc(probs: torch.Tensor, gold: torch.Tensor, **kwargs) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    gold_np = gold.numpy()
    if len(np.unique(gold_np)) < 3:
        return [np.nan]
    return roc_auc_score(gold.numpy(), np.exp(probs.numpy()), **kwargs)


def segment_percentage(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    if len(segments_gold) == 0:
        return 1 if len(segments) == 0 else len(segments)

    return len(segments) / len(segments_gold)


def segment_IoU(segments: List[dict], segments_gold: List[dict], max_len=1000000) -> float:
    segments_v = np.zeros(max_len)
    for segment in segments:
        segments_v[segment['start']:(segment['end'] + 1)] = 1

    segments_gold_v = np.zeros(max_len)
    for segment in segments_gold:
        segments_gold_v[segment['start']:(segment['end'] + 1)] = 1

    intersection = np.logical_and(segments_v, segments_gold_v)
    union = np.logical_or(segments_v, segments_gold_v)

    if np.sum(union) == 0:
        return 1 if np.sum(intersection) == 0 else 0

    return float(np.sum(intersection) / np.sum(union))


def segment_boundary_f1(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segment boundary f1 as described in Bull et al.
    segments: [{'start': 1, 'end': 2}, ...]
    """
    return segment_f1(segments_to_boundaries(segments), segments_to_boundaries(segments_gold))


def segments_to_boundaries(segments: List[dict]) -> List[dict]:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    boundaries = []
    for i, segment in enumerate(segments):
        if i == len(segments) - 1:
            break
        segment_next = segments[i + 1]
        boundary = {"start": segment["end"], "end": segment_next["start"]}
        boundaries.append(boundary)
    return boundaries


def segment_f1(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    if len(segments_gold) == 0 or len(segments) == 0:
        return 1 if len(segments) == len(segments_gold) else 0

    precision = segment_precision(segments, segments_gold)
    recall = segment_recall(segments, segments_gold)
    return (precision * recall) / (precision + recall) if (precision > 0 and recall > 0) else 0


def segment_precision(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    return segment_recall(segments_gold, segments)


def segment_recall(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    hit = 0
    allowed_shift = 15  # 0.6s under 25 fps
    allowed_shift = allowed_shift + 2
    for segment_gold in segments_gold:
        start = segment_gold["start"] - allowed_shift
        end = segment_gold["end"] + allowed_shift
        segment_gold_range = range(start, end)
        # if there is intersection betweeen segment_gold_range and any of the segments
        if any([len(set(range(segment['start'], segment['end'])).intersection(segment_gold_range)) > 0 for segment in
                segments]):
            hit = hit + 1
    return hit / len(segments_gold)

def segment_mF1S(pred_segments, gt_segments, thresholds=None):
    """
    Compute the mean sign‐level F1 (mF1S) over a range of IoU thresholds.

    Args:
        pred_segments (List[Dict]): predicted segments, each with 'start' and 'end'
        gt_segments   (List[Dict]): ground‐truth segments, same format
        thresholds    (List[float], optional): IoU thresholds to evaluate.
                         Defaults to [0.40, 0.45, …, 0.75].

    Returns:
        float: the mean F1 score across all thresholds.
    """
    if thresholds is None:
        thresholds = [round(0.4 + i * 0.05, 2)
                      for i in range(int((0.75 - 0.4) / 0.05) + 1)]

    def _iou_matrix(gt, pred):
        if not gt or not pred:
            # shape (len(gt), len(pred))
            return np.zeros((len(gt), len(pred)))
        gt_arr   = np.array([[s['start'], s['end']] for s in gt])
        pred_arr = np.array([[s['start'], s['end']] for s in pred])
        # intersection
        istart = np.maximum(gt_arr[:, 0][:, None], pred_arr[:, 0][None, :])
        iend   = np.minimum(gt_arr[:, 1][:, None], pred_arr[:, 1][None, :])
        intersect = np.maximum(0, iend - istart)
        # union
        ustart = np.minimum(gt_arr[:, 0][:, None], pred_arr[:, 0][None, :])
        uend   = np.maximum(gt_arr[:, 1][:, None], pred_arr[:, 1][None, :])
        union = uend - ustart
        return intersect / union

    def _tp_fp_fn(gt, pred, thr):
        if not gt:
            # all preds are false positives
            return 0, len(pred), 0
        if not pred:
            # all gts are false negatives
            return 0, 0, len(gt)
        iou_m = _iou_matrix(gt, pred)
        # match to maximize total IoU = minimize negative IoU
        row_ind, col_ind = linear_sum_assignment(-iou_m)
        tp = sum(1 for r, c in zip(row_ind, col_ind) if iou_m[r, c] >= thr)
        fp = len(pred) - tp
        fn = len(gt)   - tp
        return tp, fp, fn

    f1_scores = []
    for thr in thresholds:
        tp, fp, fn = _tp_fp_fn(gt_segments, pred_segments, thr)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))