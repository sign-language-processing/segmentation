#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

FPS = 25  # frames per second for frame-level evaluation

def parse_vtt_timestamp(ts):
    parts = ts.split(':')
    if len(parts) != 3:
        raise ValueError("Invalid timestamp format")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def get_subtitle_cues(vtt_content):
    cues = []
    lines = vtt_content.splitlines()
    for line in lines:
        if "-->" in line:
            parts = line.split("-->")
            try:
                start = parse_vtt_timestamp(parts[0].strip())
                end = parse_vtt_timestamp(parts[1].strip())
                cues.append({"start": start, "end": end})
            except Exception:
                continue
    return (None, cues)

def get_sign_segments_from_eaf(segmentation_file):
    """Parse an ELAN (.eaf) file and return all segments from the SIGN tier."""
    segments = []
    try:
        tree = ET.parse(segmentation_file)
        root = tree.getroot()
    except Exception:
        return segments
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_value = ts.get("TIME_VALUE")
            if ts_value is not None:
                try:
                    time_slots[ts_id] = float(ts_value) / 1000.0
                except ValueError:
                    time_slots[ts_id] = None
    else:
        return segments
    sign_tier = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "SIGN":
            sign_tier = tier
            break
    if sign_tier is None:
        return segments
    for annotation in sign_tier.findall("ANNOTATION"):
        annotation_elem = next(iter(annotation), None)
        if annotation_elem is None:
            continue
        text_elem = annotation_elem.find("ANNOTATION_VALUE")
        text = text_elem.text if text_elem is not None else ""
        start_time = None
        end_time = None
        if "TIME_SLOT_REF1" in annotation_elem.attrib and "TIME_SLOT_REF2" in annotation_elem.attrib:
            ts1 = annotation_elem.attrib["TIME_SLOT_REF1"]
            ts2 = annotation_elem.attrib["TIME_SLOT_REF2"]
            start_time = time_slots.get(ts1, None)
            end_time = time_slots.get(ts2, None)
        mid = (start_time + end_time) / 2 if start_time is not None and end_time is not None else None
        segments.append({'start': start_time, 'end': end_time, 'mid': mid, 'text': text})
    return segments

def get_segmentation_signs(video_id, segmentation_dir):
    """
    Read sign segments from segmentation files.
    If a file named <video_id>.eaf exists in segmentation_dir, it is parsed as an ELAN file.
    Otherwise, it falls back to reading a VTT file located at segmentation_dir/video_id/demo.vtt.
    """
    video_id = str(video_id)  # ensure video_id is a string
    # Try ELAN file first.
    eaf_path = os.path.join(segmentation_dir, f"{video_id}.eaf")
    if os.path.exists(eaf_path):
        segments = get_sign_segments_from_eaf(eaf_path)
        return segments
    # Fall back to VTT file.
    vtt_path = os.path.join(segmentation_dir, video_id, "demo.vtt")
    if not os.path.exists(vtt_path):
        return []
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            vtt_content = f.read()
    except Exception:
        return []
    _, cues = get_subtitle_cues(vtt_content)
    segments = []
    for cue in cues:
        segments.append({
            'start': cue['start'],
            'end': cue['end'],
            'mid': (cue['start'] + cue['end']) / 2,
            'text': ""
        })
    return segments

def compute_iou_matrix(gt_segments, pred_segments):
    if len(gt_segments) == 0 or len(pred_segments) == 0:
        return np.zeros((len(gt_segments), len(pred_segments)))
    gt_array = np.array([[seg['start'], seg['end']] for seg in gt_segments])
    pred_array = np.array([[seg['start'], seg['end']] for seg in pred_segments])
    inter_start = np.maximum(gt_array[:, 0][:, None], pred_array[:, 0][None, :])
    inter_end = np.minimum(gt_array[:, 1][:, None], pred_array[:, 1][None, :])
    intersection = np.maximum(0, inter_end - inter_start)
    union_start = np.minimum(gt_array[:, 0][:, None], pred_array[:, 0][None, :])
    union_end = np.maximum(gt_array[:, 1][:, None], pred_array[:, 1][None, :])
    union = union_end - union_start
    iou_matrix = intersection / union
    return iou_matrix

def compute_tp_fp_fn_vectorized(gt_segments, pred_segments, threshold):
    if len(gt_segments) == 0:
        return 0, len(pred_segments), 0
    if len(pred_segments) == 0:
        return 0, 0, len(gt_segments)
    iou_matrix = compute_iou_matrix(gt_segments, pred_segments)
    cost_matrix = -iou_matrix  # maximize IoU via minimizing negative IoU
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            tp += 1
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    return tp, fp, fn

def compute_metrics_vectorized(gt_segments_by_video, pred_segments_by_video, thresholds):
    total_tp = {thr: 0 for thr in thresholds}
    total_fp = {thr: 0 for thr in thresholds}
    total_fn = {thr: 0 for thr in thresholds}
    for video_id, gt_segments in tqdm(gt_segments_by_video.items(), 
                                        desc="Computing sign-level metrics", 
                                        total=len(gt_segments_by_video)):
        pred_segments = pred_segments_by_video.get(video_id, [])
        for thr in thresholds:
            tp, fp, fn = compute_tp_fp_fn_vectorized(gt_segments, pred_segments, thr)
            total_tp[thr] += tp
            total_fp[thr] += fp
            total_fn[thr] += fn
    f1_scores = {}
    for thr in thresholds:
        tp = total_tp[thr]
        fp = total_fp[thr]
        fn = total_fn[thr]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[thr] = f1
    mF1S = sum(f1_scores.values()) / len(f1_scores)
    return f1_scores, mF1S

def main():
    parser = argparse.ArgumentParser(
        description="Compute mF1S (sign-level) and frame-level F1 metrics between CSV annotations and segmentation predictions."
    )
    parser.add_argument("--split", type=str, default="val",
                        help="The split value to filter the CSV data (default: 'val')")
    parser.add_argument("--segmentation_dir", type=str,
                        default="/users/zifan/BOBSL/derivatives/segmentation_bsl/mstcn_bsl1k_cmpl",
                        help="Directory with segmentation files (ELAN .eaf or VTT) (optional)")
    args = parser.parse_args()

    # Read data.csv from the same directory as the script, forcing video_id to be a string.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data.csv")
    try:
        df = pd.read_csv(csv_path, dtype={'video_id': str})
    except FileNotFoundError:
        print(f"Error: data.csv not found in {script_dir}.")
        return

    if "split" not in df.columns:
        print("Error: 'split' column not found in data.csv.")
        return

    # Filter CSV rows based on the provided split value.
    filtered_df = df[df["split"] == args.split]
    print("Filtered CSV data:")
    print(filtered_df)

    # Build ground truth sign segments using start_gloss and end_gloss.
    gt_df = filtered_df[['video_id', 'example_id', 'start_gloss', 'end_gloss']].drop_duplicates()
    gt_segments_by_video = {}
    for _, row in gt_df.iterrows():
        vid = str(row['video_id'])
        segment = {'start': float(row['start_gloss']), 'end': float(row['end_gloss'])}
        gt_segments_by_video.setdefault(vid, []).append(segment)

    total_gt_signs = sum(len(segments) for segments in gt_segments_by_video.values())
    total_videos = len(gt_segments_by_video)
    print(f"CSV Summary for split '{args.split}':")
    print(f"  Total ground truth signs: {total_gt_signs}")
    print(f"  Total unique videos: {total_videos}")

    # Pre-read predicted segmentation signs for each video.
    pred_segments_by_video = {}
    print("\nReading predicted segmentation signs:")
    for video_id in tqdm(gt_segments_by_video.keys(), desc="Reading segmentations", total=len(gt_segments_by_video)):
        pred_segments_by_video[video_id] = get_segmentation_signs(video_id, args.segmentation_dir)

    total_pred_signs = sum(len(segments) for segments in pred_segments_by_video.values())
    print(f"\nTotal predicted segmentation signs (before filtering): {total_pred_signs}")
    for video_id, segments in pred_segments_by_video.items():
        print(f"  Video ID {video_id}: {len(segments)} predicted sign(s)")

    # --- NEW STEP: Filter predicted signs based on example intervals ---
    # For each (video_id, example_id), compute the overall example interval:
    #   start_time = min(start_gloss, start_sub) and end_time = max(end_gloss, end_sub)
    example_intervals_by_video = {}
    grouped = filtered_df.groupby(["video_id", "example_id"])
    for (video_id, example_id), group in grouped:
        video_id = str(video_id)
        start_time = min(group["start_gloss"].min(), group["start_sub"].min())
        end_time = max(group["end_gloss"].max(), group["end_sub"].max())
        example_intervals_by_video.setdefault(video_id, []).append((example_id, start_time, end_time))
    
    # For every video, filter the predicted signs: only keep those that overlap with at least one example interval.
    filtered_pred_segments_by_video = {}
    print("\nFiltering predicted segmentation signs based on example intervals:")
    for video_id, pred_segments in pred_segments_by_video.items():
        intervals = example_intervals_by_video.get(video_id, [])
        filtered_segments = []
        for seg in pred_segments:
            seg_start, seg_end = seg["start"], seg["end"]
            for (_, ex_start, ex_end) in intervals:
                if seg_end > ex_start and seg_start < ex_end:
                    filtered_segments.append(seg)
                    break
        filtered_pred_segments_by_video[video_id] = filtered_segments
        print(f"  Video ID {video_id}: {len(filtered_segments)}/{len(pred_segments)} predicted sign(s) remain after filtering")

    # --- Frame-level Evaluation ---
    total_frame_tp = 0
    total_frame_fp = 0
    total_frame_fn = 0
    total_frames = 0

    # Build ground truth segments per example.
    gt_segments_by_example = {}
    for (video_id, example_id), group in grouped:
        video_id = str(video_id)
        segments = []
        for _, row in group.iterrows():
            segments.append((float(row["start_gloss"]), float(row["end_gloss"])))
        gt_segments_by_example[(video_id, example_id)] = segments

    print("\nComputing frame-level evaluation (FPS = 25)...")
    for (video_id, example_id), group in tqdm(grouped, desc="Frame-level evaluation", total=len(grouped)):
        video_id = str(video_id)
        start_time = min(group["start_gloss"].min(), group["start_sub"].min())
        end_time = max(group["end_gloss"].max(), group["end_sub"].max())
        duration = end_time - start_time
        num_frames = int(round(duration * FPS))
        if num_frames < 1:
            continue
        times = np.linspace(start_time, end_time, num=num_frames, endpoint=False)
        gt_mask = np.zeros(num_frames, dtype=bool)
        for seg_start, seg_end in gt_segments_by_example[(video_id, example_id)]:
            gt_mask |= ((times >= seg_start) & (times < seg_end))
        pred_mask = np.zeros(num_frames, dtype=bool)
        if video_id in filtered_pred_segments_by_video:
            for seg in filtered_pred_segments_by_video[video_id]:
                if seg["end"] > start_time and seg["start"] < end_time:
                    seg_start = max(seg["start"], start_time)
                    seg_end = min(seg["end"], end_time)
                    pred_mask |= ((times >= seg_start) & (times < seg_end))
        tp = np.sum(pred_mask & gt_mask)
        fp = np.sum(pred_mask & ~gt_mask)
        fn = np.sum(~pred_mask & gt_mask)
        total_frame_tp += tp
        total_frame_fp += fp
        total_frame_fn += fn
        total_frames += num_frames

    frame_precision = total_frame_tp / (total_frame_tp + total_frame_fp) if (total_frame_tp + total_frame_fp) > 0 else 0
    frame_recall = total_frame_tp / (total_frame_tp + total_frame_fn) if (total_frame_tp + total_frame_fn) > 0 else 0
    frame_f1 = 2 * frame_precision * frame_recall / (frame_precision + frame_recall) if (frame_precision + frame_recall) > 0 else 0

    # --- Sign-level (mF1S) Evaluation ---
    thresholds = [round(0.4 + i * 0.05, 2) for i in range(int((0.75 - 0.4) / 0.05) + 1)]
    f1_scores, mF1S = compute_metrics_vectorized(gt_segments_by_video, filtered_pred_segments_by_video, thresholds)

    # --- Print All Metrics Together in a Summary ---
    print("\n==========================================")
    print("           Evaluation Summary")
    print("==========================================")
    print("Frame-level Evaluation:")
    print(f"   Total frames evaluated: {total_frames}")
    print(f"   Precision:            {frame_precision:.4f}")
    print(f"   Recall:               {frame_recall:.4f}")
    print(f"   F1 Score:             {frame_f1:.4f}")
    print("\nSign-level Evaluation:")
    print(f"   Mean F1 Score (mF1S):   {mF1S:.4f}")
    print("   F1 Scores at different IoU thresholds:")
    for thr in sorted(f1_scores.keys()):
        print(f"      IoU >= {thr:0.2f}:        {f1_scores[thr]:.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()
