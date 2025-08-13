import os
import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sign_language_segmentation.bin import load_model, process_pose, predict
from utils.metrics import frame_accuracy, frame_f1, frame_precision, frame_recall, frame_roc_auc, segment_percentage, segment_IoU
from pose_format import PoseHeader, Pose

# ----------------------
# Define a simple Subtitle class.
# ----------------------
class Subtitle:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text

    def __repr__(self):
        return f"Subtitle(start={self.start}, end={self.end}, text='{self.text}')"

# ----------------------
# Functions for converting time and building gold labels.
# ----------------------
def build_bio(identifier: str, timestamps: torch.Tensor, segments: list, b_tag="B"):
    BIO = {"O": 0, "B": 1, "I": 2}
    bio = torch.zeros(len(timestamps), dtype=torch.long)
    timestamp_i = 0
    for segment in segments:
        if segment["start_time"] >= timestamps[-1]:
            print(f"Video {identifier} segment {segment} starts after the end of the pose {timestamps[-1]}")
            continue

        while timestamps[timestamp_i] < segment["start_time"]:
            timestamp_i += 1
        segment_start_i = timestamp_i
        while timestamp_i < (len(timestamps) - 1) and timestamps[timestamp_i] < segment["end_time"]:
            timestamp_i += 1
        segment_end_i = timestamp_i

        bio[segment_start_i] = BIO[b_tag]
        if segment_start_i + 1 < segment_end_i:
            bio[segment_start_i + 1:segment_end_i] = BIO["I"]
    return bio

def build_classes_vectors_cslr(datum) -> tuple:
    """
    Given a datum with:
      - "pose": the loaded pose data,
      - "subtitles": a list of Subtitle objects,
      - "signs": a list of dictionaries each with keys "start", "end", and "text"
    Build segmentation gold labels using the pose timestamps.
    
    It creates two sets of segments:
      - "sentence": built from datum["subtitles"] (converted to dicts with keys "start_time" and "end_time")
      - "sign": built from datum["signs"] (converted similarly from "start" and "end")
    
    Returns a tuple (segments, bio) where segments is a dict with keys "sentence" and "sign"
    and bio is a dict with the corresponding gold label tensors.
    """
    pose = datum["pose"]
    pose_length = len(pose.body.data)
    timestamps = torch.div(torch.arange(0, pose_length), pose.body.fps)
    
    # Build sentence segments from subtitles.
    sentence_segments = [{"start_time": s.start, "end_time": s.end} for s in datum["subtitles"]]
    # Build sign segments from signs.
    sign_segments = [{"start_time": s["start"], "end_time": s["end"]} for s in datum["signs"]]
    
    segments = {
        "sentence": sentence_segments,
        "sign": sign_segments
    }
    
    b_tag = "B"
    bio = {key: build_bio('test', timestamps, seg_list, b_tag=b_tag) for key, seg_list in segments.items()}
    return segments, bio

# ----------------------
# New read_dataset function.
# ----------------------
def read_dataset(data_path: str, pose_path: str, split: str = 'test', fps: int = 25, optical_flow=False, hand_normalization=False) -> list:
    """
    Read the aggregated CSV file (produced by data.py) and group rows by the unique (video_id, example_id)
    combination. For each group:
      - Compute the absolute start_time as the minimum of all start_sub and start_gloss values,
        and the absolute end_time as the maximum of all end_sub and end_gloss values.
      - **subtitles:** Merge rows by unique (video_id, vtt_id) and convert their absolute times to times relative 
         to the computed absolute start_time.
      - **signs:** Merge rows by unique (start_gloss, end_gloss, gloss) and convert their times to be relative 
         to the computed absolute start_time.
    The corresponding pose is loaded from the pose_path directory using a file named <video_id>.pose.
    When reading the pose, we use the absolute times:
      - start_frame = int(absolute_start_time * fps)
      - end_frame = int(absolute_end_time * fps)
    
    Returns a list of datum dictionaries. Each datum contains:
         - "id": a string combining video_id and example_id (e.g. "12345_0")
         - "video_id": the video's ID (string)
         - "example_id": the example identifier (integer)
         - "abs_start": the absolute start time for this example (float)
         - "abs_end": the absolute end time for this example (float)
         - "subtitles": a list of Subtitle objects (with times relative to abs_start)
         - "signs": a list of dictionaries with keys "start", "end", "text" (with times relative to abs_start)
         - "pose": the loaded pose (via Pose.read)
    """
    import os
    from pose_format import Pose
    df = pd.read_csv(data_path)
    # Filter to the desired split using the "split_new" column.
    # df = df[df["split_new"] == split]
    df = df[df["split"] == split]
    dataset = []
    grouped = df.groupby(["video_id", "example_id"])
    for (video_id, example_id), group in grouped:
        group_sorted = group.sort_values(by="start_sub")
        # Compute absolute start time as the minimal value among start_sub and start_gloss.
        abs_start = min(group["start_sub"].min(), group["start_gloss"].min())
        # Compute absolute end time as the maximum value among end_sub and end_gloss.
        abs_end = max(group["end_sub"].max(), group["end_gloss"].max())
        
        # Build subtitles: convert absolute times to relative times (relative to abs_start).
        subtitles = []
        for _, row in group_sorted.iterrows():
            rel_start = row["start_sub"] - abs_start
            rel_end = row["end_sub"] - abs_start
            subtitles.append(Subtitle(start=rel_start, end=rel_end, text=row["subtitle"]))
        
        # Build signs: drop duplicates on (start_gloss, end_gloss, gloss) and convert times relative to abs_start.
        unique_signs = group_sorted.drop_duplicates(subset=["start_gloss", "end_gloss", "gloss"])
        signs = []
        for _, row in unique_signs.iterrows():
            rel_start = row["start_gloss"] - abs_start
            rel_end = row["end_gloss"] - abs_start
            signs.append({"start": rel_start, "end": rel_end, "text": row["gloss"]})
        
        if len(subtitles) == 0:
            continue
        
        # Use absolute times for pose extraction.
        start_time = abs_start
        end_time = abs_end
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        pose_file = os.path.join(pose_path, f"{video_id}.pose")
        if not os.path.exists(pose_file):
            print(f"Warning: Pose file {pose_file} not found for video {video_id}. Skipping datum.")
            continue
        try:
            with open(pose_file, "rb") as f:
                pose = Pose.read(f, start_frame=start_frame, end_frame=end_frame)
                pose = process_pose(pose, optical_flow=optical_flow, hand_normalization=hand_normalization)
        except Exception as e:
            print(f"Error loading pose file {pose_file}: {e}. Skipping video {video_id}.")
            continue
        
        datum = {
            "id": f"{video_id}_{example_id}",
            "video_id": video_id,
            "example_id": example_id,
            "abs_start": abs_start,
            "abs_end": abs_end,
            "subtitles": subtitles,
            "signs": signs,
            "pose": pose
        }
        dataset.append(datum)
    return dataset

# ----------------------
# Main evaluation function.
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='./sign_language_segmentation/dist/model_E4s-1.pth', help="Path to the segmentation model (.pth)")
    parser.add_argument("--data-path", type=str, default="./sign_language_segmentation/src/bsl/data.csv",
                        help="Path to the aggregated CSV annotation file (default: data.csv)")
    parser.add_argument("--pose-path", type=str,
                        default="/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe_v2_refine_face_complexity_2",
                        help="Directory containing pose files named as <video_id>.pose")
    parser.add_argument("--optical-flow", action="store_true")
    parser.add_argument("--hand-normalization", action="store_true")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    args = parser.parse_args()

    # Read dataset: each datum corresponds to a unique (video_id, example_id) group.
    dataset = read_dataset(args.data_path, args.pose_path, split="test", fps=args.fps, optical_flow=args.optical_flow, hand_normalization=args.hand_normalization)
    print(f"Loaded {len(dataset)} data items from {args.data_path}.")

    model = load_model(args.model_path)

    # Initialize metrics for both sentence and sign.
    metrics = {
        'frame_f1_sentence': [],
        'frame_f1_O_sentence': [],
        'frame_precision_O_sentence': [],
        'frame_recall_O_sentence': [],
        'frame_roc_auc_O_sentence': [],
        'frame_f1_sign': [],
        'frame_f1_O_sign': [],
        'frame_precision_O_sign': [],
        'frame_recall_O_sign': [],
        'frame_roc_auc_O_sign': [],
    }

    for datum in tqdm(dataset, desc="Evaluating"):
        print("---------------")
        print(f"Evaluating datum {datum['id']}")
        # print("Subtitles:", datum["subtitles"])
        # print("Signs:", datum["signs"])
        # Build segmentation segments for both sentence and sign using build_classes_vectors_cslr.
        segments, bio = build_classes_vectors_cslr(datum)
        
        probs = predict(model, datum['pose'])
        # Assume that probs is a dictionary with keys "sentence" and "sign"
        for seg_type in ["sentence", "sign"]:
            gold = bio[seg_type]
            pred = probs[seg_type].squeeze()
            metrics[f"frame_f1_{seg_type}"].append(frame_f1(pred, gold, average='macro'))
            if torch.count_nonzero(gold) > 0:
                metrics[f"frame_f1_O_{seg_type}"].append(frame_f1(pred, gold, average=None)[0])
                metrics[f"frame_precision_O_{seg_type}"].append(frame_precision(pred, gold, average=None)[0])
                metrics[f"frame_recall_O_{seg_type}"].append(frame_recall(pred, gold, average=None)[0])
                metrics[f"frame_roc_auc_O_{seg_type}"].append(frame_roc_auc(pred, gold, average=None, multi_class='ovr', labels=[0, 1, 2])[0])
            print(f"Current metrics for {seg_type}:", {k: metrics[k][-1] for k in metrics if seg_type in k})

    # Average metrics over all data items, filtering out NaN values.
    for key, value in metrics.items():
        arr = np.array(value)
        metrics[key] = float(np.nanmean(arr))
    print("Final metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
