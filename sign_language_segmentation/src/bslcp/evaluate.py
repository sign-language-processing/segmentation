import os
import pandas as pd
import numpy as np
import torch
import argparse
import random
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

def build_classes_vectors(datum) -> tuple:
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

def read_dataset(data_path: str, pose_path: str, split: str = 'test', fps: int = 25, optical_flow=False, hand_normalization=False) -> list:
    """
    Read the aggregated CSV file and group rows by unique video_id.

    For each group:
      - "subtitles" is always an empty list.
      - "signs" are created by dropping duplicates on (start_gloss, end_gloss, gloss),
        using start_gloss and end_gloss directly (no absolute/relative time adjustment).

    The corresponding pose is loaded fully (no frame range selection).

    Returns:
      - A list of datum dictionaries with keys:
          "id", "video_id", "example_id" (always 0), "subtitles" (empty), "signs", "pose"
    """
    df = pd.read_csv(data_path)
    df = df[df["split"] == split]

    dataset = []
    grouped = df.groupby("video_id")

    for video_id, group in grouped:
        # Build signs: drop duplicates and use start_gloss, end_gloss directly
        unique_signs = group.drop_duplicates(subset=["start_gloss", "end_gloss", "gloss"])
        signs = [
            {
                "start": row["start_gloss"],
                "end": row["end_gloss"],
                "text": row["gloss"]
            }
            for _, row in unique_signs.iterrows()
        ]

        if len(signs) == 0:
            continue  # Skip if no signs

        pose_file = os.path.join(pose_path, f"{video_id}.pose")
        if not os.path.exists(pose_file):
            print(f"Warning: Pose file {pose_file} not found for video {video_id}. Skipping datum.")
            continue

        try:
            with open(pose_file, "rb") as f:
                pose = Pose.read(f)  # Read entire pose, no start_frame / end_frame
                pose = process_pose(pose, optical_flow=optical_flow, hand_normalization=hand_normalization)
        except Exception as e:
            print(f"Error loading pose file {pose_file}: {e}. Skipping video {video_id}.")
            continue

        datum = {
            "id": video_id,
            "video_id": video_id,
            "subtitles": [],       # Always empty
            "signs": signs,
            "pose": pose
        }
        dataset.append(datum)

    return dataset

def read_dataset_segmented(
    data_path: str,
    pose_path: str,
    split: str = 'test',
    fps: int = 25,
    optical_flow=False,
    hand_normalization=False,
    max_frame: int = 10000,
    augment_pseudo_label=False,
    augment_bobsl_negative=False,
) -> list:
    """
    - If split in ['test','val'], or split=='train' with no augmentation flags:
        • One example per sentence (grouped by sentence_id).
    - If split=='train' and augment_pseudo_label=True:
        • Video‑level segmentation into max_frame chunks (gloss‑boundary adjusted).
    - If split=='train' and augment_bobsl_negative=True:
        • Video‑level example by concatenating sentence‑level segments with
          random‑gap segments before the first, between sentences, and after the last,
          sampling from find_negatives.csv.

    In all modes:
      - subtitles = []
      - signs times relative to segment/sentence start
      - each Pose.read and process_pose called once per final assembled pose
    """
    # reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    df = pd.read_csv(data_path)
    df = df[df["split"] == split]
    dataset = []

    # Sentence‐level examples for test/val or train without aug flags
    if split in ['test', 'val'] or (split == 'train' and not augment_pseudo_label and not augment_bobsl_negative):
        df_sent = df.dropna(subset=["sentence_id"])
        for video_id in tqdm(df_sent["video_id"].unique(), desc="Sentences", unit="video"):
            grp = df_sent[df_sent["video_id"] == video_id]
            pose_file = os.path.join(pose_path, f"{video_id}.pose")
            if not os.path.exists(pose_file):
                tqdm.write(f"Warning: no pose for video {video_id}, skipping.")
                continue
            with open(pose_file, "rb") as f:
                for sent_id, sent_grp in grp.groupby("sentence_id"):
                    start_s = sent_grp["start_sentence"].min()
                    end_s   = sent_grp["end_sentence"].max()
                    sf = int(start_s * fps)
                    ef = int(end_s   * fps)

                    signs = [
                        {"start": row.start_gloss - start_s,
                         "end":   row.end_gloss   - start_s,
                         "text":  row.gloss}
                        for _, row in sent_grp.iterrows()
                    ]
                    if not signs:
                        continue

                    f.seek(0)
                    segment_pose = Pose.read(f, start_frame=sf, end_frame=ef)
                    segment_pose = process_pose(segment_pose, optical_flow, hand_normalization)

                    dataset.append({
                        "id":         f"{video_id}_sent{sent_id}",
                        "video_id":   video_id,
                        "example_id": sent_id,
                        "subtitles":  [],
                        "signs":      signs,
                        "pose":       segment_pose
                    })

    elif split=='train' and augment_bobsl_negative:
        # load gap definitions
        gaps_csv = os.path.join(os.path.dirname(__file__), 'find_negatives.csv')
        gaps_df = pd.read_csv(gaps_csv, dtype={'video_id': str})

        df_sent = df.dropna(subset=["sentence_id"])
        for video_id in tqdm(df_sent["video_id"].unique(), desc="Sentences (bobsl_neg)", unit="video"):
            grp = df_sent[df_sent["video_id"] == video_id]
            pose_file = os.path.join(pose_path, f"{video_id}.pose")
            if not os.path.exists(pose_file):
                tqdm.write(f"Warning: no pose for video {video_id}, skipping.")
                continue

            with open(pose_file, "rb") as f:
                for sent_id, sent_grp in grp.groupby("sentence_id"):
                    # sentence segment
                    start_s = sent_grp["start_sentence"].min()
                    end_s   = sent_grp["end_sentence"].max()
                    sf, ef  = int(start_s*fps), int(end_s*fps)

                    # build relative signs
                    rel_signs = [
                        {"start": row.start_gloss - start_s,
                        "end":   row.end_gloss   - start_s,
                        "text":  row.gloss}
                        for _, row in sent_grp.iterrows()
                    ]
                    if not rel_signs:
                        continue

                    # read core sentence pose
                    f.seek(0)
                    core_pose = Pose.read(f, start_frame=sf, end_frame=ef)

                    # sample one gap
                    gap = gaps_df.sample(1).iloc[0]
                    g_sf    = int(gap.start_gap * fps)
                    g_ef    = int(gap.end_gap   * fps)
                    gap_dur = gap.duration
                    gap_vid = gap.video_id
                    gap_path = os.path.join(
                        "/users/zifan/BOBSL/derivatives/video_features/"
                        "mediapipe_v2_refine_face_complexity_2",
                        f"{gap_vid}.pose"
                    )
                    with open(gap_path, "rb") as gf:
                        gap_pose = Pose.read(gf, start_frame=g_sf, end_frame=g_ef)

                    # 4a) original example
                    pose_orig = core_pose.copy()
                    pose_orig = process_pose(pose_orig, optical_flow, hand_normalization)
                    dataset.append({
                        "id":         f"{video_id}_sent{sent_id}_orig",
                        "video_id":   video_id,
                        "example_id": f"{sent_id}_orig",
                        "subtitles":  [],
                        "signs":      rel_signs,
                        "pose":       pose_orig
                    })

                    # randomly pick one augmentation of the remaining three
                    choice = random.choice(['app', 'pre', 'rep'])
                    if choice == 'app':
                        # append gap
                        pose_app = core_pose.copy()
                        pose_app.body.data = np.vstack((pose_app.body.data, gap_pose.body.data))
                        pose_app.body.confidence = np.vstack((pose_app.body.confidence, gap_pose.body.confidence))
                        pose_app = process_pose(pose_app, optical_flow, hand_normalization)
                        dataset.append({
                            "id":         f"{video_id}_sent{sent_id}_app",
                            "video_id":   video_id,
                            "example_id": f"{sent_id}_app",
                            "subtitles":  [],
                            "signs":      rel_signs,
                            "pose":       pose_app
                        })
                    elif choice == 'pre':
                        # prepend gap
                        pose_pre = gap_pose.copy()
                        pose_pre.body.data = np.vstack((pose_pre.body.data, core_pose.body.data))
                        pose_pre.body.confidence = np.vstack((pose_pre.body.confidence, core_pose.body.confidence))
                        # shift signs by gap duration
                        shifted = [
                            {"start": s["start"] + gap_dur,
                            "end":   s["end"]   + gap_dur,
                            "text":  s["text"]}
                            for s in rel_signs
                        ]
                        pose_pre = process_pose(pose_pre, optical_flow, hand_normalization)
                        dataset.append({
                            "id":         f"{video_id}_sent{sent_id}_pre",
                            "video_id":   video_id,
                            "example_id": f"{sent_id}_pre",
                            "subtitles":  [],
                            "signs":      shifted,
                            "pose":       pose_pre
                        })
                    else:
                        # replace with gap only
                        pose_rep = gap_pose.copy()
                        pose_rep = process_pose(pose_rep, optical_flow, hand_normalization)
                        dataset.append({
                            "id":         f"{video_id}_sent{sent_id}_rep",
                            "video_id":   video_id,
                            "example_id": f"{sent_id}_rep",
                            "subtitles":  [],
                            "signs":      [],
                            "pose":       pose_rep
                        })

    # Video‐level segmentation for train + pseudo‐label augmentation
    elif split == 'train' and augment_pseudo_label:
        for video_id in tqdm(df["video_id"].unique(), desc="Videos (pseudo)", unit="video"):
            group = df[df["video_id"] == video_id]
            # build unique gloss list
            unique = (
                group
                .drop_duplicates(subset=["start_gloss", "end_gloss", "gloss"])
                .sort_values("start_gloss")
            )
            gloss_list = [
                {
                    "start_time":  row.start_gloss,
                    "end_time":    row.end_gloss,
                    "start_frame": int(row.start_gloss * fps),
                    "end_frame":   int(row.end_gloss   * fps),
                    "text":        row.gloss
                }
                for _, row in unique.iterrows()
            ]
            if not gloss_list:
                continue

            pose_file = os.path.join(pose_path, f"{video_id}.pose")
            if not os.path.exists(pose_file):
                tqdm.write(f"Warning: no pose for video {video_id}, skipping.")
                continue

            with open(pose_file, "rb") as f:
                full_pose = Pose.read(f)
                total_frames = full_pose.body.data.shape[0]

                # segment by max_frame, snapping to gloss boundaries
                segments = []
                seg_start = 0
                while seg_start < total_frames:
                    ideal = seg_start + max_frame
                    if ideal >= total_frames:
                        seg_end = total_frames
                    else:
                        overlap = next(
                            (g for g in gloss_list
                             if g["start_frame"] < ideal < g["end_frame"]),
                            None
                        )
                        seg_end = overlap["start_frame"] if overlap else ideal

                    seg_signs = [
                        g for g in gloss_list
                        if g["start_frame"] >= seg_start and g["end_frame"] <= seg_end
                    ]
                    segments.append((seg_start, seg_end, seg_signs))
                    seg_start = seg_end

                # read each segment
                for idx, (sf, ef, seg_signs) in enumerate(segments):
                    seg_start_t = sf / fps
                    rel_signs = [
                        {
                            "start": g["start_time"] - seg_start_t,
                            "end":   g["end_time"]   - seg_start_t,
                            "text":  g["text"]
                        }
                        for g in seg_signs
                    ]

                    f.seek(0)
                    segment_pose = Pose.read(f, start_frame=sf, end_frame=ef)
                    segment_pose = process_pose(segment_pose, optical_flow, hand_normalization)

                    dataset.append({
                        "id":         f"{video_id}_{idx}",
                        "video_id":   video_id,
                        "example_id": idx,
                        "subtitles":  [],
                        "signs":      rel_signs,
                        "pose":       segment_pose
                    })

    # # 3) BobSL negative: concat sentences + gaps before, between, after
    # elif split=='train' and augment_bobsl_negative:
    #     # filter out rows without sentence_id
    #     df_sent = df.dropna(subset=["sentence_id"])
        
    #     # for gap replacements
    #     gaps_csv = os.path.join(os.path.dirname(__file__), 'find_negatives.csv')
    #     gaps_df = pd.read_csv(gaps_csv, dtype={'video_id': str})
    #     gap_pose_dir = "/users/zifan/BOBSL/derivatives/video_features/mediapipe_v2_refine_face_complexity_2"
        
    #     for video_id in tqdm(df_sent["video_id"].unique(), desc="Videos (bobsl_neg)", unit="video"):
    #         group = df_sent[df_sent["video_id"] == video_id]
    #         pose_file = os.path.join(pose_path, f"{video_id}.pose")
    #         if not os.path.exists(pose_file):
    #             tqdm.write(f"Warning: no pose for video {video_id}, skipping.")
    #             continue

    #         with open(pose_file, "rb") as f:
    #             full_pose = Pose.read(f)
    #             total_frames = full_pose.body.data.shape[0]

    #         # collect sentence intervals (annotated)
    #         sent_intervals = []
    #         for _, sent_grp in group.groupby("sentence_id"):
    #             start_s = sent_grp["start_sentence"].min()
    #             end_s = sent_grp["end_sentence"].max()
    #             sent_intervals.append((int(start_s * fps), int(end_s * fps)))
    #         sent_intervals.sort()

    #         # compute gap intervals: spans not covered by any sentence
    #         gap_intervals = []
    #         prev_end = 0
    #         for (s0, s1) in sent_intervals:
    #             if s0 > prev_end:
    #                 gap_intervals.append((prev_end, s0))
    #             prev_end = s1
    #         if prev_end < total_frames:
    #             gap_intervals.append((prev_end, total_frames))

    #         # print(video_id)
    #         # print([(s/25, e/25) for s, e in gap_intervals])
    #         # exit()

    #         # segment by max_frame, snapping to gloss boundaries
    #         unique = (
    #             group
    #             .drop_duplicates(subset=["start_gloss", "end_gloss", "gloss"])
    #             .sort_values("start_gloss")
    #         )
    #         gloss_list = [
    #             {
    #                 "start_time":  row.start_gloss,
    #                 "end_time":    row.end_gloss,
    #                 "start_frame": int(row.start_gloss * fps),
    #                 "end_frame":   int(row.end_gloss * fps),
    #                 "text":        row.gloss
    #             }
    #             for _, row in unique.iterrows()
    #         ]
    #         if not gloss_list:
    #             continue

    #         segments = []
    #         seg_start = 0
    #         while seg_start < total_frames:
    #             ideal = seg_start + max_frame
    #             if ideal >= total_frames:
    #                 seg_end = total_frames
    #             else:
    #                 overlap = next(
    #                     (g for g in gloss_list if g["start_frame"] < ideal < g["end_frame"]),
    #                     None
    #                 )
    #                 seg_end = overlap["start_frame"] if overlap else ideal

    #             seg_signs = [
    #                 g for g in gloss_list
    #                 if g["start_frame"] >= seg_start and g["end_frame"] <= seg_end
    #             ]
    #             segments.append((seg_start, seg_end, seg_signs))
    #             seg_start = seg_end

    #         # process each segment with gap replacement
    #         for idx, (sf, ef, seg_signs) in enumerate(segments):
    #             # initial read of segment
    #             f.seek(0)
    #             segment_pose = Pose.read(f, start_frame=sf, end_frame=ef)

    #             # replace each gap interval within this segment
    #             prev = sf
    #             for (s0, s1) in sent_intervals:
    #                 if s0 > prev and prev < ef:
    #                     g0 = prev
    #                     g1 = min(s0, ef)
    #                     # sample a random gap
    #                     gap = gaps_df.sample(1).iloc[0]
    #                     t_video = gap["video_id"]
    #                     t_sf = int(gap.start_gap * fps)
    #                     t_ef = int(gap.end_gap * fps)
    #                     gp_file = os.path.join(gap_pose_dir, f"{t_video}.pose")
    #                     with open(gp_file, "rb") as gf:
    #                         tgt = Pose.read(gf, start_frame=t_sf, end_frame=t_ef)
    #                     # local indices
    #                     o0 = g0 - sf
    #                     o1 = g1 - sf
    #                     segment_pose.body.data[o0:o1] = tgt.body.data[:o1-o0]
    #                     segment_pose.body.confidence[o0:o1] = tgt.body.confidence[:o1-o0]
    #                     prev = g1
    #                 prev = max(prev, s1)
    #             # final gap after last sentence
    #             if prev < ef:
    #                 g0 = prev
    #                 g1 = ef
    #                 gap = gaps_df.sample(1).iloc[0]
    #                 t_video = gap["video_id"]
    #                 t_sf = int(gap.start_gap * fps)
    #                 t_ef = int(gap.end_gap * fps)
    #                 gp_file = os.path.join(gap_pose_dir, f"{t_video}.pose")
    #                 with open(gp_file, "rb") as gf:
    #                     tgt = Pose.read(gf, start_frame=t_sf, end_frame=t_ef)
    #                 o0 = g0 - sf
    #                 o1 = g1 - sf
    #                 segment_pose.body.data[o0:o1] = tgt.body.data[:o1-o0]
    #                 segment_pose.body.confidence[o0:o1] = tgt.body.confidence[:o1-o0]

    #             # finalize signs
    #             seg_start_t = sf / fps
    #             rel_signs = [
    #                 {"start": g["start_time"] - seg_start_t,
    #                     "end":   g["end_time"]   - seg_start_t,
    #                     "text":  g["text"]}
    #                 for g in seg_signs
    #             ]

    #             # single process
    #             segment_pose = process_pose(segment_pose, optical_flow, hand_normalization)

    #             dataset.append({
    #                 "id":         f"{video_id}_{idx}",
    #                 "video_id":   video_id,
    #                 "example_id": idx,
    #                 "subtitles":  [],
    #                 "signs":      rel_signs,
    #                 "pose":       segment_pose
    #             })

    # # 3) BobSL negative: concat sentences + gaps before, between, after
    # elif split=='train' and augment_bobsl_negative:
    #     df_sent = df.dropna(subset=["sentence_id"])
    #     for video_id in tqdm(df_sent["video_id"].unique(), desc="Videos (bobsl_neg)", unit="video"):  # noqa
    #         grp = df_sent[df_sent["video_id"] == video_id]
    #         pose_file = os.path.join(pose_path, f"{video_id}.pose")
    #         if not os.path.exists(pose_file):
    #             tqdm.write(f"Warning: no pose for video {video_id}, skipping.")
    #             continue

    #         # Collect sentence entries
    #         sentence_entries = []
    #         with open(pose_file, "rb") as f:
    #             for sent_id, sent_grp in grp.groupby("sentence_id"):
    #                 start_s = sent_grp["start_sentence"].min()
    #                 end_s   = sent_grp["end_sentence"].max()
    #                 sf, ef = int(start_s*fps), int(end_s*fps)
    #                 signs = [
    #                     {"start": row.start_gloss - start_s,
    #                      "end":   row.end_gloss   - start_s,
    #                      "text":  row.gloss}
    #                     for _, row in sent_grp.iterrows()
    #                 ]
    #                 if not signs:
    #                     continue
    #                 f.seek(0)
    #                 pose_seg = Pose.read(f, start_frame=sf, end_frame=ef)
    #                 dur = end_s - start_s
    #                 sentence_entries.append((signs, pose_seg, dur))

    #         if not sentence_entries:
    #             continue

    #         # Assemble video-level pose and signs
    #         video_signs = []
    #         cum_dur = 0.0
    #         video_pose = None

    #         # Insert gap before first sentence, then each sentence + gap, then last sentence
    #         for i, (signs, pose_seg, dur) in enumerate(sentence_entries):
    #             # # gap insertion before every sentence (including first)
    #             # gap = gaps_df.sample(1).iloc[0]
    #             # gap_vid = gap.video_id  # already a string from CSV dtype
    #             # gap_sf = int(gap.start_gap * fps)
    #             # gap_ef = int(gap.end_gap   * fps)
    #             # gap_dur = gap.duration

    #             # gap_pose_file = os.path.join(
    #             #     "/users/zifan/BOBSL/derivatives/video_features/"
    #             #     "mediapipe_v2_refine_face_complexity_2",
    #             #     f"{gap_vid}.pose"
    #             # )
    #             # with open(gap_pose_file, "rb") as gf:
    #             #     gap_pose = Pose.read(gf, start_frame=gap_sf, end_frame=gap_ef)

    #             # # print(gap_vid)
    #             # # print(gap.start_gap)
    #             # # print(gap.end_gap)

    #             # # concat gap
    #             # if video_pose is None:
    #             #     video_pose = gap_pose
    #             # else:
    #             #     # print(video_pose.body.data.shape)
    #             #     # print(gap_pose.body.data.shape)
    #             #     video_pose.body.data = np.vstack((video_pose.body.data, gap_pose.body.data))
    #             #     video_pose.body.confidence = np.vstack((video_pose.body.confidence, gap_pose.body.confidence))
    #             # cum_dur += gap_dur

    #             # concat sentence pose
    #             if video_pose is None:
    #                 video_pose = pose_seg
    #             else:
    #                 # print(video_pose.body.data.shape)
    #                 # print(pose_seg.body.data.shape)
    #                 video_pose.body.data = np.vstack((video_pose.body.data, pose_seg.body.data))
    #                 video_pose.body.confidence = np.vstack((video_pose.body.confidence, pose_seg.body.confidence))

    #             # append signs (offset)
    #             for s in signs:
    #                 video_signs.append({
    #                     "start": s["start"] + cum_dur,
    #                     "end":   s["end"]   + cum_dur,
    #                     "text":  s["text"]
    #                 })
    #             cum_dur += dur

    #         # # also insert a gap after the last sentence
    #         # gap = gaps_df.sample(1).iloc[0]
    #         # gap_vid = str(gap.video_id)
    #         # gap_sf = int(gap.start_gap * fps)
    #         # gap_ef = int(gap.end_gap   * fps)
    #         # gap_dur = gap.duration
    #         # gap_pose_file = os.path.join(
    #         #     "/users/zifan/BOBSL/derivatives/video_features/"
    #         #     "mediapipe_v2_refine_face_complexity_2",
    #         #     f"{gap_vid}.pose"
    #         # )
    #         # with open(gap_pose_file, "rb") as gf:
    #         #     gap_pose = Pose.read(gf, start_frame=gap_sf, end_frame=gap_ef)
    #         # video_pose.body.data = np.vstack((video_pose.body.data, gap_pose.body.data))
    #         # video_pose.body.confidence = np.vstack((video_pose.body.confidence, gap_pose.body.confidence))
    #         # cum_dur += gap_dur

    #         print(video_id)
    #         print(len(sentence_entries))
    #         print(len(video_signs))
    #         print(video_pose.body.data.shape)
    #         print(video_signs)

    #         # def pose_visualize(pose):
    #         #     from pose_format.pose_visualizer import PoseVisualizer
    #         #     pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    #         #     # Visualize
    #         #     v = PoseVisualizer(pose, thickness=1)

    #         #     output_path = "/users/zifan/segmentation/sign_language_segmentation/src/bslcp/{}.mp4".format(video_id)
    #         #     v.save_video(output_path, v.draw())

    #         #     return output_path

    #         # pose_visualize(video_pose)

    #         # exit()

    #         # final process
    #         video_pose = process_pose(video_pose, optical_flow, hand_normalization)

    #         dataset.append({
    #             "id":         f"{video_id}_bobsl_neg",
    #             "video_id":   video_id,
    #             "example_id": 0,
    #             "subtitles":  [],
    #             "signs":      video_signs,
    #             "pose":       video_pose
    #         })

    return dataset

# ----------------------
# Main evaluation function.
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='./sign_language_segmentation/dist/model_E4s-1.pth', help="Path to the segmentation model (.pth)")
    parser.add_argument("--data-path", type=str, default="./sign_language_segmentation/src/bslcp/data_merged.csv",
                        help="Path to the aggregated CSV annotation file (default: data.csv)")
    parser.add_argument("--pose-path", type=str,
                        default="/scratch/shared/beegfs/zifan/bsl-corpus/mediapipe",
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
        # 'frame_f1_sentence': [],
        # 'frame_f1_O_sentence': [],
        # 'frame_precision_O_sentence': [],
        # 'frame_recall_O_sentence': [],
        # 'frame_roc_auc_O_sentence': [],
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
        # Build segmentation segments for both sentence and sign using build_classes_vectors.
        segments, bio = build_classes_vectors(datum)
        
        probs = predict(model, datum['pose'])
        # Assume that probs is a dictionary with keys "sentence" and "sign"
        # for seg_type in ["sentence", "sign"]:
        for seg_type in ["sign"]:
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
