import os
import glob
import csv
import re
import argparse
import pandas as pd
import numpy as np
import webvtt

# ----------------------
# Existing processing code
# ----------------------
def process_csv_file(csv_file, split):
    """
    Process a single CSV file to extract both subtitle and gloss annotation info.
    
    Returns:
      - annotations: a list of dictionaries with columns:
          video_id, split, start_sub, end_sub, subtitle, gloss, start_gloss, end_gloss.
      - subtitle_count: number of annotated subtitles (CSV rows) in the file.
      - subtitle_durations: list of durations for each subtitle row (if valid).
    """
    annotations = []
    subtitle_count = 0
    subtitle_durations = []
    video_id = os.path.splitext(os.path.basename(csv_file))[0]
    
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subtitle_count += 1
                start_sub_raw = row.get("start_sub", "").strip()
                end_sub_raw = row.get("end_sub", "").strip()
                try:
                    start_sub_val = float(start_sub_raw) if start_sub_raw != "" else None
                    end_sub_val = float(end_sub_raw) if end_sub_raw != "" else None
                except ValueError:
                    start_sub_val = None
                    end_sub_val = None

                if start_sub_val is not None and end_sub_val is not None:
                    subtitle_durations.append(end_sub_val - start_sub_val)
                
                subtitle = row.get("english sentence", "").strip()
                approx_gloss = row.get("approx gloss sequence", "")
                matches = re.findall(r'(\S+)\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]', approx_gloss)
                for gloss_word, start_gloss_str, end_gloss_str in matches:
                    try:
                        start_gloss = float(start_gloss_str)
                        end_gloss = float(end_gloss_str)
                    except ValueError:
                        continue
                    annotations.append({
                        "video_id": video_id,
                        "split": split,
                        "start_sub": start_sub_val,
                        "end_sub": end_sub_val,
                        "subtitle": subtitle,
                        "gloss": gloss_word,
                        "start_gloss": start_gloss,
                        "end_gloss": end_gloss
                    })
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
    
    return annotations, subtitle_count, subtitle_durations

def process_split_folder(split_folder, split):
    pattern = os.path.join(split_folder, '**', '*.csv')
    csv_files = glob.glob(pattern, recursive=True)
    
    annotations = []
    total_subtitle_count = 0
    all_subtitle_durations = []
    
    for csv_file in csv_files:
        ann, sub_count, sub_durs = process_csv_file(csv_file, split)
        annotations.extend(ann)
        total_subtitle_count += sub_count
        all_subtitle_durations.extend(sub_durs)
        
    return len(csv_files), annotations, total_subtitle_count, all_subtitle_durations

def reassign_splits(df, train_ratio=9, val_ratio=1, test_ratio=1, seed=42):
    unique_episodes = df['video_id'].unique()
    np.random.seed(seed)
    shuffled = np.random.permutation(unique_episodes)
    n = len(shuffled)
    
    total_parts = train_ratio + val_ratio + test_ratio
    n_train = int(n * train_ratio / total_parts)
    n_val = int(n * val_ratio / total_parts)
    n_test = n - n_train - n_val

    video_id_to_split = {}
    for vid in shuffled[:n_train]:
        video_id_to_split[vid] = 'train'
    for vid in shuffled[n_train:n_train+n_val]:
        video_id_to_split[vid] = 'val'
    for vid in shuffled[n_train+n_val:]:
        video_id_to_split[vid] = 'test'
    
    df['split_new'] = df['video_id'].map(video_id_to_split)
    return df

def seconds_to_vtt_time(seconds: float) -> str:
    """
    Convert seconds (float) to a VTT time string of the form "HH:MM:SS.mmm".
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def group_video_subtitles(video_id: str, df_subs: pd.DataFrame, subtitle_dir: str) -> pd.DataFrame:
    """
    For a given video_id and its unique annotated subtitles (DataFrame with columns:
    start_sub, end_sub, subtitle), load the corresponding VTT file from subtitle_dir,
    and assign each subtitle a matching VTT cue index by converting the CSV seconds to
    VTT time strings and comparing them strictly to the cue's start and end strings.
    
    If a match is not found, print the VTT time string derived from the row and raise an error.
    
    Then, iterate over the subtitles (sorted by start_sub) and assign an integer example_id:
      - For the first subtitle, example_id = 0.
      - For each subsequent subtitle, if both the previous and current subtitle have valid 
        vtt_index and the current vtt_index equals the previous vtt_index + 1, then they are grouped together;
        otherwise, a new group is started.
    
    Returns a copy of df_subs with new columns 'vtt_index' and 'example_id'.
    """
    vtt_file = os.path.join(subtitle_dir, f"{video_id}.vtt")
    if not os.path.exists(vtt_file):
        raise FileNotFoundError(f"VTT file {vtt_file} not found for video {video_id}.")
    try:
        cues = list(webvtt.read(vtt_file))
    except Exception as e:
        raise RuntimeError(f"Error reading VTT file {vtt_file}: {e}")
    
    # Build list of cues using the cue's own time strings
    vtt_cues = []
    for idx, cue in enumerate(cues):
        vtt_cues.append({
            "index": idx,
            "start": cue.start,
            "end": cue.end
        })
    # For each subtitle row, convert start_sub and end_sub (seconds) to VTT time string
    vtt_indices = []
    for _, row in df_subs.iterrows():
        vtt_time_start = seconds_to_vtt_time(row['start_sub'])
        vtt_time_end = seconds_to_vtt_time(row['end_sub'])
        matched = None
        for cue in vtt_cues:
            if cue["start"] == vtt_time_start and cue["end"] == vtt_time_end:
                matched = cue["index"]
                break
        if matched is None:
            print(f"No matching VTT cue found for subtitle with times: {vtt_time_start} --> {vtt_time_end}")
            raise ValueError(f"Subtitle time mismatch for video {video_id}: {vtt_time_start} --> {vtt_time_end}")
        vtt_indices.append(matched)
    df_subs = df_subs.copy()
    df_subs['vtt_index'] = vtt_indices

    # Sort subtitles by start_sub (chronological order)
    df_subs = df_subs.sort_values(by=['start_sub']).reset_index(drop=True)
    
    example_ids = []
    current_example = 0
    for i, row in df_subs.iterrows():
        if i == 0:
            example_ids.append(current_example)
        else:
            prev = df_subs.iloc[i-1]
            if row['vtt_index'] == prev['vtt_index'] + 1:
                example_ids.append(current_example)
            else:
                current_example += 1
                example_ids.append(current_example)
    df_subs['example_id'] = example_ids
    return df_subs

def assign_example_ids(df: pd.DataFrame, subtitle_dir: str) -> pd.DataFrame:
    """
    For each video_id in the aggregated DataFrame df, group unique subtitles
    (deduplicated by start_sub, end_sub, subtitle) using the standard:
      - Group only if the subtitles appear consecutively in the CSV (sorted by start_sub)
        and their corresponding VTT cues (matched by converting seconds to VTT time strings)
        are consecutive (i.e. current cue index equals previous cue index + 1).
    
    This function builds a mapping:
         key = (start_sub, end_sub, subtitle)  --> example_id (integer)
         and also records the corresponding vtt_index as vtt_id.
    Then it assigns two new columns in df: 'example_id' and 'vtt_id'.
    """
    df = df.copy()
    example_map = {}
    vtt_map = {}
    for video_id, group in df.groupby('video_id'):
        unique_subs = group[['start_sub', 'end_sub', 'subtitle']].drop_duplicates().reset_index(drop=True)
        unique_subs['start_sub'] = unique_subs['start_sub'].astype(float)
        unique_subs['end_sub'] = unique_subs['end_sub'].astype(float)
        grouped = group_video_subtitles(video_id, unique_subs, subtitle_dir)
        for _, row in grouped.iterrows():
            key = (row['start_sub'], row['end_sub'], row['subtitle'])
            example_map.setdefault(video_id, {})[key] = row['example_id']
            vtt_map.setdefault(video_id, {})[key] = row['vtt_index']
    
    def get_ids(row):
        key = (row['start_sub'], row['end_sub'], row['subtitle'])
        ex_id = example_map.get(row['video_id'], {}).get(key, 0)
        vtt_id = vtt_map.get(row['video_id'], {}).get(key, None)
        return pd.Series([ex_id, vtt_id])
    
    df[['example_id','vtt_id']] = df.apply(get_ids, axis=1)
    return df

# ----------------------
# Helper functions for new requirements
# ----------------------
def read_video_ids(file_path):
    try:
        with open(file_path, "r") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return set()

def count_subtitle_units(video_ids, subtitle_dir):
    total_units = 0
    for vid in video_ids:
        vtt_file = os.path.join(subtitle_dir, f"{vid}.vtt")
        if os.path.exists(vtt_file):
            try:
                cues = list(webvtt.read(vtt_file))
                total_units += len(cues)
            except Exception as e:
                print(f"Error reading VTT file {vtt_file}: {e}")
        else:
            print(f"Warning: VTT file for video {vid} not found in {subtitle_dir}.")
    return total_units

# ----------------------
# Main function: Aggregation, re-splitting, grouping, statistics, and column ordering.
# ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate CSLR annotation statistics, re-split on episode-level, group consecutive subtitles into examples, and store to data.csv."
    )
    parser.add_argument("--cslr_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/continuous_sign_sequences/cslr-raw",
                        help="Directory with CSLR CSV files (searched recursively).")
    parser.add_argument("--subtitle_dir", type=str,
                        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
                        help="Directory containing full VTT files for each episode (named <video_id>.vtt)")
    parser.add_argument("--gap_threshold", type=float, default=2,
                        help="Maximum allowed gap (in seconds) between consecutive subtitles (default: 2)")
    # New arguments for video list txt files
    parser.add_argument('--train_videos_txt', type=str,
                        default='/athenahomes/zifan/subtitle_align/data/bobsl_align_train.txt',
                        help='txt file with one line per video name for training set')
    parser.add_argument('--val_videos_txt', type=str,
                        default='/athenahomes/zifan/subtitle_align/data/bobsl_align_val.txt',
                        help='txt file with one line per video name for validation set')
    parser.add_argument('--test_videos_txt', type=str,
                        default='/athenahomes/zifan/subtitle_align/data/bobsl_align_test.txt',
                        help='txt file with one line per video name for test set')
    
    args = parser.parse_args()

    dataset_types = ['train', 'val', 'test']
    overall_files = 0
    overall_annotations = []
    overall_subtitle_count = 0
    overall_subtitle_durations = []

    # Read video IDs from txt files for each dataset
    video_txt_paths = {
        "train": args.train_videos_txt,
        "val": args.val_videos_txt,
        "test": args.test_videos_txt
    }
    video_ids_from_txt = {ds: read_video_ids(path) for ds, path in video_txt_paths.items()}

    print("Processing CSLR annotations and aggregating data...")
    for dataset in dataset_types:
        split_folder = os.path.join(args.cslr_dir, dataset)
        if not os.path.exists(split_folder):
            print(f"Warning: Sub-folder '{dataset}' not found in {args.cslr_dir}")
            continue
        num_files, annotations, subtitle_count, subtitle_durations = process_split_folder(split_folder, dataset)
        overall_files += num_files
        overall_annotations.extend(annotations)
        overall_subtitle_count += subtitle_count
        overall_subtitle_durations.extend(subtitle_durations)
        
        # Get the total number of episodes from the corresponding txt file
        txt_video_ids = video_ids_from_txt.get(dataset, set())
        txt_episode_count = len(txt_video_ids)
        
        # Count total subtitle units (VTT cues) for episodes listed in the txt file
        total_subtitle_units = count_subtitle_units(txt_video_ids, args.subtitle_dir)
        
        print(f"{dataset.capitalize()} set: {num_files}/{txt_episode_count} episodes, {subtitle_count}/{total_subtitle_units} annotated subtitles, {len(annotations)} gloss annotations")

    df = pd.DataFrame(overall_annotations, columns=[
        "video_id", "split", "start_sub", "end_sub", "subtitle", "gloss", "start_gloss", "end_gloss"
    ])
    total_gloss_annotations = len(df)
    print(f"\nOverall: {overall_files} episodes processed")
    print(f"Total annotated subtitles (CSV rows): {overall_subtitle_count}")
    print(f"Total gloss annotations in aggregated DataFrame: {total_gloss_annotations}")

    # Reassign splits for episode-level (this creates the 'split_new' column)
    df = reassign_splits(df, train_ratio=9, val_ratio=1, test_ratio=1, seed=42)
    print("\nNew episode-level split (9/1/1) distribution:")
    print(df["split_new"].value_counts())

    df = assign_example_ids(df, args.subtitle_dir)

    df["gloss_duration"] = df["end_gloss"] - df["start_gloss"]
    gloss_overall = {
        "Count": len(df),
        "Total Duration (s)": df["gloss_duration"].sum(),
        "Mean Duration (s)": df["gloss_duration"].mean(),
        "Std Duration (s)": df["gloss_duration"].std()
    }
    # Group gloss statistics by original 'split'
    gloss_by_split = df.groupby("split")["gloss_duration"].agg(
        Count="count", **{"Total Duration (s)": "sum", "Mean Duration (s)": "mean", "Std Duration (s)": "std"}
    ).reset_index()

    df_sub = df.drop_duplicates(subset=["video_id", "start_sub", "end_sub", "subtitle"]).copy()
    df_sub["subtitle_duration"] = df_sub["end_sub"] - df_sub["start_sub"]
    subtitle_overall = {
        "Count": len(df_sub),
        "Total Duration (s)": df_sub["subtitle_duration"].sum(),
        "Mean Duration (s)": df_sub["subtitle_duration"].mean(),
        "Std Duration (s)": df_sub["subtitle_duration"].std()
    }
    # Group subtitle statistics by original 'split'
    subtitle_by_split = df_sub.groupby("split")["subtitle_duration"].agg(
        Count="count", **{"Total Duration (s)": "sum", "Mean Duration (s)": "mean", "Std Duration (s)": "std"}
    ).reset_index()

    df_ex = df_sub.groupby(["video_id", "example_id"]).agg(
        start_sub=("start_sub", "min"),
        end_sub=("end_sub", "max"),
        split=("split", "first")  # use original split
    ).reset_index()
    df_ex["example_duration"] = df_ex["end_sub"] - df_ex["start_sub"]
    example_overall = {
        "Count": len(df_ex),
        "Total Duration (s)": df_ex["example_duration"].sum(),
        "Mean Duration (s)": df_ex["example_duration"].mean(),
        "Std Duration (s)": df_ex["example_duration"].std()
    }
    # Group example statistics by original 'split'
    example_by_split = df_ex.groupby("split")["example_duration"].agg(
        Count="count", **{"Total Duration (s)": "sum", "Mean Duration (s)": "mean", "Std Duration (s)": "std"}
    ).reset_index()

    df = df.sort_values(by=["video_id", "start_sub", "start_gloss"])
    desired_order = [
        "video_id", "example_id", "vtt_id", "split", "split_new",
        "start_sub", "end_sub", "subtitle", "gloss", "start_gloss", "end_gloss"
    ]
    df = df[desired_order]

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
    df.to_csv(output_file, index=False)
    print(f"\nFinal aggregated data saved to {output_file}")
    print("\nFirst few rows of final aggregated DataFrame:")
    print(df.head())

    # Print final statistics based on the original 'split'
    gloss_overall_df = pd.DataFrame([{"Split": "Overall", **gloss_overall}])
    gloss_by_split.rename(columns={"split": "Split"}, inplace=True)
    gloss_stats = pd.concat([gloss_overall_df, gloss_by_split], ignore_index=True)
    print("\nGloss Statistics:")
    print(gloss_stats.to_string(index=False))

    subtitle_overall_df = pd.DataFrame([{"Split": "Overall", **subtitle_overall}])
    subtitle_by_split.rename(columns={"split": "Split"}, inplace=True)
    subtitle_stats = pd.concat([subtitle_overall_df, subtitle_by_split], ignore_index=True)
    print("\nSubtitle Statistics:")
    print(subtitle_stats.to_string(index=False))

    example_overall_df = pd.DataFrame([{"Split": "Overall", **example_overall}])
    example_by_split.rename(columns={"split": "Split"}, inplace=True)
    example_stats = pd.concat([example_overall_df, example_by_split], ignore_index=True)
    print("\nExample (Grouped Subtitles) Statistics:")
    print(example_stats.to_string(index=False))

if __name__ == '__main__':
    main()
