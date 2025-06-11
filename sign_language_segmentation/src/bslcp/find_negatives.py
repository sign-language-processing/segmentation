#!/usr/bin/env python3
"""
find_negatives.py

Scan all .vtt subtitle files under a given directory, find “negative” (silent) intervals
that are at least `margin` seconds away from surrounding subtitles, filter by a
minimum and maximum duration, then remove any gaps that overlap a sign‐level segment
(from the SIGN tier of an .eaf in segmentation_dir). Write all remaining gaps
to find_negatives.csv (with duration), and print counts before/after filtering,
summary statistics, plus a random sample of up to 20 gaps in mm:ss format, with progress bars.
Ignore VTTs without a matching EAF file.
"""

import os
import re
import csv
import argparse
import statistics
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm


def parse_timestamp(ts):
    """
    Parse a WebVTT timestamp "HH:MM:SS.mmm" (or with comma) into total seconds (float).
    """
    hours, minutes, rest = ts.replace(',', '.').split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def format_mmss(seconds):
    """
    Convert a float seconds timestamp to "MM:SS" (dropping fractional part).
    """
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def find_vtt_files(root_dir):
    """
    Walk root_dir recursively and return a list of all *.vtt file paths.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.vtt'):
                paths.append(os.path.join(dirpath, fn))
    return paths


def extract_subtitles(vtt_path):
    """
    Read a .vtt file and return a list of (start_sec, end_sec) tuples, in order.
    """
    times = []
    timing_re = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})')
    with open(vtt_path, encoding='utf-8') as f:
        for line in f:
            m = timing_re.match(line.strip())
            if m:
                t0, t1 = m.groups()
                times.append((parse_timestamp(t0), parse_timestamp(t1)))
    return times


def load_sign_segments(video_id, segmentation_dir, cache):
    """
    Load all sign segments (start_sec, end_sec) from the SIGN tier of video_id.eaf,
    caching results in `cache`.
    """
    if video_id in cache:
        return cache[video_id]

    path = os.path.join(segmentation_dir, f"{video_id}.eaf")
    segments = []
    if os.path.isfile(path):
        tree = ET.parse(path)
        root = tree.getroot()
        # build timeline mapping TIME_SLOT_ID -> seconds
        timeline = {}
        to = root.find('TIME_ORDER')
        if to is not None:
            for ts in to.findall('TIME_SLOT'):
                ts_id = ts.attrib['TIME_SLOT_ID']
                tv = ts.attrib.get('TIME_VALUE')
                if tv is not None:
                    timeline[ts_id] = int(tv) / 1000.0

        # find the SIGN tier
        for tier in root.findall('TIER'):
            if tier.attrib.get('TIER_ID') == 'SIGN':
                for ann in tier.findall('.//ALIGNABLE_ANNOTATION'):
                    ref1 = ann.attrib.get('TIME_SLOT_REF1')
                    ref2 = ann.attrib.get('TIME_SLOT_REF2')
                    if ref1 in timeline and ref2 in timeline:
                        segments.append((timeline[ref1], timeline[ref2]))
                break

    cache[video_id] = segments
    return segments


def main():
    parser = argparse.ArgumentParser(description="Find long silent gaps in .vtt subtitles")
    parser.add_argument(
        "--subtitle_dir",
        default="/users/zifan/BOBSL/v1.4/manual_annotations/signing_aligned_subtitles",
        help="Root directory to search for .vtt files"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10.0,
        help="Seconds to trim off each side of a gap (so gap must exceed 2×margin)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum duration (seconds) of gap to include"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum duration (seconds) of gap to include"
    )
    parser.add_argument(
        "--segmentation_dir",
        default="/users/zifan/BOBSL/derivatives/segmentation/E4s-1_30_50",
        help="Directory containing <video_id>.eaf files with a SIGN tier"
    )
    args = parser.parse_args()

    # Prepare output CSV in the same folder as this script:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(script_dir, "find_negatives.csv")

    # Step 1: gather VTT paths
    vtt_paths = find_vtt_files(args.subtitle_dir)
    gap_records = []
    skip_count = 0
    # Scan each VTT, skipping those without matching EAF
    for vtt_path in tqdm(vtt_paths, desc="Scanning VTT files", unit="file"):
        video_id = os.path.splitext(os.path.basename(vtt_path))[0]
        eaf_path = os.path.join(args.segmentation_dir, f"{video_id}.eaf")
        if not os.path.isfile(eaf_path):
            skip_count += 1
            continue
        subs = extract_subtitles(vtt_path)
        for (_, prev_end), (next_start, _) in zip(subs, subs[1:]):
            raw_gap = next_start - prev_end
            if raw_gap > 2 * args.margin:
                start_gap = prev_end + args.margin
                end_gap   = next_start - args.margin
                duration  = end_gap - start_gap
                if args.min_duration <= duration <= args.max_duration:
                    gap_records.append((video_id, start_gap, end_gap, duration))

    before_count = len(gap_records)
    print(f"Ignored {skip_count} VTT files without matching EAF.")

    # Step 2: filter out gaps overlapping a SIGN segment
    seg_cache = {}
    filtered_records = []
    for vid, start, end, dur in tqdm(gap_records, desc="Filtering sign segments", unit="gap"):
        segments = load_sign_segments(vid, args.segmentation_dir, seg_cache)
        if not any(s < end and e > start for s, e in segments):
            filtered_records.append((vid, start, end, dur))

    after_count = len(filtered_records)

    # Step 3: write filtered gaps to CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_id", "start_gap", "end_gap", "duration"])
        for vid, start, end, dur in tqdm(filtered_records, desc="Writing CSV", unit="row"):
            writer.writerow([vid, f"{start:.3f}", f"{end:.3f}", f"{dur:.3f}"])

    # Step 4: print counts, stats, and sample
    print(f"\nGaps before segmentation filter: {before_count}")
    print(f"Gaps after segmentation filter:  {after_count}")

    if filtered_records:
        durations = [rec[3] for rec in filtered_records]
        count  = len(durations)
        mean   = statistics.mean(durations)
        median = statistics.median(durations)
        std    = statistics.stdev(durations) if count > 1 else 0.0
        mn     = min(durations)
        mx     = max(durations)

        print("\nSummary statistics for remaining gap durations (seconds):")
        print(f"  Count:               {count}")
        print(f"  Mean:                {mean:.3f}")
        print(f"  Median:              {median:.3f}")
        print(f"  Standard deviation:  {std:.3f}")
        print(f"  Minimum:             {mn:.3f}")
        print(f"  Maximum:             {mx:.3f}")

        sample_size = min(20, count)
        sample = random.sample(filtered_records, sample_size)
        print(f"\nRandom sample of {sample_size} gaps (start – end in mm:ss):")
        for vid, start, end, _ in sample:
            print(f"  {vid}: {format_mmss(start)} – {format_mmss(end)}")
    else:
        print("\nNo gaps remain after segmentation filtering; check your parameters or data.")

if __name__ == "__main__":
    main()
