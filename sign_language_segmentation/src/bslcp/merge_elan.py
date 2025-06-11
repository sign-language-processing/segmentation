#!/usr/bin/env python3
import os
import argparse
import csv
from glob import glob
from collections import namedtuple, defaultdict
import pandas as pd
from lxml import etree as ET
from tabulate import tabulate
import pickle

TimeSlot = namedtuple("TimeSlot", ["ts_id", "time"])

def parse_eaf(path):
    tree = ET.parse(path)
    root = tree.getroot()
    to = root.find("TIME_ORDER")
    timeslots = {
        ts.get("TIME_SLOT_ID"): TimeSlot(ts.get("TIME_SLOT_ID"),
                                         int(ts.get("TIME_VALUE")))
        for ts in to.findall("TIME_SLOT")
    }
    tiers = {tier.get("TIER_ID"): tier for tier in root.findall("TIER")}
    return root, timeslots, tiers

def get_intervals(tier, ts_map):
    ivals = []
    for ann in tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
        s = ts_map[ann.get("TIME_SLOT_REF1")].time
        e = ts_map[ann.get("TIME_SLOT_REF2")].time
        ivals.append((s, e))
    return ivals

def overlaps(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])

def load_info():
    here = os.path.dirname(os.path.abspath(__file__))
    return pickle.load(open(os.path.join(here, "info.pkl"), "rb"))

def summary_stats(df, info):
    # compute frames per unit
    df['frames'] = (df['end_gloss'] - df['start_gloss']) * 25
    # total frames per video
    frames_video = df.groupby('video_id')['frames'].sum()
    # glosses per video = count of rows per video
    glosses_video = df.groupby('video_id').size()

    stats = {}
    for split_idx, split in zip([0,1,2], ['train','val','test']):
        sdf = df[df['split']==split]
        stats[split] = {
            'avg_frames_per_sign':
                f"{sdf['frames'].mean():.1f} ± {sdf['frames'].std():.1f}",
            'avg_frames_per_video':
                f"{frames_video.loc[sdf['video_id'].unique()].mean():.1f} ± "
                f"{frames_video.loc[sdf['video_id'].unique()].std():.1f}",
            'avg_glosses_per_video':
                f"{glosses_video.loc[sdf['video_id'].unique()].mean():.1f} ± "
                f"{glosses_video.loc[sdf['video_id'].unique()].std():.1f}",
            'total_videos': sdf['video_id'].nunique(),
            'total_signers': len({
                s for s, sp in zip(info['videos']['signer'], info['videos']['split'])
                if sp == split_idx
            }),
            # unique glosses in this split
            'total_glosses': sdf['gloss'].nunique()
        }
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--segmentation_path",
        default="/users/zifan/BSL-Corpus/derivatives/segmentation/E4s-1_60_50",
        help="ELANs with SIGN+SENTENCE tiers"
    )
    p.add_argument(
        "--annotation_path",
        default="/users/zifan/BSL-Corpus/derivatives/annotation",
        help="ELANs with GLOSS+SENTENCE tiers"
    )
    p.add_argument(
        "--output_path",
        default="/users/zifan/BSL-Corpus/derivatives/annotation_merged",
        help="Where to write merged ELANs and data_merged.csv"
    )
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    # 1) Merge ELAN tiers
    for ann_file in glob(os.path.join(args.annotation_path, "*.eaf")):
        vid = os.path.splitext(os.path.basename(ann_file))[0]
        seg_file = os.path.join(args.segmentation_path, f"{vid}.eaf")
        out_eaf = os.path.join(args.output_path, f"{vid}.eaf")

        if not os.path.exists(seg_file):
            print(f"Warning: no segmentation EAF for {vid}, copying annotation only.")
            os.system(f"cp {ann_file} {out_eaf}")
            continue

        ann_root, ann_ts, ann_tiers = parse_eaf(ann_file)
        seg_root, seg_ts, seg_tiers = parse_eaf(seg_file)

        # import segmentation timeslots
        to_elem = ann_root.find("TIME_ORDER")
        time_map = {ts.time: ts.ts_id for ts in ann_ts.values()}
        for slot in seg_ts.values():
            if slot.time not in time_map:
                nid = f"ts{len(time_map)+1}"
                ET.SubElement(to_elem, "TIME_SLOT",
                              TIME_SLOT_ID=nid, TIME_VALUE=str(slot.time))
                time_map[slot.time] = nid
        ann_ts = {ts_id: TimeSlot(ts_id, t)
                  for t, ts_id in time_map.items()}

        gloss_tier    = ann_tiers.get("GLOSS")
        sentence_tier = ann_tiers.get("SENTENCE")
        sent_iv = get_intervals(sentence_tier, ann_ts) if sentence_tier else []

        # PSEUDO GLOSS tier
        pg = ET.SubElement(ann_root, "TIER",
                           TIER_ID="PSEUDO GLOSS",
                           LINGUISTIC_TYPE_REF="default-lt")
        aid = 1
        if gloss_tier:
            for ann in gloss_tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
                a = ET.SubElement(pg, "ANNOTATION")
                a2 = ET.SubElement(a, "ALIGNABLE_ANNOTATION", {
                    "ANNOTATION_ID": f"pg{aid}",
                    "TIME_SLOT_REF1": ann.get("TIME_SLOT_REF1"),
                    "TIME_SLOT_REF2": ann.get("TIME_SLOT_REF2")
                })
                ET.SubElement(a2, "ANNOTATION_VALUE").text = ann.findtext("ANNOTATION_VALUE")
                aid += 1

        # SIGN tier
        sign_tier = seg_tiers.get("SIGN")
        if sign_tier:
            new_sign = ET.SubElement(ann_root, "TIER",
                                     TIER_ID="SIGN",
                                     LINGUISTIC_TYPE_REF="default-lt")
            for ann in sign_tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
                ms1 = seg_ts[ann.get("TIME_SLOT_REF1")].time
                ms2 = seg_ts[ann.get("TIME_SLOT_REF2")].time
                t1 = time_map[ms1]; t2 = time_map[ms2]
                a = ET.SubElement(new_sign, "ANNOTATION")
                a2 = ET.SubElement(a, "ALIGNABLE_ANNOTATION", {
                    "ANNOTATION_ID": ann.get("ANNOTATION_ID"),
                    "TIME_SLOT_REF1": t1,
                    "TIME_SLOT_REF2": t2
                })
                ET.SubElement(a2, "ANNOTATION_VALUE").text = ann.findtext("ANNOTATION_VALUE")
        else:
            print(f"  [merge] no SIGN tier for {vid}")

        # PSEUDO SENTENCE tier
        seg_sent = seg_tiers.get("SENTENCE")
        if seg_sent:
            new_ps = ET.SubElement(ann_root, "TIER",
                                   TIER_ID="PSEUDO SENTENCE",
                                   LINGUISTIC_TYPE_REF="default-lt")
            for ann in seg_sent.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
                ms1 = seg_ts[ann.get("TIME_SLOT_REF1")].time
                ms2 = seg_ts[ann.get("TIME_SLOT_REF2")].time
                t1 = time_map[ms1]; t2 = time_map[ms2]
                a = ET.SubElement(new_ps, "ANNOTATION")
                a2 = ET.SubElement(a, "ALIGNABLE_ANNOTATION", {
                    "ANNOTATION_ID": ann.get("ANNOTATION_ID"),
                    "TIME_SLOT_REF1": t1,
                    "TIME_SLOT_REF2": t2
                })
                ET.SubElement(a2, "ANNOTATION_VALUE").text = ann.findtext("ANNOTATION_VALUE")
        else:
            print(f"  [merge] no SENTENCE tier for {vid}")

        # Merge SIGN → PSEUDO GLOSS non-overlapping
        if sign_tier:
            for ann in sign_tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
                ms1 = seg_ts[ann.get("TIME_SLOT_REF1")].time
                ms2 = seg_ts[ann.get("TIME_SLOT_REF2")].time
                if any(overlaps((ms1,ms2), si) for si in sent_iv):
                    continue
                t1 = time_map[ms1]; t2 = time_map[ms2]
                a = ET.SubElement(pg, "ANNOTATION")
                a2 = ET.SubElement(a, "ALIGNABLE_ANNOTATION", {
                    "ANNOTATION_ID": f"pg{aid}",
                    "TIME_SLOT_REF1": t1,
                    "TIME_SLOT_REF2": t2
                })
                ET.SubElement(a2, "ANNOTATION_VALUE").text = ann.findtext("ANNOTATION_VALUE")
                aid += 1

        # Write merged EAF
        ET.ElementTree(ann_root).write(
            out_eaf,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True
        )
        print(f"Merged {vid} → {out_eaf}")

    # 2) Build data_merged.csv
    orig_df = pd.read_csv(os.path.join(here, "data.csv"))
    new_rows, skipped = [], []

    for vid, grp in orig_df.groupby("video_id"):
        seg_file = os.path.join(args.segmentation_path, f"{vid}.eaf")
        if not os.path.exists(seg_file):
            skipped.append(vid)
            continue

        _, ann_ts, ann_tiers = parse_eaf(os.path.join(args.annotation_path, f"{vid}.eaf"))
        sent_iv = get_intervals(ann_tiers["SENTENCE"], ann_ts)

        _, seg_ts, seg_tiers = parse_eaf(seg_file)
        sign_tier = seg_tiers.get("SIGN")
        if not sign_tier:
            continue

        for ann in sign_tier.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            ms1 = seg_ts[ann.get("TIME_SLOT_REF1")].time
            ms2 = seg_ts[ann.get("TIME_SLOT_REF2")].time
            if any(overlaps((ms1,ms2), si) for si in sent_iv):
                continue
            new_rows.append({
                "video_id": vid,
                "video_path": grp["video_path"].iat[0],
                "sentence_id": "",
                "split": grp["split"].iat[0],
                "start_sentence": "",
                "end_sentence": "",
                "gloss": ann.findtext("ANNOTATION_VALUE"),
                "start_gloss": ms1/1000.0,
                "end_gloss": ms2/1000.0
            })

    if skipped:
        print("\nSkipped pseudo-gloss for missing segmentation EAFs:")
        for v in skipped:
            print(f" - {v}")

    merged_df = pd.concat([orig_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged_df.sort_values(["video_id","start_gloss"], inplace=True)
    merged_df.to_csv(os.path.join(here, "data_merged.csv"), index=False)
    print(f"\nSaved {len(merged_df)} annotations to data_merged.csv")

    # 3) Print summary
    info = load_info()
    orig_stats   = summary_stats(orig_df, info)
    merged_stats = summary_stats(merged_df, info)

    headers = [""] + ["train","val","test"]
    table = []
    for key,label in [
        ("avg_frames_per_sign","Avg frames per sign"),
        ("avg_frames_per_video","Avg frames per video"),
        ("avg_glosses_per_video","Avg glosses per video"),
    ]:
        table.append([ label ] +
                     [ orig_stats[s][key] for s in ["train","val","test"] ])
        table.append([ f"{label} (incl. pseudo)" ] +
                     [ merged_stats[s][key] for s in ["train","val","test"] ])

    table.append(["Total videos"] +
                 [ orig_stats[s]["total_videos"] for s in ["train","val","test"] ])
    table.append(["Total signers"] +
                 [ orig_stats[s]["total_signers"] for s in ["train","val","test"] ])
    table.append(["Total glosses"] +
                 [ orig_stats[s]["total_glosses"] for s in ["train","val","test"] ])

    print("\nSummary Statistics:\n")
    print(tabulate(table, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
