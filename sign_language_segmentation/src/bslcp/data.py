import os
import argparse
import pickle
import csv
import subprocess
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm  # progress bar

def load_info_pkl():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    info_path = os.path.join(current_dir, 'info.pkl')
    with open(info_path, 'rb') as f:
        return pickle.load(f)

def find_video_path(video_path_root, org_name):
    filename = os.path.basename(org_name)
    org_name_lower = org_name.lower()
    for category in ['conversation', 'interview', 'narrative']:
        if category in org_name_lower:
            candidate = os.path.join(video_path_root, category, filename)
            if os.path.isfile(candidate):
                return candidate
            break
    return None

def is_video_wellformed(path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode == 0 and res.stdout.strip()

def group_by_org_name(info):
    grouped = defaultdict(list)
    for i, org in enumerate(info['videos']['org_name']):
        grouped[org].append(i)
    return grouped

def find_consecutive_zeros(boundaries):
    segments, in_seg, start = [], False, 0
    for idx, val in enumerate(boundaries):
        if val == 0 and not in_seg:
            start, in_seg = idx, True
        elif val == 1 and in_seg:
            segments.append((start, idx))
            in_seg = False
    if in_seg:
        segments.append((start, len(boundaries)))
    return segments

def compute_summary_table(df, info):
    df['duration_frames'] = (df['end_gloss'] - df['start_gloss']) * 25
    summary = defaultdict(dict)
    for key, name in zip([0,1,2], ['train','val','test']):
        split_df = df[df['split']==name].copy()
        fps = split_df['duration_frames']
        summary[name]['avg_frames_per_sign'] = f"{fps.mean():.1f} ± {fps.std():.1f}"
        vid_frames = split_df.groupby('video_id')['duration_frames'].sum()
        summary[name]['avg_frames_per_video'] = f"{vid_frames.mean():.1f} ± {vid_frames.std():.1f}"
        glosses = split_df.groupby('video_id')['gloss'].nunique()
        summary[name]['avg_glosses_per_video'] = f"{glosses.mean():.1f} ± {glosses.std():.1f}"
        summary[name]['total_videos'] = split_df['video_id'].nunique()
        signers = set(
            s for s, sp in zip(info['videos']['signer'], info['videos']['split']) if sp==key
        )
        summary[name]['total_signers'] = len(signers)
        summary[name]['total_glosses'] = split_df['gloss'].nunique()
    return pd.DataFrame(summary).T

def save_elan_file(df, video_id, output_path):
    video_df = df[df['video_id']==video_id].sort_values('start_gloss')
    sentence_df = video_df.groupby('sentence_id').agg({
        'start_sentence':'first','end_sentence':'first'
    }).reset_index()
    video_path = video_df.iloc[0]['video_path']
    video_fname = os.path.basename(video_path)

    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR":"", "DATE":"", "FORMAT":"3.0", "VERSION":"2.8"
    })
    ET.SubElement(root, "HEADER", {
        "MEDIA_FILE": video_fname,
        "TIME_UNITS": "milliseconds"
    })
    to = ET.SubElement(root, "TIME_ORDER")
    timeslots, ts_id, all_times = {}, 1, set()

    # collect all time points
    for _, r in video_df.iterrows():
        all_times |= {int(r['start_gloss']*1000), int(r['end_gloss']*1000)}
    for _, r in sentence_df.iterrows():
        all_times |= {int(r['start_sentence']*1000), int(r['end_sentence']*1000)}

    # create time slots
    for t in sorted(all_times):
        ET.SubElement(to, "TIME_SLOT", {
            "TIME_SLOT_ID": f"ts{ts_id}",
            "TIME_VALUE": str(t)
        })
        timeslots[t] = f"ts{ts_id}"
        ts_id += 1

    # GLOSS tier
    gloss_tier = ET.SubElement(root, "TIER", {
        "TIER_ID": "GLOSS",
        "LINGUISTIC_TYPE_REF": "default-lt"
    })
    aid = 1
    for _, r in video_df.iterrows():
        ann = ET.SubElement(gloss_tier, "ANNOTATION")
        align = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{aid}",
            "TIME_SLOT_REF1": timeslots[int(r['start_gloss']*1000)],
            "TIME_SLOT_REF2": timeslots[int(r['end_gloss']*1000)]
        })
        ET.SubElement(align, "ANNOTATION_VALUE").text = r['gloss']
        aid += 1

    # SENTENCE tier
    sent_tier = ET.SubElement(root, "TIER", {
        "TIER_ID": "SENTENCE",
        "LINGUISTIC_TYPE_REF": "default-lt"
    })
    for _, r in sentence_df.iterrows():
        ann = ET.SubElement(sent_tier, "ANNOTATION")
        align = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{aid}",
            "TIME_SLOT_REF1": timeslots[int(r['start_sentence']*1000)],
            "TIME_SLOT_REF2": timeslots[int(r['end_sentence']*1000)]
        })
        ET.SubElement(align, "ANNOTATION_VALUE").text = ""
        aid += 1

    ET.SubElement(root, "LINGUISTIC_TYPE", {
        "LINGUISTIC_TYPE_ID": "default-lt",
        "TIME_ALIGNABLE": "true"
    })

    xml_str = ET.tostring(root, 'utf-8')
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',       default='/users/zifan/BSL-Corpus/raw')
    parser.add_argument('--video_path_new',   default='/users/zifan/BSL-Corpus/derivatives/videos')
    parser.add_argument('--annotation_path',  default='/users/zifan/BSL-Corpus/derivatives/annotation')
    args = parser.parse_args()

    info     = load_info_pkl()
    grouped  = group_by_org_name(info)
    total    = len(grouped)
    print(f"Total videos in metadata: {total}")

    missing, fps_set, rows, seen, dup, found = set(), set(), [], set(), set(), set()
    for org, idxs in grouped.items():
        path = find_video_path(args.video_path, org)
        vid  = os.path.splitext(os.path.basename(org))[0]
        if vid in seen: dup.add(vid)
        seen.add(vid)
        if not path:
            missing.add(org)
            continue
        found.add(path)
        for i in idxs:
            fps    = info['videos']['videos']['fps'][i]; fps_set.add(fps)
            st, et = info['videos']['start'][i], info['videos']['end'][i]
            b      = info['videos']['alignments']['boundaries'][i]
            glosss = info['videos']['alignments']['gloss'][i]
            split  = {0:"train",1:"val",2:"test"}[info['videos']['split'][i]]
            for sf, ef in find_consecutive_zeros(b):
                rows.append({
                    'video_id':      vid,
                    'video_path':    path,
                    'sentence_id':   i,
                    'split':         split,
                    'start_sentence':st,
                    'end_sentence':  et,
                    'gloss':         glosss[sf] if sf < len(glosss) else "",
                    'start_gloss':   st + sf/fps,
                    'end_gloss':     st + ef/fps
                })

    print(f"Videos found: {len(found)}  (missing {len(missing)})")
    if dup:
        print(f"⚠️ Duplicate IDs: {dup}")
    if missing:
        print("Missing org_names:")
        for m in missing:
            print(f" - {m}")

    # well-formedness
    print("\nChecking well-formedness of video files...")
    valid, broken = set(), set()
    for p in tqdm(found, desc="FFprobe"):
        (valid if is_video_wellformed(p) else broken).add(p)
    print(f"Well-formed: {len(valid)}  Broken: {len(broken)}")
    if broken:
        print("Broken files:")
        for b in broken:
            print(f" - {b}")

    # filter out broken
    rows = [r for r in rows if r['video_path'] in valid]
    print(f"\nVideos after filtering broken: {len(valid)}")

    # create new video symlinks
    os.makedirs(args.video_path_new, exist_ok=True)
    newmap = {}
    for p in valid:
        vid = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1]
        newp = os.path.join(args.video_path_new, f"{vid}{ext}")
        if os.path.exists(newp): os.remove(newp)
        os.symlink(p, newp)
        newmap[p] = newp
    for r in rows:
        r['video_path'] = newmap[r['video_path']]

    # write CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_csv     = os.path.join(script_dir, 'data.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'video_id','video_path','sentence_id','split',
            'start_sentence','end_sentence','gloss',
            'start_gloss','end_gloss'
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} annotations to {out_csv}")

    # summary
    df   = pd.DataFrame(rows)
    summ = compute_summary_table(df, info)
    print("\nSummary Statistics:\n")
    table = []
    for stat in [
        'avg_frames_per_sign','avg_frames_per_video',
        'avg_glosses_per_video','total_videos',
        'total_signers','total_glosses'
    ]:
        row = [ stat.replace('_',' ').capitalize() ]
        for split in ['train','val','test']:
            row.append(summ.loc[split, stat] if split in summ.index else '-')
        table.append(row)
    print(tabulate(table, headers=[""]+['train','val','test'], tablefmt="github"))

    # save ELAN for all videos
    os.makedirs(args.annotation_path, exist_ok=True)
    print(f"\nSaving ELAN files to {args.annotation_path}")
    for vid in df['video_id'].unique():
        eaf_path = os.path.join(args.annotation_path, f"{vid}.eaf")
        save_elan_file(df, vid, eaf_path)
        print(f" - {vid}: {eaf_path}")

if __name__ == '__main__':
    main()
