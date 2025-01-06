import argparse
import copy
import json
import math
import os
import shutil
from functools import lru_cache
from itertools import chain

import gcsfs
import numpy as np
import psycopg2
from dotenv import load_dotenv
from pose_format import Pose
from tqdm import tqdm

from sign_language_segmentation.data.utils import preprocess_pose, BIO

load_dotenv()


@lru_cache(maxsize=None)
def get_database():
    for var in ['DB_NAME', 'DB_USER', 'DB_PASS', 'DB_HOST']:
        if os.environ.get(var, None) is None:
            raise Exception(f"Missing {var} environment variable")

    return psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASS'],
        host=os.environ['DB_HOST']
    )


def get_database_annotations():
    with open("query.sql", "r") as f:
        query = f.read()

    print("Getting database annotations...")
    conn = get_database()
    with conn.cursor() as cursor:
        cursor.execute(query)
        columns = list(cursor.description)
        columns = [c.name for c in columns]
        rows = [{c: v for c, v in zip(columns, instance)} for instance in cursor.fetchall()]

    videos = {}
    for row in rows:
        if row["videoId"] not in videos:
            videos[row["videoId"]] = []
        videos[row["videoId"]].append(row)

    return videos


def get_splits():
    with open("splits.json", "r") as f:
        splits = json.load(f)

    splits["dev"] = set(splits["dev"])
    splits["test"] = set(splits["test"])

    return splits


def create_bio(annotations, num_frames, fps):
    default_value = BIO["UNK"] if len(annotations) == 0 else BIO["O"]
    bio = np.full((num_frames,), fill_value=default_value, dtype=np.uint8)

    for row in annotations:
        start_frame = math.floor(row["start"] * fps / 1000)
        end_frame = math.ceil(row["end"] * fps / 1000)

        bio[start_frame] = BIO["B"]
        bio[start_frame + 1:end_frame + 1] = BIO["I"]

    return bio


def is_sign_annotation(row):
    if row["language"] in ["Sgnw", "hns"]:
        return True
    if row["language"] == "gloss" and " " not in row["text"]:
        return True
    return False


def read_pose_file(video_id: str, poses_dir: str):
    pose_path = os.path.join(poses_dir, f"{video_id}.pose")
    if pose_path.startswith("gs://"):
        with gcsfs.GCSFileSystem().open(pose_path, "rb") as f:
            return Pose.read(f.read())

    with open(pose_path, "rb") as f:
        return Pose.read(f.read())


def process_video(video_id: str, video_annotations, poses_dir: str):
    pose = read_pose_file(video_id, poses_dir)
    pose = preprocess_pose(pose)

    sign_annotations = [row for row in video_annotations if is_sign_annotation(row)]
    sentence_annotations = [row for row in video_annotations if not is_sign_annotation(row)]

    num_frames = len(pose.body.data)
    fps = pose.body.fps

    return {
        "pose": pose.body.data.filled(0)[:, 0],
        "sign_bio": create_bio(sign_annotations, num_frames, fps),
        "sentence_bio": create_bio(sentence_annotations, num_frames, fps)
    }


def save_split(output_dir: str, poses_dir: str, data, padding=50,
               chunk_size=2 ** 16):  # todo increase chunk_size to 2 ** 20 (1GB)
    if len(data) == 0:
        return

    new_chunk = {
        "total_frames": 0,
        "pose": [],
        "sign_bio": [],
        "sentence_bio": []
    }
    chunk = copy.deepcopy(new_chunk)
    chunk_id = 0
    for video_id, annotations in tqdm(data.items()):
        datum = process_video(video_id, annotations, poses_dir)
        chunk["pose"].append(datum["pose"])
        chunk["sign_bio"].append(datum["sign_bio"])
        chunk["sentence_bio"].append(datum["sentence_bio"])
        chunk["total_frames"] += len(datum["pose"]) + padding

        if chunk["total_frames"] > chunk_size:
            save_chunk(output_dir, chunk, chunk_id, padding)
            chunk = copy.deepcopy(new_chunk)
            chunk_id += 1

    if chunk["total_frames"] > 0:
        save_chunk(output_dir, chunk, chunk_id, padding)


def save_chunk(output_dir: str, chunk, chunk_id: int, padding: int):
    os.makedirs(output_dir, exist_ok=True)
    example_pose = chunk["pose"][0]
    assert len(example_pose.shape) == 3, "Expected 3D pose tensor"

    padding_shape = (padding, example_pose.shape[1], example_pose.shape[2])  # (padding, 178, 3)
    padding_pose = np.zeros(padding_shape, dtype=np.float16)
    padding_bio = np.full((padding,), fill_value=BIO["UNK"], dtype=np.uint8)

    all_poses = chain.from_iterable([(pose, padding_pose) for pose in chunk["pose"]])
    pose_tensors = np.concatenate(list(all_poses), dtype=np.float16)

    all_sign_bio = chain.from_iterable([(sign_bio, padding_bio) for sign_bio in chunk["sign_bio"]])
    sign_bio_tensors = np.concatenate(list(all_sign_bio), dtype=np.uint8)

    all_sentence_bio = chain.from_iterable([(sentence_bio, padding_bio) for sentence_bio in chunk["sentence_bio"]])
    sentence_bio_tensors = np.concatenate(list(all_sentence_bio), dtype=np.uint8)

    np.save(os.path.join(output_dir, f"pose.{chunk_id}.npy"), pose_tensors)
    np.save(os.path.join(output_dir, f"sign_bio.{chunk_id}.npy"), sign_bio_tensors)
    np.save(os.path.join(output_dir, f"sentence_bio.{chunk_id}.npy"), sentence_bio_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", default="/Volumes/Echo/GCS/sign-mt-poses/")  # default="gs://sign-mt-poses/"
    parser.add_argument("--output", default="/tmp/segmentation")
    args = parser.parse_args()

    videos = get_database_annotations()
    splits = get_splits()
    splits_data = {"train": {}, "dev": {}, "test": {}}
    for video_id, annotations in tqdm(videos.items()):
        split = "dev" if video_id in splits["dev"] else "test" if video_id in splits["test"] else "train"
        splits_data[split][video_id] = annotations

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)
    
    for split, data in splits_data.items():
        save_split(os.path.join(args.output, split), args.poses, data)
