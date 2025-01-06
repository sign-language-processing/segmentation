import argparse
import glob
import os
from typing import Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

Split = Literal["train", "dev", "test"]


class SegmentationDataset(IterableDataset):
    def __init__(self, data_dir: str, split: Split = "train", num_frames: int = 2 ** 16):
        self.is_finite = split != "train"
        self.split_dir = os.path.join(data_dir, split)
        self.num_frames = num_frames

        self.pose_files = self._list_files("pose")
        self.sign_bio_files = self._list_files("sign_bio")
        self.sentence_bio_files = self._list_files("sentence_bio")

        assert len(self.pose_files) == len(self.sign_bio_files) == len(self.sentence_bio_files), \
            f"Mismatched number of pose ({len(self.pose_files)}) and bio ({len(self.sign_bio_files)}, {len(self.sentence_bio_files)}) files"

    def _list_files(self, name: str) -> list[str]:
        return sorted(glob.glob(os.path.join(self.split_dir, f"{name}.*.npy")))

    def _load_chunk(self, chunk_idx: int) -> dict[str, torch.Tensor]:
        real_chunk_idx = chunk_idx % len(self.pose_files)
        return {
            'pose': torch.from_numpy(np.load(self.pose_files[real_chunk_idx])),
            'sign_bio': torch.from_numpy(np.load(self.sign_bio_files[real_chunk_idx])),
            'sentence_bio': torch.from_numpy(np.load(self.sentence_bio_files[real_chunk_idx]))
        }

    def __iter__(self):
        current_chunk_idx = 0
        current_frame_idx = 0
        current_chunk = self._load_chunk(current_chunk_idx)

        # Train set is infinite, dev and test sets are finite
        is_done = lambda : self.is_finite and current_chunk_idx >= len(self.pose_files)

        while not is_done():
            pose = [current_chunk['pose'][current_frame_idx:current_frame_idx + self.num_frames]]
            sign_bio = [current_chunk['sign_bio'][current_frame_idx:current_frame_idx + self.num_frames]]
            sentence_bio = [current_chunk['sentence_bio'][current_frame_idx:current_frame_idx + self.num_frames]]
            current_frame_idx += self.num_frames
            datum_length = len(pose[0])

            while datum_length < self.num_frames:
                remaining_frames = self.num_frames - datum_length
                current_chunk_idx += 1
                if is_done():
                    break

                current_chunk = self._load_chunk(current_chunk_idx)
                pose.append(current_chunk['pose'][0:remaining_frames])
                sign_bio.append(current_chunk['sign_bio'][0:remaining_frames])
                sentence_bio.append(current_chunk['sentence_bio'][0:remaining_frames])
                current_frame_idx = remaining_frames
                datum_length += len(pose[-1])

            yield {
                'pose': torch.cat(pose).type(torch.float32),
                'bio': {
                    'sign': torch.cat(sign_bio).type(torch.long),
                    'sentence': torch.cat(sentence_bio).type(torch.long)
                }
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/tmp/segmentation")
    args = parser.parse_args()

    import time
    import psutil

    process = psutil.Process()
    current_memory = lambda: process.memory_info().rss / 1024 / 1024  # Convert to MB

    initial_memory = current_memory()
    now = time.time()
    num_frames = 2 ** 16
    dataset = SegmentationDataset(args.dataset, split="train", num_frames=num_frames)
    print(f"Init time taken: {time.time() - now:.2f} seconds")
    print(f"Memory used: {current_memory() - initial_memory:.1f} MB")

    now = time.time()
    dataset.is_finite = True
    for datum in tqdm(dataset):
        assert datum["pose"].shape[0] == datum["bio"]["sign"].shape[0] == datum["bio"]["sentence"].shape[0], \
            f"Mismatched number of frames in pose ({datum['pose'].shape[0]}), sign_bio ({datum['bio']['sign'].shape[0]}), and sentence_bio ({datum['bio']['sentence'].shape[0]}) tensors"
        assert datum["pose"].shape[0] <= num_frames, \
            f"More frames than expected: {datum['pose'].shape[0]} > {num_frames}"
    print(f"Get all rows time taken: {time.time() - now:.2f} seconds")
