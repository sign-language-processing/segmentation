"""Tests for datasets/common.py — enums, md5sum, load_and_augment, collate_fn, BaseSegmentationDataset."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from sign_language_segmentation.datasets.common import (
    BaseSegmentationDataset,
    DatasetType,
    Split,
    collate_fn,
    load_and_augment,
    md5sum,
)
from sign_language_segmentation.utils.bio import BIO

EXAMPLE_POSE = Path(__file__).parent / "example.pose"


# -- enums -------------------------------------------------------------------

class TestEnums:
    def test_split_values(self):
        assert Split.TRAIN == "train"
        assert Split.DEV == "dev"
        assert Split.TEST == "test"

    def test_dataset_type_values(self):
        assert DatasetType.DGS == "dgs"
        assert DatasetType.PLATFORM == "platform"
        assert DatasetType.COMBINED == "combined"

    def test_split_from_string(self):
        assert Split("train") is Split.TRAIN

    def test_dataset_type_from_string(self):
        assert DatasetType("combined") is DatasetType.COMBINED


# -- md5sum -------------------------------------------------------------------

class TestMd5sum:
    def test_deterministic(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        assert md5sum(str(f)) == md5sum(str(f))

    def test_different_content_different_hash(self, tmp_path: Path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert md5sum(str(a)) != md5sum(str(b))

    def test_known_hash(self, tmp_path: Path):
        f = tmp_path / "known.bin"
        f.write_bytes(b"")
        # md5 of empty string is d41d8cd98f00b204e9800998ecf8427e
        assert md5sum(str(f)) == "d41d8cd98f00b204e9800998ecf8427e"


# -- load_and_augment ---------------------------------------------------------

class TestLoadAndAugment:
    """integration tests using the real example.pose file."""

    def test_output_keys(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[{"start": 500, "end": 1500}],
            sentences=[{"start": 0, "end": 3000}],
            split=Split.DEV,
            num_frames=1024,
            velocity=True,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        assert "pose" in result
        assert "timestamps" in result
        assert "bio" in result
        assert "sign" in result["bio"]
        assert "sentence" in result["bio"]

    def test_output_shapes_with_velocity(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[{"start": 500, "end": 1500}],
            sentences=[],
            split=Split.DEV,
            num_frames=1024,
            velocity=True,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        T = result["pose"].shape[0]
        assert result["pose"].ndim == 3  # (T, joints, 6)
        assert result["pose"].shape[2] == 6  # xyz + velocity xyz
        assert result["timestamps"].shape == (T,)
        assert result["bio"]["sign"].shape == (T,)
        assert result["bio"]["sentence"].shape == (T,)

    def test_output_shapes_without_velocity(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[],
            sentences=[],
            split=Split.DEV,
            num_frames=1024,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        assert result["pose"].shape[2] == 3  # xyz only

    def test_bio_labels_match_annotations(self):
        # annotation covering frames roughly 15-45 at 30fps (500ms-1500ms)
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[{"start": 500, "end": 1500}],
            sentences=[],
            split=Split.DEV,
            num_frames=1024,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        sign_bio = result["bio"]["sign"].numpy()
        # should have at least one B and some I labels
        assert BIO["B"] in sign_bio
        assert BIO["I"] in sign_bio
        # first frame (0ms) should be O (annotation starts at 500ms)
        assert sign_bio[0] == BIO["O"]

    def test_empty_annotations_all_outside(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[],
            sentences=[],
            split=Split.DEV,
            num_frames=1024,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        assert all(result["bio"]["sign"] == BIO["O"])
        assert all(result["bio"]["sentence"] == BIO["O"])

    def test_train_split_crops_to_num_frames(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[],
            sentences=[],
            split=Split.TRAIN,
            num_frames=50,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        assert result["pose"].shape[0] <= 50

    def test_dev_split_loads_full_video(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[],
            sentences=[],
            split=Split.DEV,
            num_frames=50,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        # dev should load all 130 frames regardless of num_frames
        assert result["pose"].shape[0] == 130

    def test_tensors_are_float(self):
        result = load_and_augment(
            pose_path=str(EXAMPLE_POSE),
            fps=30.0,
            total_frames=130,
            signs=[],
            sentences=[],
            split=Split.DEV,
            num_frames=1024,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        assert result["pose"].dtype == torch.float32
        assert result["timestamps"].dtype == torch.float32
        assert result["bio"]["sign"].dtype == torch.int64


# -- collate_fn ---------------------------------------------------------------

class TestCollateFn:
    def _make_sample(self, seq_len: int, joints: int = 10, dims: int = 3) -> dict:
        return {
            "pose": torch.randn(seq_len, joints, dims),
            "timestamps": torch.arange(seq_len, dtype=torch.float32) / 25.0,
            "bio": {
                "sign": torch.full((seq_len,), BIO["O"], dtype=torch.long),
                "sentence": torch.full((seq_len,), BIO["O"], dtype=torch.long),
            },
        }

    def test_single_item_batch(self):
        batch = collate_fn([self._make_sample(seq_len=20)])
        assert batch["pose"].shape == (1, 20, 10, 3)
        assert batch["lengths"].tolist() == [20]

    def test_padding_to_max_length(self):
        batch = collate_fn([self._make_sample(seq_len=10), self._make_sample(seq_len=20)])
        assert batch["pose"].shape[0] == 2
        assert batch["pose"].shape[1] == 20  # padded to max
        assert batch["lengths"].tolist() == [10, 20]

    def test_padded_bio_is_unk(self):
        batch = collate_fn([self._make_sample(seq_len=5), self._make_sample(seq_len=10)])
        # first sample padded from 5 to 10, positions 5-9 should be UNK
        assert batch["bio"]["sign"][0, 5:].tolist() == [BIO["UNK"]] * 5
        assert batch["bio"]["sentence"][0, 5:].tolist() == [BIO["UNK"]] * 5

    def test_padded_pose_is_zero(self):
        batch = collate_fn([self._make_sample(seq_len=5), self._make_sample(seq_len=10)])
        assert torch.all(batch["pose"][0, 5:] == 0)

    def test_timestamps_present(self):
        batch = collate_fn([self._make_sample(seq_len=8)])
        assert "timestamps" in batch
        assert batch["timestamps"].shape == (1, 8)

    def test_equal_length_no_padding(self):
        batch = collate_fn([self._make_sample(seq_len=15), self._make_sample(seq_len=15)])
        assert batch["pose"].shape == (2, 15, 10, 3)
        # no UNK padding needed
        assert torch.all(batch["bio"]["sign"] != BIO["UNK"])


# -- BaseSegmentationDataset --------------------------------------------------

class TestBaseSegmentationDataset:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSegmentationDataset()

    def test_subclass_inherits_len_and_getitem(self):
        class DummyDataset(BaseSegmentationDataset):
            def __init__(self):
                self.items = [{"id": "test"}]
                self.split = Split.TRAIN
                self.num_frames = 1024
                self.velocity = False
                self.fps_aug = False
                self.frame_dropout = 0.0
                self.body_part_dropout = 0.0

            def get_split_manifest(self) -> dict:
                return {"dataset": "dummy"}

        ds = DummyDataset()
        assert len(ds) == 1
        assert ds.get_split_manifest() == {"dataset": "dummy"}
