"""Tests for datasets/signtube/ — dataset class and defensive checks."""
import json
import shutil
from pathlib import Path

import pytest

from sign_language_segmentation.datasets.common import Split
from sign_language_segmentation.datasets.signtube.dataset import SignTubeSegmentationDataset
from sign_language_segmentation.datasets.signtube.sync import _is_sign_annotation

EXAMPLE_POSE = Path(__file__).parent / "example.pose"


@pytest.fixture
def cache_with_poses(tmp_path: Path) -> Path:
    """create a cache JSON with 10 videos pointing at copies of example.pose."""
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    videos = {}
    for i in range(10):
        video_id = f"swn_vid_{i:03d}"
        pose_path = poses_dir / f"{video_id}.pose"
        shutil.copy(EXAMPLE_POSE, pose_path)
        videos[video_id] = {
            "pose_path": str(pose_path),
            "fps": 30.0,
            "total_frames": 130,
            "signs": [{"start": 500, "end": 1500}],
            "sentences": [{"start": 0, "end": 3000}],
        }

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"videos": videos}))
    return cache_path


class TestSignTubeSegmentationDataset:
    def test_loads_items(self, cache_with_poses: Path):
        ds = SignTubeSegmentationDataset(
            annotations_path=str(cache_with_poses),
            split=Split.TRAIN,
            velocity=False,
            fps_aug=False,
        )
        assert len(ds) > 0

    def test_all_splits_cover_all_videos(self, cache_with_poses: Path):
        all_ids: set[str] = set()
        for split in Split:
            ds = SignTubeSegmentationDataset(
                annotations_path=str(cache_with_poses),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            all_ids.update(item["id"] for item in ds.items)
        assert all_ids == {f"swn_vid_{i:03d}" for i in range(10)}

    def test_splits_are_disjoint(self, cache_with_poses: Path):
        split_ids: dict[Split, set[str]] = {}
        for split in Split:
            ds = SignTubeSegmentationDataset(
                annotations_path=str(cache_with_poses),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            split_ids[split] = {item["id"] for item in ds.items}
        assert split_ids[Split.TRAIN] & split_ids[Split.DEV] == set()
        assert split_ids[Split.TRAIN] & split_ids[Split.TEST] == set()
        assert split_ids[Split.DEV] & split_ids[Split.TEST] == set()

    def test_split_is_deterministic(self, cache_with_poses: Path):
        ids_a = [
            item["id"]
            for item in SignTubeSegmentationDataset(
                annotations_path=str(cache_with_poses),
                split=Split.TRAIN,
                velocity=False,
                fps_aug=False,
            ).items
        ]
        ids_b = [
            item["id"]
            for item in SignTubeSegmentationDataset(
                annotations_path=str(cache_with_poses),
                split=Split.TRAIN,
                velocity=False,
                fps_aug=False,
            ).items
        ]
        assert ids_a == ids_b

    def test_get_split_manifest_schema(self, cache_with_poses: Path):
        ds = SignTubeSegmentationDataset(
            annotations_path=str(cache_with_poses),
            split=Split.TRAIN,
            velocity=False,
            fps_aug=False,
        )
        manifest = ds.get_split_manifest()
        assert manifest["dataset"] == "signtube"
        assert "split_seed" in manifest
        assert set(manifest["splits"].keys()) == {"train", "dev", "test"}
        for ids in manifest["splits"].values():
            assert isinstance(ids, list)
            assert ids == sorted(ids)

    def test_getitem_returns_expected_format(self, cache_with_poses: Path):
        ds = SignTubeSegmentationDataset(
            annotations_path=str(cache_with_poses),
            split=Split.TRAIN,
            velocity=False,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        if len(ds) == 0:
            pytest.skip("no items in train split for this seed")
        sample = ds[0]
        assert "pose" in sample
        assert "timestamps" in sample
        assert "bio" in sample
        assert sample["pose"].ndim == 3

    def test_skips_missing_pose_files(self, tmp_path: Path):
        cache = {
            "videos": {
                "missing": {
                    "pose_path": str(tmp_path / "does_not_exist.pose"),
                    "fps": 30.0,
                    "total_frames": 130,
                    "signs": [{"start": 0, "end": 1000}],
                },
            }
        }
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache))
        ds = SignTubeSegmentationDataset(
            annotations_path=str(cache_path),
            split=Split.TRAIN,
            velocity=False,
            fps_aug=False,
        )
        assert len(ds) == 0

    def test_skips_videos_with_too_few_frames(self, tmp_path: Path):
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        good_pose = poses_dir / "good.pose"
        short_pose = poses_dir / "short.pose"
        shutil.copy(EXAMPLE_POSE, good_pose)
        shutil.copy(EXAMPLE_POSE, short_pose)
        cache = {
            "videos": {
                "good": {
                    "pose_path": str(good_pose),
                    "fps": 30.0,
                    "total_frames": 130,
                    "signs": [{"start": 0, "end": 1000}],
                },
                "short": {
                    "pose_path": str(short_pose),
                    "fps": 30.0,
                    "total_frames": 1,
                    "signs": [{"start": 0, "end": 100}],
                },
            }
        }
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache))
        all_ids: set[str] = set()
        for split in Split:
            ds = SignTubeSegmentationDataset(
                annotations_path=str(cache_path),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            all_ids.update(item["id"] for item in ds.items)
        assert "good" in all_ids
        assert "short" not in all_ids

    def test_raises_on_corrupted_cache(self, tmp_path: Path):
        cache_path = tmp_path / "bad.json"
        cache_path.write_text(json.dumps({"not_videos": {}}))
        with pytest.raises(ValueError, match="missing 'videos' key"):
            SignTubeSegmentationDataset(
                annotations_path=str(cache_path),
                split=Split.TRAIN,
                velocity=False,
                fps_aug=False,
            )

    def test_pose_path_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # point CACHE_DIR at tmp_path so poses_dir = tmp_path/signtube/poses
        monkeypatch.setattr(
            "sign_language_segmentation.datasets.signtube.dataset.CACHE_DIR",
            tmp_path,
        )
        fallback_dir = tmp_path / "signtube" / "poses"
        fallback_dir.mkdir(parents=True)

        video_id = "swn_vid_fb"
        shutil.copy(EXAMPLE_POSE, fallback_dir / f"{video_id}.pose")

        # cache pose_path is invalid; real file only exists at the fallback location
        cache = {
            "videos": {
                video_id: {
                    "pose_path": str(tmp_path / "invalid" / "nope.pose"),
                    "fps": 30.0,
                    "total_frames": 130,
                    "signs": [{"start": 0, "end": 1000}],
                },
            }
        }
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache))

        all_ids: set[str] = set()
        for split in Split:
            ds = SignTubeSegmentationDataset(
                annotations_path=str(cache_path),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            all_ids.update(item["id"] for item in ds.items)
        assert video_id in all_ids


class TestIsSignAnnotation:
    def test_sgnw_is_sign(self):
        assert _is_sign_annotation({"language": "Sgnw"}) is True

    def test_hns_is_sign(self):
        assert _is_sign_annotation({"language": "hns"}) is True

    def test_other_languages_are_sentences(self):
        for lang in ("en", "de", "gloss", "", "Sgnw-other"):
            assert _is_sign_annotation({"language": lang}) is False
