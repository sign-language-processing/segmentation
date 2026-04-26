"""Tests for datasets/signtube/ — dataset class and defensive checks."""
import json
import shutil
from pathlib import Path

import pytest

from sign_language_segmentation.datasets.common import Split
from sign_language_segmentation.datasets.signtube import sync
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
        # invariant: 10-video fixture + split_seed=42 + default ratios place ≥1 in TRAIN — re-verify if any of those change.
        assert len(ds) > 0
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


class TestBuildCache:
    def test_happy_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        md5 = "abc123"
        video_id = "swn_vid_001"
        pose_file = tmp_path / f"{md5}.pose"
        pose_file.write_bytes(b"fake-bytes")

        monkeypatch.setattr(sync, "_NAS_POSES_DIR", tmp_path)
        monkeypatch.setattr(
            sync,
            "_build_signtube_video_lookup",
            lambda: {video_id: {"md5": md5, "fps": 30.0, "total_frames": 130}},
        )

        videos = {
            video_id: [
                {"language": "Sgnw", "start": 500, "end": 1500},
                {"language": "en", "start": 0, "end": 3000},
            ]
        }
        cache = sync._build_cache(videos)

        assert set(cache.keys()) == {"videos"}
        assert video_id in cache["videos"]
        entry = cache["videos"][video_id]
        assert entry["pose_path"] == str(pose_file)
        assert entry["fps"] == 30.0
        assert entry["total_frames"] == 130
        assert entry["signs"] == [{"start": 500.0, "end": 1500.0}]
        assert entry["sentences"] == [{"start": 0.0, "end": 3000.0}]

    def test_skips_when_video_metadata_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sync, "_NAS_POSES_DIR", tmp_path)
        monkeypatch.setattr(sync, "_build_signtube_video_lookup", lambda: {})

        videos = {"swn_vid_nomatch": [{"language": "Sgnw", "start": 500, "end": 1500}]}
        cache = sync._build_cache(videos)

        assert cache == {"videos": {}}

    def test_build_signtube_video_lookup_uses_csv_metadata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        video_list = tmp_path / "video_list.csv"
        video_list.write_text(
            "name,md5Hash,age,gender,duration,avg_frame_rate,skin_tone\n"
            "sign-tube/foo.mp4,abc123,,,2.86,50/1,\n"
            "other/bar.mp4,ignored,,,2.00,25/1,\n"
        )
        monkeypatch.setattr(sync, "_NAS_VIDEO_LIST", video_list)

        lookup = sync._build_signtube_video_lookup()

        assert lookup == {"foo": {"md5": "abc123", "fps": 50.0, "total_frames": 143}}
