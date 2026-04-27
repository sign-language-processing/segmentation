"""Tests for datasets/annotation_platform/ — dataset class, split logic, sync helpers."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from sign_language_segmentation.datasets.annotation_platform.dataset import (
    AnnotationPlatformSegmentationDataset,
)
from sign_language_segmentation.datasets.annotation_platform.sync import (
    _gcs_url_to_local,
    convex_query,
    fetch_ontology_class_map,
    sync,
)
from sign_language_segmentation.datasets.common import Split, split_bucket as _split_bucket

EXAMPLE_POSE = Path(__file__).parent / "example.pose"


# -- _split_bucket ------------------------------------------------------------

class TestSplitBucket:
    def test_deterministic(self):
        a = _split_bucket(video_id="video-1", seed=42)
        b = _split_bucket(video_id="video-1", seed=42)
        assert a == b

    def test_range_0_to_999(self):
        for i in range(100):
            bucket = _split_bucket(video_id=f"video-{i}", seed=0)
            assert 0 <= bucket <= 999

    def test_different_seeds_different_buckets(self):
        # not guaranteed to differ per pair, but extremely unlikely to be equal across many
        diffs = sum(
            _split_bucket(video_id=f"v-{i}", seed=1) != _split_bucket(video_id=f"v-{i}", seed=2)
            for i in range(20)
        )
        assert diffs > 10  # most should differ

    def test_different_ids_spread_across_range(self):
        buckets = {_split_bucket(video_id=f"video-{i}", seed=42) for i in range(200)}
        # 200 IDs should produce a reasonable spread
        assert len(buckets) > 100


# -- AnnotationPlatformSegmentationDataset ------------------------------------

@pytest.fixture
def cache_with_poses(tmp_path: Path) -> tuple[Path, Path]:
    """create a cache JSON with videos pointing to copies of example.pose."""
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()

    videos = {}
    for i in range(10):
        pose_hash = f"hash{i:04d}"
        shutil.copy(EXAMPLE_POSE, poses_dir / f"{pose_hash}.pose")
        videos[f"video-{i}"] = {
            "pose_hash": pose_hash,
            "fps": 30.0,
            "total_frames": 130,
            "signs": [{"start": 500, "end": 1500}],
            "phrases": [{"start": 0, "end": 3000}],
            "quality_score": (10 - i) / 10,  # video-0 = 1.0, video-9 = 0.1
        }

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"videos": videos}))
    return cache_path, poses_dir


class TestAnnotationPlatformDataset:
    def test_loads_items(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path),
            poses_dir=str(poses_dir),
            split=Split.TRAIN,
            velocity=False,
            fps_aug=False,
        )
        # with 80/10/10 split and 10 videos, train should get ~8
        assert len(ds) > 0

    def test_all_splits_cover_all_videos(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        all_ids: set[str] = set()
        for split in Split:
            ds = AnnotationPlatformSegmentationDataset(
                annotations_path=str(cache_path),
                poses_dir=str(poses_dir),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            all_ids.update(item["id"] for item in ds.items)
        assert all_ids == {f"video-{i}" for i in range(10)}

    def test_splits_are_disjoint(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        split_ids: dict[str, set[str]] = {}
        for split in Split:
            ds = AnnotationPlatformSegmentationDataset(
                annotations_path=str(cache_path),
                poses_dir=str(poses_dir),
                split=split,
                velocity=False,
                fps_aug=False,
            )
            split_ids[split] = {item["id"] for item in ds.items}
        assert split_ids[Split.TRAIN] & split_ids[Split.DEV] == set()
        assert split_ids[Split.TRAIN] & split_ids[Split.TEST] == set()
        assert split_ids[Split.DEV] & split_ids[Split.TEST] == set()

    def test_split_is_deterministic(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        ids_a = [
            item["id"] for item in AnnotationPlatformSegmentationDataset(
                annotations_path=str(cache_path), poses_dir=str(poses_dir),
                split=Split.TRAIN, velocity=False, fps_aug=False,
            ).items
        ]
        ids_b = [
            item["id"] for item in AnnotationPlatformSegmentationDataset(
                annotations_path=str(cache_path), poses_dir=str(poses_dir),
                split=Split.TRAIN, velocity=False, fps_aug=False,
            ).items
        ]
        assert ids_a == ids_b

    def test_quality_filtering(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        # with quality_percentile=0.5, keep top 5 of 10 videos
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path),
            poses_dir=str(poses_dir),
            split=Split.TRAIN,
            quality_percentile=0.5,
            velocity=False,
            fps_aug=False,
        )
        manifest = ds.get_split_manifest()
        total_across_splits = sum(len(ids) for ids in manifest["splits"].values())
        assert total_across_splits == 5

    def test_quality_filtering_keeps_highest_scores(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        ds_all = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, quality_percentile=1.0, velocity=False, fps_aug=False,
        )
        ds_filtered = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, quality_percentile=0.3, velocity=False, fps_aug=False,
        )
        # filtered items should be a subset of all items
        all_ids = {item["id"] for item in ds_all.items}
        filtered_ids = {item["id"] for item in ds_filtered.items}
        assert filtered_ids <= all_ids

    def test_get_split_manifest_schema(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, velocity=False, fps_aug=False,
        )
        manifest = ds.get_split_manifest()
        assert manifest["dataset"] == "annotation_platform"
        assert "split_seed" in manifest
        assert "quality_percentile" in manifest
        assert set(manifest["splits"].keys()) == {"train", "dev", "test"}
        for ids in manifest["splits"].values():
            assert isinstance(ids, list)
            assert ids == sorted(ids)  # should be sorted

    def test_getitem_returns_expected_format(self, cache_with_poses: tuple[Path, Path]):
        cache_path, poses_dir = cache_with_poses
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, velocity=False, fps_aug=False, frame_dropout=0.0,
            body_part_dropout=0.0,
        )
        if len(ds) == 0:
            pytest.skip("no items in train split for this seed")
        sample = ds[0]
        assert "pose" in sample
        assert "timestamps" in sample
        assert "bio" in sample
        assert sample["pose"].ndim == 3

    def test_skips_videos_without_signs(self, tmp_path: Path):
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        shutil.copy(EXAMPLE_POSE, poses_dir / "h.pose")
        cache = {"videos": {
            "has-signs": {"pose_hash": "h", "fps": 30.0, "total_frames": 130,
                          "signs": [{"start": 0, "end": 1000}]},
            "no-signs": {"pose_hash": "h", "fps": 30.0, "total_frames": 130,
                         "signs": []},
        }}
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache))
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, velocity=False, fps_aug=False,
            dev_ratio=0.0, test_ratio=0.0,  # all to train
        )
        ids = {item["id"] for item in ds.items}
        assert "has-signs" in ids
        assert "no-signs" not in ids

    def test_skips_missing_pose_files(self, tmp_path: Path):
        poses_dir = tmp_path / "poses"
        poses_dir.mkdir()
        # don't actually copy the pose file
        cache = {"videos": {
            "missing": {"pose_hash": "nonexistent", "fps": 30.0, "total_frames": 130,
                        "signs": [{"start": 0, "end": 1000}]},
        }}
        cache_path = tmp_path / "c.json"
        cache_path.write_text(json.dumps(cache))
        ds = AnnotationPlatformSegmentationDataset(
            annotations_path=str(cache_path), poses_dir=str(poses_dir),
            split=Split.TRAIN, velocity=False, fps_aug=False,
        )
        assert len(ds) == 0


# -- _gcs_url_to_local --------------------------------------------------------

class TestGcsUrlToLocal:
    def test_gs_protocol(self):
        result = _gcs_url_to_local("gs://my-bucket/path/to/video.mp4", gcs_root="/mnt/nas/GCS")
        assert result == Path("/mnt/nas/GCS/my-bucket/path/to/video.mp4")

    def test_https_storage_googleapis(self):
        result = _gcs_url_to_local(
            "https://storage.googleapis.com/my-bucket/video.mp4",
            gcs_root="/mnt/nas/GCS",
        )
        assert result == Path("/mnt/nas/GCS/my-bucket/video.mp4")

    def test_other_http_url(self):
        result = _gcs_url_to_local("https://example.com/files/video.mp4", gcs_root="/mnt/nas/GCS")
        assert result == Path("/mnt/nas/GCS/files/video.mp4")

    def test_local_path_passthrough(self):
        result = _gcs_url_to_local("/local/path/video.mp4", gcs_root="/mnt/nas/GCS")
        assert result == Path("/local/path/video.mp4")


# -- convex_query (mocked) ----------------------------------------------------

class TestConvexQuery:
    @patch("sign_language_segmentation.datasets.annotation_platform.sync.httpx.post")
    def test_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {"status": "success", "value": [{"_id": "abc"}]}

        result = convex_query(url="https://example.convex.cloud", path="test:list")
        assert result == [{"_id": "abc"}]
        mock_post.assert_called_once()

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.httpx.post")
    def test_convex_error_raises(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {
            "status": "error",
            "errorMessage": "Function not found",
        }

        with pytest.raises(RuntimeError, match="Function not found"):
            convex_query(url="https://example.convex.cloud", path="bad:func")

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.httpx.post")
    def test_auth_token_passed_in_header(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {"status": "success", "value": []}

        convex_query(url="https://example.convex.cloud", path="test:list", token="secret123")
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer secret123"


# -- fetch_ontology_class_map (mocked) ----------------------------------------

class TestFetchOntologyClassMap:
    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_maps_sign_and_phrase(self, mock_query):
        mock_query.side_effect = [
            {
                "ontologyGroupId": "group1",
                "objectClasses": [],
            },
            [{
                "status": "published",
                "objectClasses": [
                    {"_id": "cls1", "annotationType": "time_aligned", "type": "sign_language_sign"},
                    {"_id": "cls2", "annotationType": "time_aligned", "type": "spoken_language_phrase"},
                    {"_id": "cls3", "annotationType": "global", "type": "personal_attributes"},
                ],
            }],
        ]
        result = fetch_ontology_class_map(convex_url="https://x.convex.cloud", ontology_id="onto1")
        assert result == {"cls1": "sign", "cls2": "phrase"}
        assert "cls3" not in result  # global should be skipped

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_maps_all_ontology_versions_in_group(self, mock_query):
        mock_query.side_effect = [
            {"ontologyGroupId": "group1", "objectClasses": []},
            [
                {"_id": "v1", "status": "published"},
                {"_id": "v2", "status": "published"},
            ],
            {
                "status": "published",
                "objectClasses": [
                    {"_id": "old_cls", "annotationType": "time_aligned", "type": "sign_language_sign"},
                ],
            },
            {
                "status": "published",
                "objectClasses": [
                    {"_id": "new_cls", "annotationType": "time_aligned", "type": "sign_language_sign"},
                ],
            },
        ]

        result = fetch_ontology_class_map(convex_url="https://x.convex.cloud", ontology_id="onto1")

        assert result == {"old_cls": "sign", "new_cls": "sign"}

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_uses_version_object_classes_when_present(self, mock_query):
        mock_query.side_effect = [
            {"ontologyGroupId": "group1", "objectClasses": []},
            [
                {
                    "_id": "v1",
                    "status": "published",
                    "objectClasses": [
                        {"_id": "old_cls", "annotationType": "time_aligned", "type": "sign_language_sign"},
                    ],
                },
            ],
        ]

        result = fetch_ontology_class_map(convex_url="https://x.convex.cloud", ontology_id="onto1")

        assert result == {"old_cls": "sign"}

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_falls_back_to_current_ontology_without_group(self, mock_query):
        mock_query.return_value = {
            "objectClasses": [
                {"_id": "cls1", "annotationType": "time_aligned", "type": "sign_language_sign"},
            ]
        }
        result = fetch_ontology_class_map(convex_url="https://x.convex.cloud", ontology_id="onto1")
        assert result == {"cls1": "sign"}

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_empty_ontology(self, mock_query):
        mock_query.return_value = {"objectClasses": []}
        result = fetch_ontology_class_map(convex_url="https://x.convex.cloud", ontology_id="onto1")
        assert result == {}

    @patch("sign_language_segmentation.datasets.annotation_platform.sync.convex_query")
    def test_filters_ontology_status(self, mock_query):
        mock_query.side_effect = [
            {"ontologyGroupId": "group1", "objectClasses": []},
            [{
                "status": "draft",
                "objectClasses": [
                    {"_id": "cls1", "annotationType": "time_aligned", "type": "sign_language_sign"},
                ],
            }],
        ]
        result = fetch_ontology_class_map(
            convex_url="https://x.convex.cloud",
            ontology_id="onto1",
            allowed_statuses={"published"},
        )
        assert result == {}


# -- sync ---------------------------------------------------------------------

class TestSync:
    @patch("sign_language_segmentation.datasets.annotation_platform.sync.resolve_video_paths")
    @patch("sign_language_segmentation.datasets.annotation_platform.sync.fetch_project_annotations")
    def test_creates_output_parent_directory(self, mock_fetch, mock_resolve, tmp_path: Path):
        output_path = tmp_path / "missing" / "annotation_platform" / "annotations_cache.json"
        mock_fetch.return_value = (
            {"video-1": [{"type": "sign", "start": 0, "end": 1000}]},
            ["dataset-1"],
        )
        mock_resolve.return_value = {
            "video-1": {"pose_hash": "hash1", "fps": 30.0, "total_frames": 100},
        }

        sync(
            convex_url="https://x.convex.cloud",
            project_ids=["project-1"],
            poses_dir=str(tmp_path / "poses"),
            gcs_root=str(tmp_path / "gcs"),
            output_path=output_path,
        )

        assert output_path.exists()
        cache = json.loads(output_path.read_text())
        assert set(cache["videos"]) == {"video-1"}
