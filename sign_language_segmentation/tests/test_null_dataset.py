from pathlib import Path

from pose_format import Pose

from sign_language_segmentation.datasets.common import Split
from sign_language_segmentation.datasets.null.dataset import NullSegmentationDataset
from sign_language_segmentation.datasets.null.sync import _sample_frame_counts, build_cache
from sign_language_segmentation.utils.bio import BIO


class TestNullSync:
    def test_sample_frame_counts_uses_ten_even_steps(self):
        assert _sample_frame_counts(max_frames=1000, sample_count=10) == (
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        )

    def test_build_cache_generates_blank_samples_up_to_max_frames(self, tmp_path: Path):
        cache = build_cache(
            output_path=tmp_path / "annotations_cache.json",
            poses_dir=tmp_path / "poses",
            max_frames=30,
            sample_count=3,
            fps=30.0,
            force=True,
        )

        videos = list(cache["videos"].values())
        assert [video["total_frames"] for video in videos] == [10, 20, 30]
        assert {video["kind"] for video in videos} == {"blank"}
        assert all(Path(video["pose_path"]).exists() for video in videos)
        with open(videos[0]["pose_path"], "rb") as f:
            pose = Pose.read(f, start_frame=0, end_frame=videos[0]["total_frames"])
        assert len(pose.body.data) == videos[0]["total_frames"]

    def test_build_cache_keeps_sample_count_when_static_source_exists(self, tmp_path: Path):
        source_cache = build_cache(
            output_path=tmp_path / "source_cache.json",
            poses_dir=tmp_path / "source_poses",
            max_frames=5,
            sample_count=1,
            fps=30.0,
            force=True,
        )
        source_pose = Path(next(iter(source_cache["videos"].values()))["pose_path"])

        cache = build_cache(
            output_path=tmp_path / "annotations_cache.json",
            poses_dir=tmp_path / "poses",
            max_frames=40,
            sample_count=4,
            fps=30.0,
            force=True,
            static_source_pose=source_pose,
        )

        videos = list(cache["videos"].values())
        assert [video["total_frames"] for video in videos] == [10, 20, 30, 40]
        assert [video["kind"] for video in videos] == ["blank", "static", "blank", "static"]
        for video in videos:
            with open(video["pose_path"], "rb") as f:
                pose = Pose.read(f, start_frame=0, end_frame=video["total_frames"])
            assert len(pose.body.data) == video["total_frames"]

    def test_generated_cache_loads_through_null_dataset(self, tmp_path: Path):
        build_cache(
            output_path=tmp_path / "annotations_cache.json",
            poses_dir=tmp_path / "poses",
            max_frames=30,
            sample_count=3,
            fps=30.0,
            force=True,
        )

        dataset = NullSegmentationDataset(
            annotations_path=str(tmp_path / "annotations_cache.json"),
            split=Split.TRAIN,
            velocity=True,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
            dev_ratio=0.0,
            test_ratio=0.0,
        )
        sample = dataset[0]

        assert len(dataset) == 3
        assert sample["pose"].shape == (10, 50, 6)
        assert sample["bio"]["sign"].unique().tolist() == [BIO["O"]]
