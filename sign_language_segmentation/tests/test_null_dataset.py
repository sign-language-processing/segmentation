from pathlib import Path
from unittest.mock import patch

from sign_language_segmentation.datasets.null import sync
from sign_language_segmentation.datasets.common import Split
from sign_language_segmentation.datasets.null.dataset import NullSegmentationDataset


class TestNullSync:
    def test_sample_frame_counts_uses_ten_even_steps(self):
        assert sync._sample_frame_counts(max_frames=1000, sample_count=10) == (
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
        cache = sync.build_cache(
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

    def test_build_cache_keeps_sample_count_when_static_source_exists(self, tmp_path: Path):
        source_pose = tmp_path / "source.pose"

        def fake_static_pose(source_frame, num_frames: int, fps: float):
            return sync._make_blank_pose(num_frames=num_frames, fps=fps)

        with patch.object(sync, "_load_static_frame", return_value=object()) as load_static_frame:
            with patch.object(sync, "_make_static_pose", side_effect=fake_static_pose) as make_static_pose:
                cache = sync.build_cache(
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
        load_static_frame.assert_called_once_with(source_pose_path=source_pose, frame_index=0)
        assert make_static_pose.call_count == 2

    def test_generated_cache_loads_through_null_dataset(self, tmp_path: Path):
        sync.build_cache(
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

        assert len(dataset) == 3
        assert [item["total_frames"] for item in dataset.items] == [10, 20, 30]

        with patch("sign_language_segmentation.datasets.common.load_and_augment", return_value={"pose": "sample"}) as load:
            assert dataset[0] == {"pose": "sample"}
        load.assert_called_once_with(
            pose_path=dataset.items[0]["pose_path"],
            fps=30.0,
            total_frames=10,
            signs=[],
            sentences=[],
            split=Split.TRAIN,
            num_frames=1024,
            velocity=True,
            fps_aug=False,
            frame_dropout=0.0,
            body_part_dropout=0.0,
        )
