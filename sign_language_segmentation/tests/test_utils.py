"""Tests for utils.bio and utils.pose modules."""
import numpy as np
import pytest

from sign_language_segmentation.utils.bio import BIO, create_bio, create_bio_from_times


class TestBIOConstants:
    def test_bio_values_are_distinct(self):
        assert len(set(BIO.values())) == 4

    def test_bio_has_expected_keys(self):
        assert set(BIO.keys()) == {"UNK", "O", "B", "I"}


class TestCreateBio:
    def test_empty_annotations_all_outside(self):
        bio = create_bio(annotations=[], num_frames=10, fps=25.0)
        assert len(bio) == 10
        assert all(b == BIO["O"] for b in bio)

    def test_single_annotation_marks_b_and_i(self):
        # annotation from 0ms to 200ms at 25fps -> frames 0 to 5
        bio = create_bio(annotations=[{"start": 0, "end": 200}], num_frames=10, fps=25.0)
        assert bio[0] == BIO["B"]
        assert all(bio[i] == BIO["I"] for i in range(1, 6))
        assert all(bio[i] == BIO["O"] for i in range(6, 10))

    def test_multiple_annotations(self):
        bio = create_bio(
            annotations=[{"start": 0, "end": 40}, {"start": 200, "end": 240}],
            num_frames=10,
            fps=50.0,
        )
        b_indices = [i for i in range(10) if bio[i] == BIO["B"]]
        assert len(b_indices) == 2


class TestCreateBioFromTimes:
    def test_empty_annotations(self):
        frame_times_ms = np.arange(10, dtype=np.float64) * 40  # 25fps
        bio = create_bio_from_times(annotations=[], frame_times_ms=frame_times_ms)
        assert len(bio) == 10
        assert all(b == BIO["O"] for b in bio)

    def test_single_annotation(self):
        frame_times_ms = np.arange(20, dtype=np.float64) * 40  # 25fps, 0-760ms
        bio = create_bio_from_times(
            annotations=[{"start": 100, "end": 300}],
            frame_times_ms=frame_times_ms,
        )
        assert BIO["B"] in bio
        assert any(bio == BIO["I"])
        # frames outside annotation should be O
        assert bio[0] == BIO["O"]
        assert bio[-1] == BIO["O"]

    def test_non_uniform_spacing(self):
        # simulate dropped frames: gaps at different intervals
        frame_times_ms = np.array([0, 40, 80, 200, 240, 400, 440], dtype=np.float64)
        bio = create_bio_from_times(
            annotations=[{"start": 50, "end": 250}],
            frame_times_ms=frame_times_ms,
        )
        assert len(bio) == 7
        assert bio[0] == BIO["O"]  # 0ms is before 50ms
        assert bio[-1] == BIO["O"]  # 440ms is after 250ms


class TestComputeVelocity:
    def test_zero_velocity_for_static_pose(self):
        from sign_language_segmentation.utils.pose import compute_velocity

        pose_data = np.ones((5, 3, 3), dtype=np.float32)  # static
        times = np.arange(5, dtype=np.float32) / 25.0
        vel = compute_velocity(pose_data, times)
        assert vel.shape == pose_data.shape
        # all velocity should be zero (constant position)
        np.testing.assert_allclose(vel, 0.0, atol=1e-6)

    def test_linear_motion_constant_velocity(self):
        from sign_language_segmentation.utils.pose import compute_velocity

        # linear motion: position increases by 1 unit per frame at 25fps
        pose_data = np.arange(5, dtype=np.float32).reshape(5, 1, 1) * np.ones((1, 2, 3))
        times = np.arange(5, dtype=np.float32) / 25.0
        vel = compute_velocity(pose_data, times)
        # velocity should be 25 units/second (1 unit per 0.04s)
        np.testing.assert_allclose(vel[0], 0.0)  # first frame always zero
        np.testing.assert_allclose(vel[1:], 25.0, atol=1e-4)

    def test_single_frame_returns_zeros(self):
        from sign_language_segmentation.utils.pose import compute_velocity

        pose_data = np.ones((1, 3, 3), dtype=np.float32)
        times = np.array([0.0], dtype=np.float32)
        vel = compute_velocity(pose_data, times)
        np.testing.assert_allclose(vel, 0.0)
