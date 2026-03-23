"""Smoke tests for inference pipeline."""
from pathlib import Path

import pytest
from pose_format import Pose

EXAMPLE_POSE = Path(__file__).parent / "example.pose"


@pytest.fixture
def example_pose():
    with open(EXAMPLE_POSE, "rb") as f:
        return Pose.read(f)


def test_segment_pose_returns_tiers(example_pose):
    from sign_language_segmentation.bin import segment_pose

    eaf, tiers = segment_pose(example_pose)

    assert "SIGN" in tiers
    assert "SENTENCE" in tiers
    assert isinstance(tiers["SIGN"], list)
    assert isinstance(tiers["SENTENCE"], list)


def test_segment_pose_segments_have_start_end(example_pose):
    from sign_language_segmentation.bin import segment_pose

    _, tiers = segment_pose(example_pose)

    for tier in tiers.values():
        for seg in tier:
            assert "start" in seg
            assert "end" in seg
            assert seg["end"] >= seg["start"]


def test_eaf_has_tiers(example_pose):
    from sign_language_segmentation.bin import segment_pose

    eaf, _ = segment_pose(example_pose)

    tier_names = list(eaf.tiers.keys())
    assert "SIGN" in tier_names
    assert "SENTENCE" in tier_names
