"""Smoke tests for inference pipeline."""
import json
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from pose_format import Pose
from safetensors.torch import save_file as save_safetensors

from sign_language_segmentation.model.model import PoseTaggingModel

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


@pytest.fixture
def tiny_model_config() -> dict:
    return {
        "pose_dims": (50, 6),
        "hidden_dim": 8,
        "encoder_depth": 1,
        "attn_nhead": 2,
        "attn_ff_mult": 1,
    }


def test_load_model_reads_safetensors_config_json(tmp_path: Path, tiny_model_config: dict):
    from sign_language_segmentation.bin import load_model

    model = PoseTaggingModel(**tiny_model_config)
    save_safetensors(tensors=model.state_dict(), filename=str(tmp_path / "model.safetensors"))
    config_json = {**tiny_model_config, "pose_dims": list(tiny_model_config["pose_dims"])}
    (tmp_path / "config.json").write_text(json.dumps(obj=config_json))

    loaded = load_model(model_dir=str(tmp_path), device="cpu")

    assert loaded.training is False
    assert loaded.hparams.pose_dims == tiny_model_config["pose_dims"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(input=loaded.state_dict()[name], other=tensor)


def test_load_model_accepts_external_safetensors_config(tmp_path: Path, tiny_model_config: dict):
    from sign_language_segmentation.bin import load_model

    model = PoseTaggingModel(**tiny_model_config)
    model_path = tmp_path / "model.safetensors"
    save_safetensors(tensors=model.state_dict(), filename=str(model_path))

    loaded = load_model(
        model_dir=str(model_path),
        device="cpu",
        config_overrides=tiny_model_config,
        eval_mode=False,
    )

    assert loaded.training is True
    assert loaded.hparams.pose_dims == tiny_model_config["pose_dims"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(input=loaded.state_dict()[name], other=tensor)


def test_load_model_ckpt_with_config_overrides(tmp_path: Path, tiny_model_config: dict):
    from sign_language_segmentation.bin import load_model

    model = PoseTaggingModel(**tiny_model_config)
    ckpt_path = tmp_path / "best.ckpt"
    torch.save(
        obj={
            "state_dict": model.state_dict(),
            "hyper_parameters": tiny_model_config,
            "pytorch-lightning_version": pl.__version__,
        },
        f=str(ckpt_path),
    )

    overrides = {**tiny_model_config, "learning_rate": 5e-4}
    loaded = load_model(
        model_dir=str(ckpt_path),
        device="cpu",
        config_overrides=overrides,
        eval_mode=False,
    )

    assert loaded.training is True
    assert loaded.hparams.learning_rate == 5e-4
    assert loaded.hparams.pose_dims == tiny_model_config["pose_dims"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(input=loaded.state_dict()[name], other=tensor)


def test_load_model_raises_when_config_missing_and_no_overrides(tmp_path: Path, tiny_model_config: dict):
    from sign_language_segmentation.bin import load_model

    model = PoseTaggingModel(**tiny_model_config)
    save_safetensors(tensors=model.state_dict(), filename=str(tmp_path / "model.safetensors"))

    with pytest.raises(FileNotFoundError, match="config.json"):
        load_model(model_dir=str(tmp_path), device="cpu")


def test_load_model_safetensors_dir_without_config_uses_overrides(tmp_path: Path, tiny_model_config: dict):
    from sign_language_segmentation.bin import load_model

    model = PoseTaggingModel(**tiny_model_config)
    save_safetensors(tensors=model.state_dict(), filename=str(tmp_path / "model.safetensors"))

    loaded = load_model(
        model_dir=str(tmp_path),
        device="cpu",
        config_overrides=tiny_model_config,
        eval_mode=False,
    )

    assert loaded.training is True
    assert loaded.hparams.pose_dims == tiny_model_config["pose_dims"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(input=loaded.state_dict()[name], other=tensor)
