"""Tests for publish() CLI orchestration with all HF + eval boundaries mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestPublishIntegration:
    """End-to-end publish() orchestration with every HF + eval boundary mocked."""

    @pytest.fixture
    def ckpt_fixture(self, tmp_path):
        import torch
        ckpt_path = tmp_path / "fake.ckpt"
        fake_ckpt = {
            "state_dict": {"layer.weight": torch.randn(2, 2)},
            "hyper_parameters": {
                "hidden_dim": 128,
                "encoder_depth": 4,
                "attn_nhead": 8,
                "attn_ff_mult": 4,
                "attn_dropout": 0.1,
                "num_frames": 1024,
                "pose_dims": (75, 3),
                "num_classes": 3,
                "learning_rate": 1e-4,
                "optimizer": "adam",
                "dice_loss_weight": 0.5,
                "fps_aug": True,
                "frame_dropout": 0.1,
            },
        }
        torch.save(fake_ckpt, ckpt_path)
        return str(ckpt_path)

    def _eval_results(self):
        return {
            "ds_a": {
                "dev": {"sign_IoU": 0.80, "sentence_IoU": 0.70},
                "test": {"sign_IoU": 0.81, "sentence_IoU": 0.71},
            },
            "combined": {
                "dev": {"sign_IoU": 0.82, "sentence_IoU": 0.72},
                "test": {"sign_IoU": 0.83, "sentence_IoU": 0.73},
            },
        }

    def _mock_api(self):
        mock_api = MagicMock()
        # no prior version tags — regression check short-circuits to no_baseline
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="weekly", target_commit="commit123")],
            branches=[],
        )
        return mock_api

    def test_skip_eval_and_no_promote(self, ckpt_fixture):
        from sign_language_segmentation.publish.publish import publish

        mock_api = self._mock_api()
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            publish(
                checkpoint=ckpt_fixture,
                repo_id="fake/repo",
                tag="v2026.4.20",
                datasets="ds_a",
                corpus="",
                poses="",
                device="cpu",
                skip_eval=True,
                metrics_json=None,
                regression_threshold=0.005,
                no_promote=True,
            )
        mock_api.create_repo.assert_called_once()
        mock_api.create_branch.assert_called_once()
        mock_api.upload_folder.assert_called_once()
        # no_promote=True — no tag should be created
        mock_api.create_tag.assert_not_called()

    def test_skip_eval_with_promote_creates_tag(self, ckpt_fixture):
        from sign_language_segmentation.publish.publish import publish

        mock_api = self._mock_api()
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            publish(
                checkpoint=ckpt_fixture,
                repo_id="fake/repo",
                tag="v2026.4.20",
                datasets="ds_a",
                corpus="",
                poses="",
                device="cpu",
                skip_eval=True,
                metrics_json=None,
                regression_threshold=0.005,
                no_promote=False,
            )
        mock_api.create_tag.assert_called_once()
        args, kwargs = mock_api.create_tag.call_args
        assert kwargs["tag"] == "v2026.4.20"

    def test_with_eval_passes_regression_and_promotes(self, ckpt_fixture, tmp_path):
        from sign_language_segmentation.publish.publish import publish

        mock_api = self._mock_api()
        # stub run_evaluation and check_regression at the publish.py binding site,
        # since publish.py imports them at module scope
        with patch("sign_language_segmentation.publish.publish.run_evaluation",
                   return_value=self._eval_results()), \
             patch("sign_language_segmentation.publish.publish.check_regression",
                   return_value=("pass", None)), \
             patch("huggingface_hub.HfApi", return_value=mock_api):
            publish(
                checkpoint=ckpt_fixture,
                repo_id="fake/repo",
                tag="v2026.4.20",
                datasets="ds_a",
                corpus="",
                poses="",
                device="cpu",
                skip_eval=False,
                metrics_json=None,
                regression_threshold=0.005,
                no_promote=False,
            )
        mock_api.upload_folder.assert_called_once()
        mock_api.create_tag.assert_called_once()

    def test_regression_fail_does_not_promote(self, ckpt_fixture):
        from sign_language_segmentation.publish.publish import publish

        mock_api = self._mock_api()
        with patch("sign_language_segmentation.publish.publish.run_evaluation",
                   return_value=self._eval_results()), \
             patch("sign_language_segmentation.publish.publish.check_regression",
                   return_value=("fail", None)), \
             patch("huggingface_hub.HfApi", return_value=mock_api):
            publish(
                checkpoint=ckpt_fixture,
                repo_id="fake/repo",
                tag="v2026.4.20",
                datasets="ds_a",
                corpus="",
                poses="",
                device="cpu",
                skip_eval=False,
                metrics_json=None,
                regression_threshold=0.005,
                no_promote=False,
            )
        mock_api.upload_folder.assert_called_once()
        # regression failed — no promotion
        mock_api.create_tag.assert_not_called()

    def test_dry_run_writes_card_and_skips_hf(self, ckpt_fixture, capsys, tmp_path, monkeypatch):
        from sign_language_segmentation.publish.publish import publish

        monkeypatch.chdir(tmp_path)
        mock_api = self._mock_api()
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            publish(
                checkpoint=ckpt_fixture,
                repo_id="fake/repo",
                tag="v2026.4.20",
                datasets="ds_a",
                corpus="",
                poses="",
                device="cpu",
                skip_eval=True,
                metrics_json=None,
                regression_threshold=0.005,
                no_promote=False,
                dry_run=True,
            )
        mock_api.create_repo.assert_not_called()
        mock_api.create_branch.assert_not_called()
        mock_api.upload_folder.assert_not_called()
        mock_api.create_tag.assert_not_called()
        preview = tmp_path / "publish_dry_run.md"
        assert preview.exists()
        out = capsys.readouterr().out
        assert "new tag: v2026.4.20" in out
        assert "previous: none" in out
        assert "revision: weekly" in out
        assert "regression: skipped" in out
