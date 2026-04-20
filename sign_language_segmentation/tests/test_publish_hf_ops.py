"""Tests for evaluation, regression check, promotion, and publish() orchestration."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sign_language_segmentation.publish.utils import check_regression, promote


class TestCheckRegression:
    def test_no_baseline_when_no_tags(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(tags=[], branches=[])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            status, baseline = check_regression(new_metrics={}, repo_id="fake/repo", threshold=0.005)
        assert status == "no_baseline"
        assert baseline is None

    def _make_hf_http_error(self, status_code: int, msg: str):
        from huggingface_hub.utils import HfHubHTTPError
        response = MagicMock()
        response.status_code = status_code
        response.headers = {}
        return HfHubHTTPError(msg, response=response)

    def test_no_baseline_when_download_404s(self):
        # tag exists but eval_results.json was never uploaded — legitimate "no baseline"
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="v2026.1.1")], branches=[]
        )
        mock_api.hf_hub_download.side_effect = self._make_hf_http_error(status_code=404, msg="File not found")
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            status, baseline = check_regression(
                new_metrics={"combined": {"test": {"sign_IoU": 0.8}}},
                repo_id="fake/repo",
                threshold=0.005,
            )
        assert status == "no_baseline"
        assert baseline is None

    def test_non_404_download_error_propagates(self):
        # auth/network errors are real failures, not "no baseline"
        from huggingface_hub.utils import HfHubHTTPError
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="v2026.1.1")], branches=[]
        )
        mock_api.hf_hub_download.side_effect = self._make_hf_http_error(status_code=401, msg="Unauthorized")
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            with pytest.raises(HfHubHTTPError):
                check_regression(
                    new_metrics={"combined": {"test": {"sign_IoU": 0.8}}},
                    repo_id="fake/repo",
                    threshold=0.005,
                )

    def test_pass_when_within_threshold(self, tmp_path):
        baseline_file = tmp_path / "eval_results.json"
        baseline_file.write_text(
            json.dumps({"combined": {"test": {"sign_IoU": 0.80, "sentence_IoU": 0.70}}})
        )
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="v2026.1.1")], branches=[]
        )
        mock_api.hf_hub_download.return_value = str(baseline_file)
        new_metrics = {"combined": {"test": {"sign_IoU": 0.798, "sentence_IoU": 0.70}}}
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            status, baseline = check_regression(
                new_metrics=new_metrics, repo_id="fake/repo", threshold=0.005
            )
        assert status == "pass"
        assert baseline == {"combined": {"test": {"sign_IoU": 0.80, "sentence_IoU": 0.70}}}

    def test_fail_when_beyond_threshold(self, tmp_path):
        baseline_file = tmp_path / "eval_results.json"
        baseline_file.write_text(
            json.dumps({"combined": {"test": {"sign_IoU": 0.80, "sentence_IoU": 0.70}}})
        )
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="v2026.1.1")], branches=[]
        )
        mock_api.hf_hub_download.return_value = str(baseline_file)
        new_metrics = {"combined": {"test": {"sign_IoU": 0.75, "sentence_IoU": 0.70}}}
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            status, baseline = check_regression(
                new_metrics=new_metrics, repo_id="fake/repo", threshold=0.005
            )
        assert status == "fail"
        assert baseline is not None


class TestPromote:
    def test_tag_found_uses_tag_commit(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[SimpleNamespace(name="weekly", target_commit="abc12345deadbeef")],
            branches=[],
        )
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            promote(repo_id="fake/repo", tag="v2026.4.20", revision="weekly")
        mock_api.create_tag.assert_called_once_with(
            repo_id="fake/repo", tag="v2026.4.20", revision="abc12345deadbeef"
        )

    def test_branch_found_uses_branch_commit(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(
            tags=[],
            branches=[SimpleNamespace(name="weekly", target_commit="feedface00000000")],
        )
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            promote(repo_id="fake/repo", tag="v2026.4.20", revision="weekly")
        mock_api.create_tag.assert_called_once_with(
            repo_id="fake/repo", tag="v2026.4.20", revision="feedface00000000"
        )

    def test_raises_when_unresolved(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = SimpleNamespace(tags=[], branches=[])
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            with pytest.raises(ValueError, match="Could not resolve revision"):
                promote(repo_id="fake/repo", tag="v2026.4.20", revision="nonexistent")
        mock_api.create_tag.assert_not_called()


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
