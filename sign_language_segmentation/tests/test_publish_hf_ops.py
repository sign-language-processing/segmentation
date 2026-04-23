"""Tests for HF-facing publish helpers: regression check and promotion."""

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

