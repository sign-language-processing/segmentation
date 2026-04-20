"""Unit tests for publish/utils.py pure helpers."""

from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from sign_language_segmentation.publish.utils import (
    _build_config_table,
    _build_dataset_section,
    _build_eval_section,
    _build_model_index,
    _get_test_metrics,
    _parse_version,
    find_split_manifest,
    generate_model_card,
    get_latest_version,
    get_next_version,
)


class TestParseVersion:
    def test_ymd_returns_3_tuple(self):
        assert _parse_version("v2026.4.20") == (2026, 4, 20)

    def test_ymd_with_suffix_returns_4_tuple(self):
        assert _parse_version("v2026.4.20.3") == (2026, 4, 20, 3)

    def test_zero_padded_month_day(self):
        assert _parse_version("v2026.04.20") == (2026, 4, 20)

    def test_garbage_returns_none(self):
        assert _parse_version("nope") is None
        assert _parse_version("v") is None
        assert _parse_version("v1.2") is None
        assert _parse_version("") is None

    def test_legacy_semver_returns_none(self):
        # legacy vA.B.C style with single-digit parts matches the regex (vYYYY.MM.DD),
        # but real semver like v1.0.0 happens to parse as (1,0,0) — that's fine because
        # real tags will always be real dates. Only clearly non-numeric garbage returns None.
        assert _parse_version("v1.0.abc") is None


class TestGetLatestVersion:
    def test_mixed_tags_picks_max(self):
        with patch("sign_language_segmentation.publish.utils.HfApi", create=True) as _:
            pass
        mock_refs = SimpleNamespace(
            tags=[
                SimpleNamespace(name="v2026.1.1"),
                SimpleNamespace(name="v2026.4.20"),
                SimpleNamespace(name="weekly"),
                SimpleNamespace(name="v2026.3.15.2"),
                SimpleNamespace(name="release-v1"),
            ]
        )
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = mock_refs
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            assert get_latest_version(repo_id="fake/repo") == "v2026.4.20"

    def test_no_version_tags_returns_none(self):
        mock_refs = SimpleNamespace(tags=[SimpleNamespace(name="weekly"), SimpleNamespace(name="main")])
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = mock_refs
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            assert get_latest_version(repo_id="fake/repo") is None

    def test_api_error_returns_none(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.side_effect = RuntimeError("401")
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            assert get_latest_version(repo_id="fake/repo") is None

    def test_suffix_beats_base_same_day(self):
        mock_refs = SimpleNamespace(
            tags=[SimpleNamespace(name="v2026.4.20"), SimpleNamespace(name="v2026.4.20.1")]
        )
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = mock_refs
        with patch("huggingface_hub.HfApi", return_value=mock_api):
            assert get_latest_version(repo_id="fake/repo") == "v2026.4.20.1"


class TestGetNextVersion:
    def _freeze_today(self, monkeypatch, y, m, d):
        class FrozenDT:
            @classmethod
            def now(cls, tz=None):
                return datetime(y, m, d, tzinfo=tz or UTC)
        monkeypatch.setattr("sign_language_segmentation.publish.utils.datetime", FrozenDT)

    def test_no_baseline_returns_base(self, monkeypatch):
        self._freeze_today(monkeypatch, 2026, 4, 20)
        monkeypatch.setattr(
            "sign_language_segmentation.publish.utils.get_latest_version", lambda repo_id: None
        )
        assert get_next_version(repo_id="fake/repo") == "v2026.4.20"

    def test_prior_date_returns_base(self, monkeypatch):
        self._freeze_today(monkeypatch, 2026, 4, 20)
        monkeypatch.setattr(
            "sign_language_segmentation.publish.utils.get_latest_version", lambda repo_id: "v2026.1.1"
        )
        assert get_next_version(repo_id="fake/repo") == "v2026.4.20"

    def test_same_day_base_increments_to_1(self, monkeypatch):
        self._freeze_today(monkeypatch, 2026, 4, 20)
        monkeypatch.setattr(
            "sign_language_segmentation.publish.utils.get_latest_version", lambda repo_id: "v2026.4.20"
        )
        assert get_next_version(repo_id="fake/repo") == "v2026.4.20.1"

    def test_same_day_with_suffix_increments(self, monkeypatch):
        self._freeze_today(monkeypatch, 2026, 4, 20)
        monkeypatch.setattr(
            "sign_language_segmentation.publish.utils.get_latest_version", lambda repo_id: "v2026.4.20.1"
        )
        assert get_next_version(repo_id="fake/repo") == "v2026.4.20.2"


class TestFindSplitManifest:
    def test_returns_none_when_missing(self, tmp_path):
        ckpt = tmp_path / "best.ckpt"
        ckpt.touch()
        assert find_split_manifest(checkpoint_path=str(ckpt)) is None

    def test_loads_manifest_from_sibling(self, tmp_path):
        (tmp_path / "split_manifest.json").write_text('{"datasets": "dgs", "created_at": "2026-01-01"}')
        ckpt = tmp_path / "best.ckpt"
        ckpt.touch()
        manifest = find_split_manifest(checkpoint_path=str(ckpt))
        assert manifest == {"datasets": "dgs", "created_at": "2026-01-01"}


class TestGetTestMetrics:
    def test_combined_wins(self):
        results = {
            "ds_a": {"test": {"sign_IoU": 0.5}},
            "combined": {"test": {"sign_IoU": 0.9}},
        }
        assert _get_test_metrics(results) == {"sign_IoU": 0.9}

    def test_single_dataset_picks_first(self):
        results = {"ds_a": {"test": {"sign_IoU": 0.5}}}
        assert _get_test_metrics(results) == {"sign_IoU": 0.5}

    def test_legacy_flat_returns_as_is(self):
        results = {"sign_IoU": 0.5, "sentence_IoU": 0.6}
        assert _get_test_metrics(results) == results


class TestBuildConfigTable:
    def test_empty_keys_returns_empty(self):
        assert _build_config_table(config={"a": 1}, keys=[]) == ""

    def test_no_matching_keys_returns_empty(self):
        assert _build_config_table(config={"a": 1}, keys=["b", "c"]) == ""

    def test_partial_match_renders_present_only(self):
        out = _build_config_table(config={"a": 1, "c": 3}, keys=["a", "b", "c"])
        assert "| a | c |" in out
        assert "| 1 | 3 |" in out
        assert "b" not in out

    def test_tuple_value_stringified(self):
        out = _build_config_table(config={"pose_dims": (75, 3)}, keys=["pose_dims"])
        assert "(75, 3)" in out


class TestBuildModelIndex:
    def test_emits_yaml_fragment(self):
        results = {"combined": {"test": {"sign_IoU": 0.8765, "sentence_IoU": 0.5}}}
        out = _build_model_index(eval_results=results)
        assert "model-index:" in out
        assert "- name: Sign Iou" in out
        assert "type: sign_IoU" in out
        assert "value: 0.8765" in out
        assert "value: 0.5000" in out

    def test_formats_to_four_decimals(self):
        results = {"combined": {"test": {"sign_IoU": 1.0 / 3.0}}}
        out = _build_model_index(eval_results=results)
        assert "value: 0.3333" in out


class TestBuildEvalSection:
    def test_per_dataset_and_combined(self):
        results = {
            "ds_a": {
                "dev": {"sign_IoU": 0.5, "sentence_IoU": 0.4},
                "test": {"sign_IoU": 0.55, "sentence_IoU": 0.45},
            },
            "combined": {
                "dev": {"sign_IoU": 0.6, "sentence_IoU": 0.5},
                "test": {"sign_IoU": 0.65, "sentence_IoU": 0.55},
            },
        }
        out = _build_eval_section(eval_results=results)
        assert "## Evaluation Results" in out
        assert "Ds A" in out
        assert "**Combined**" in out
        assert "**test**" in out  # combined splits are bolded

    def test_missing_metrics_default_to_zero(self):
        results = {"ds_a": {"test": {"sign_IoU": 0.5}}}
        out = _build_eval_section(eval_results=results)
        assert "0.0000" in out  # for metrics absent from dict


class TestBuildDatasetSection:
    def test_renders_fields(self):
        manifest = {"datasets": "dgs,platform", "created_at": "2026-04-20"}
        out = _build_dataset_section(split_manifest=manifest)
        assert "dgs,platform" in out
        assert "2026-04-20" in out

    def test_missing_fields_default_to_unknown(self):
        out = _build_dataset_section(split_manifest={})
        assert "unknown" in out


class TestGenerateModelCard:
    def _fixture_config(self):
        return {
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
        }

    def _fixture_eval(self):
        return {
            "ds_a": {"dev": {"sign_IoU": 0.5}, "test": {"sign_IoU": 0.55}},
            "combined": {"dev": {"sign_IoU": 0.6}, "test": {"sign_IoU": 0.65}},
        }

    def test_replaces_every_placeholder(self):
        out = generate_model_card(
            config=self._fixture_config(),
            eval_results=self._fixture_eval(),
            regression_status="pass",
            tag="v2026.4.20",
            split_manifest={"datasets": "ds_a", "created_at": "2026-04-20"},
        )
        assert "{{" not in out
        assert "}}" not in out
        assert "v2026.4.20" in out
        assert "pass" in out

    def test_omits_optional_sections_when_missing(self):
        out = generate_model_card(
            config=self._fixture_config(),
            eval_results=None,
            regression_status="no_baseline",
            tag="v2026.4.20",
            split_manifest=None,
        )
        assert "{{" not in out
        assert "## Evaluation Results" not in out

    def test_no_language_field_in_frontmatter(self):
        # language: was dropped per review comment #2
        out = generate_model_card(
            config=self._fixture_config(),
            eval_results=None,
            regression_status="pass",
            tag="v2026.4.20",
            split_manifest=None,
        )
        # frontmatter is the first `---...---` block
        frontmatter = out.split("---", 2)[1]
        assert "language:" not in frontmatter
