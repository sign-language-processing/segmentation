"""tests for the slack notifier module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from sign_language_segmentation.notifier import (
    build_publish_message,
    notify,
    post_to_slack,
)


class TestBuildPublishMessage:
    def test_deployed_message_contains_key_fields(self):
        payload = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="pass",
            metrics={"dgs.test": {"sign_IoU": 0.8521, "hm_IoU": 0.7812}},
            deployed=True,
        )
        rendered = str(payload)
        assert "sign/segmentation" in rendered
        assert "v2026.04.20" in rendered
        assert "abc1234" in rendered
        assert "pass" in rendered
        assert "deployed" in rendered
        assert "0.8521" in rendered
        assert "0.7812" in rendered

    def test_not_deployed_status(self):
        payload = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="fail",
            metrics=None,
            deployed=False,
        )
        assert "not deployed" in str(payload)
        assert "fail" in str(payload)

    def test_metrics_section_only_when_metrics_provided(self):
        without = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="skipped",
            metrics=None,
            deployed=False,
        )
        with_metrics = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="pass",
            metrics={"dgs.test": {"sign_IoU": 0.85}},
            deployed=True,
        )
        assert len(with_metrics["blocks"]) == len(without["blocks"]) + 1

    def test_fallback_text_for_notifications(self):
        payload = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="pass",
            metrics=None,
            deployed=True,
        )
        assert "sign/segmentation" in payload["text"]
        assert "v2026.04.20" in payload["text"]
        assert "deployed" in payload["text"]

    def test_metrics_sorted_deterministically(self):
        payload = build_publish_message(
            repo_id="sign/segmentation",
            tag="v2026.04.20",
            revision="abc1234",
            regression_status="pass",
            metrics={
                "platform.test": {"sign_IoU": 0.7},
                "dgs.test": {"sign_IoU": 0.8},
            },
            deployed=True,
        )
        metric_block = next(
            b for b in payload["blocks"]
            if b["type"] == "section" and "fields" not in b
        )
        text = metric_block["text"]["text"]
        assert text.index("dgs.test") < text.index("platform.test")


class TestPostToSlack:
    @staticmethod
    def _ok_response(**extra):
        resp = MagicMock()
        resp.json.return_value = {"ok": True, **extra}
        resp.raise_for_status.return_value = None
        return resp

    def test_happy_path_uses_bearer_and_channel(self):
        resp = self._ok_response(ts="1234.5678")
        with patch("httpx.post", return_value=resp) as mock_post:
            result = post_to_slack(
                bot_token="xoxb-test",
                channel="C12345",
                payload={"text": "hi", "blocks": []},
            )
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer xoxb-test"
        assert kwargs["json"]["channel"] == "C12345"
        assert kwargs["json"]["text"] == "hi"
        assert result == {"ok": True, "ts": "1234.5678"}

    def test_slack_api_error_raises(self):
        resp = MagicMock()
        resp.json.return_value = {"ok": False, "error": "channel_not_found"}
        resp.raise_for_status.return_value = None
        with patch("httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="channel_not_found"):
                post_to_slack(
                    bot_token="xoxb-test",
                    channel="#nope",
                    payload={"text": "hi"},
                )

    def test_http_error_propagates(self):
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=resp,
        )
        with patch("httpx.post", return_value=resp):
            with pytest.raises(httpx.HTTPStatusError):
                post_to_slack(
                    bot_token="xoxb-test",
                    channel="#general",
                    payload={"text": "hi"},
                )


class TestNotify:
    def test_notify_builds_and_posts(self):
        resp = MagicMock()
        resp.json.return_value = {"ok": True, "ts": "1.2"}
        resp.raise_for_status.return_value = None
        with patch("httpx.post", return_value=resp) as mock_post:
            notify(
                bot_token="xoxb-test",
                channel="C999",
                repo_id="sign/segmentation",
                tag="v2026.04.20",
                revision="abc1234",
                regression_status="pass",
                metrics={"dgs.test": {"sign_IoU": 0.85}},
                deployed=True,
            )
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["channel"] == "C999"
        assert "sign/segmentation" in kwargs["json"]["text"]
        assert "v2026.04.20" in kwargs["json"]["text"]
