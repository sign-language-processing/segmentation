"""slack notifier for publish results.

designed as a standalone module: `build_publish_message` returns a slack
block-kit payload, `post_to_slack` posts a payload via chat.postMessage, and
`notify` composes both for a one-shot call. integration into the publish
pipeline is intentionally deferred — consumers wire it in by reading
`SLACK_BOT_TOKEN` and `SLACK_CHANNEL` from their own environment.
"""

from __future__ import annotations

from typing import Any

import httpx

SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
DEFAULT_TIMEOUT_SECONDS = 10.0


def build_publish_message(
    *,
    repo_id: str,
    tag: str,
    revision: str,
    regression_status: str,
    metrics: dict[str, dict[str, float]] | None,
    deployed: bool,
) -> dict[str, Any]:
    """build a slack block-kit payload describing a publish result.

    `metrics` is nested: `{"<dataset>.<split>": {"<metric>": value}}`. the
    top-level `text` field is the plaintext fallback shown in notifications.
    """
    status = "deployed" if deployed else "not deployed"
    header_text = f"{repo_id} — {tag} — {status}"

    summary_fields = [
        {"type": "mrkdwn", "text": f"*tag*\n`{tag}`"},
        {"type": "mrkdwn", "text": f"*revision*\n`{revision}`"},
        {"type": "mrkdwn", "text": f"*regression*\n{regression_status}"},
        {"type": "mrkdwn", "text": f"*status*\n{status}"},
    ]

    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": header_text}},
        {"type": "section", "fields": summary_fields},
    ]

    if metrics:
        metric_lines = []
        for key in sorted(metrics):
            pairs = " · ".join(f"{k}=`{v:.4f}`" for k, v in sorted(metrics[key].items()))
            metric_lines.append(f"• *{key}*  {pairs}")
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(metric_lines)},
            }
        )

    repo_url = f"https://huggingface.co/{repo_id}"
    blocks.append(
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"<{repo_url}|open on huggingface>"}],
        }
    )

    return {"text": header_text, "blocks": blocks}


def post_to_slack(
    *,
    bot_token: str,
    channel: str,
    payload: dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """post a payload to chat.postMessage. raises on http or slack-level error."""
    body = {"channel": channel, **payload}
    resp = httpx.post(
        SLACK_POST_MESSAGE_URL,
        headers={
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=body,
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"slack api error: {data.get('error', 'unknown')}")
    return data


def notify(
    *,
    bot_token: str,
    channel: str,
    repo_id: str,
    tag: str,
    revision: str,
    regression_status: str,
    metrics: dict[str, dict[str, float]] | None,
    deployed: bool,
) -> dict[str, Any]:
    """one-shot: build a publish payload and post it to slack."""
    payload = build_publish_message(
        repo_id=repo_id,
        tag=tag,
        revision=revision,
        regression_status=regression_status,
        metrics=metrics,
        deployed=deployed,
    )
    return post_to_slack(bot_token=bot_token, channel=channel, payload=payload)
