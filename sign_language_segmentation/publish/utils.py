"""Utility functions for model publishing: conversion, evaluation, regression, model card."""

import argparse
import json
import re
from datetime import datetime, UTC
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors


# display names for metrics used in both the YAML model-index and the markdown eval table.
# keys are the raw metric keys emitted by evaluation; values are the human-readable labels.
_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "sign_IoU": "Sign IoU",
    "sentence_IoU": "Sentence IoU",
    "hm_IoU": "HM IoU",
    "sign_frame_f1": "Sign Frame F1",
    "sign_segment_f1": "Sign Segment F1",
    "sentence_frame_f1": "Sentence Frame F1",
    "sentence_segment_f1": "Sentence Segment F1",
}


def convert_to_safetensors(checkpoint_path: str, output_dir: Path) -> dict:
    """Convert .ckpt to safetensors + config.json. Returns hyper_parameters dict."""
    # Lightning .ckpt files carry hyper_parameters alongside weights; weights_only=True would reject them.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sd_bf16 = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
    save_safetensors(tensors=sd_bf16, filename=str(output_dir / "model.safetensors"))

    config = ckpt["hyper_parameters"]
    config = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config


def find_split_manifest(checkpoint_path: str) -> dict | None:
    """Look for split_manifest.json in the same directory as the checkpoint."""
    manifest_path = Path(checkpoint_path).parent / "split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def _parse_version(tag_name: str) -> tuple[int, ...] | None:
    """Parse a vYYYY.MM.DD[.N] tag. Returns None if not a version tag."""
    version_re = re.compile(r"^v(\d{4})\.(\d{1,2})\.(\d{1,2})(?:\.(\d+))?$")
    m = version_re.match(tag_name)
    if not m:
        return None
    parts = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if m.group(4) is not None:
        return (*parts, int(m.group(4)))
    return parts


def get_latest_version(repo_id: str) -> str | None:
    """Get the latest date-based version tag from the HF repo. Returns None if no version tags exist."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id=repo_id)
    except HfHubHTTPError as e:
        # catch HfHubHTTPError (not bare Exception) so auth/network errors surface;
        # only 404 (repo not found — first publish) is legitimately "no versions".
        if e.response.status_code == 404:
            return None
        raise
    versions = []
    for tag in refs.tags:
        parsed = _parse_version(tag.name)
        if parsed:
            versions.append((parsed, tag.name))
    if not versions:
        return None
    versions.sort()
    return versions[-1][1]


def get_next_version(repo_id: str) -> str:
    """Determine the next version tag based on today's date (vYYYY.MM.DD).

    If a tag for today already exists, appends a suffix (vYYYY.MM.DD.1, .2, ...).
    """
    today = datetime.now(tz=UTC)
    base = f"v{today.year}.{today.month}.{today.day}"

    latest = get_latest_version(repo_id=repo_id)
    if latest is None:
        return base

    parsed = _parse_version(latest)
    year, month, day = parsed[0], parsed[1], parsed[2]
    if (year, month, day) != (today.year, today.month, today.day):
        return base

    # same day — increment suffix
    suffix = parsed[3] + 1 if len(parsed) == 4 else 1
    return f"{base}.{suffix}"


def _get_test_metrics(eval_results: dict) -> dict:
    """Extract flat test metrics from eval_results (handles both nested and legacy flat format)."""
    if "combined" in eval_results:
        return eval_results["combined"].get("test", {})
    # single dataset
    for key, val in eval_results.items():
        if isinstance(val, dict) and "test" in val:
            return val["test"]
    # legacy flat format (no nesting)
    return eval_results


def _build_config_table(config: dict, keys: list[str]) -> str:
    """Render config keys as a single-row transposed table (keys as columns)."""
    present = [(k, config[k]) for k in keys if k in config]
    if not present:
        return ""
    header = "| " + " | ".join(k for k, _ in present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    values = "| " + " | ".join(str(v) for _, v in present) + " |"
    return f"{header}\n{separator}\n{values}"


def _build_model_index(eval_results: dict) -> str:
    test_metrics = _get_test_metrics(eval_results)
    lines = [
        "model-index:",
        "  - name: sign-language-segmentation",
        "    results:",
        "      - task:",
        "          type: other",
        "          name: Sign Language Segmentation",
        "        metrics:",
    ]
    for key, value in test_metrics.items():
        display_name = _METRIC_DISPLAY_NAMES.get(key, key)
        lines.append(f"          - name: {display_name}")
        lines.append(f"            type: {key}")
        lines.append(f"            value: {value:.4f}")
    return "\n".join(lines)


def _build_eval_section(eval_results: dict) -> str:
    metric_keys = list(_METRIC_DISPLAY_NAMES.keys())
    headers = ["Dataset", "Split"] + [_METRIC_DISPLAY_NAMES[k] for k in metric_keys]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    # separate per-dataset entries from "combined"
    ds_names = [k for k in eval_results if k != "combined" and isinstance(eval_results[k], dict)]
    if "combined" in eval_results:
        ds_names.append("combined")

    for ds_name in ds_names:
        splits = eval_results[ds_name]
        if not isinstance(splits, dict) or "test" not in splits:
            continue
        is_combined = ds_name == "combined"
        display_name = ds_name.replace("_", " ").title()
        if is_combined:
            display_name = f"**{display_name}**"

        for i, split_name in enumerate(["dev", "test"]):
            if split_name not in splits:
                continue
            metrics = splits[split_name]
            name_cell = display_name if i == 0 else ""
            split_cell = split_name
            if is_combined:
                split_cell = f"**{split_name}**"
            values = [f"{metrics.get(key, 0):.4f}" for key in metric_keys]
            if is_combined:
                values = [f"**{v}**" for v in values]
            rows.append("| " + " | ".join([name_cell, split_cell] + values) + " |")

    return f"## Evaluation Results\n\n{header_row}\n{separator}\n" + "\n".join(rows)


def _build_dataset_section(split_manifest: dict) -> str:
    datasets = split_manifest.get("datasets", "unknown")
    created_at = split_manifest.get("created_at", "unknown")
    return f"## Dataset\n\n- **Datasets:** {datasets}\n- **Created at:** {created_at}"


def generate_model_card(
    config: dict,
    eval_results: dict | None,
    regression_status: str,
    tag: str,
    repo_id: str,
    split_manifest: dict | None,
) -> str:
    """Generate a HuggingFace model card from template."""
    template_path = Path(__file__).resolve().parent / "model_card_template.md"
    template = template_path.read_text()

    arch_keys = [
        "hidden_dim",
        "encoder_depth",
        "attn_nhead",
        "attn_ff_mult",
        "attn_dropout",
        "num_frames",
        "pose_dims",
        "num_classes",
    ]
    train_keys = ["learning_rate", "optimizer", "dice_loss_weight", "fps_aug", "frame_dropout"]

    replacements = {
        "{{published_at}}": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "{{tag}}": tag,
        "{{repo_id}}": repo_id,
        "{{regression_status}}": regression_status,
        "{{architecture_rows}}": _build_config_table(config=config, keys=arch_keys),
        "{{training_rows}}": _build_config_table(config=config, keys=train_keys),
        "{{model_index}}": _build_model_index(eval_results=eval_results) if eval_results else "",
        "{{eval_section}}": _build_eval_section(eval_results=eval_results) if eval_results else "",
        "{{dataset_section}}": _build_dataset_section(split_manifest=split_manifest) if split_manifest else "",
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    return template


def _eval_single(model, datasets: str, split: str, eval_args, fps_aug: bool, velocity: bool, device: str) -> dict:
    """Evaluate on a single dataset+split combination."""
    from sign_language_segmentation.evaluate import evaluate_model
    from sign_language_segmentation.datasets.common import Split, get_dataloader

    dataloader = get_dataloader(
        split=Split(split),
        dataset_names=datasets,
        args=eval_args,
        batch_size=1,
        num_frames=999999,
        persistent_workers=False,
        fps_aug=fps_aug,
        velocity=velocity,
    )
    return evaluate_model(model=model, dataloader=dataloader, device=device)


def run_evaluation(
    checkpoint_path: str,
    datasets: str,
    corpus: str,
    poses: str,
    device: str,
    split_manifest: dict | None = None,
) -> dict:
    """Run model evaluation on each dataset individually and combined, for dev and test.

    Returns nested dict: {dataset_name: {split: {metric: value}}}.
    Top-level keys include each individual dataset, plus "combined".
    """
    from sign_language_segmentation.datasets.common import DATASET_REGISTRY, ensure_datasets_registered
    from sign_language_segmentation.model.model import PoseTaggingModel

    model = PoseTaggingModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
    model = model.to(device)

    # fall back to training defaults from args.py (both default to True there) when a legacy
    # ckpt doesn't carry these in hparams — keeps eval consistent with how the model was trained.
    fps_aug = getattr(model.hparams, "fps_aug", True)
    velocity = getattr(model.hparams, "velocity", True)

    # extract quality_percentile from manifest if available; all per-dataset manifests
    # must agree on this value — the eval pipeline only carries a single value downstream.
    quality_percentile = 1.0
    if split_manifest:
        percentiles = {m["quality_percentile"] for m in split_manifest.get("manifests", []) if "quality_percentile" in m}
        if len(percentiles) > 1:
            raise ValueError(f"inconsistent quality_percentile across manifests: {percentiles}")
        if percentiles:
            quality_percentile = percentiles.pop()

    eval_args = argparse.Namespace(
        corpus=corpus,
        poses=poses,
        target_fps=None,
        quality_percentile=quality_percentile,
    )

    ensure_datasets_registered()
    dataset_names = sorted(DATASET_REGISTRY.keys()) if datasets == "all" else [d.strip() for d in datasets.split(",")]
    splits = ["dev", "test"]
    results = {}

    # per-dataset evaluation
    for ds_name in dataset_names:
        results[ds_name] = {}
        for s in splits:
            print(f"  evaluating {ds_name} {s}...")
            results[ds_name][s] = _eval_single(
                model=model,
                datasets=ds_name,
                split=s,
                eval_args=eval_args,
                fps_aug=fps_aug,
                velocity=velocity,
                device=device,
            )

    # combined evaluation (only if multiple datasets)
    if len(dataset_names) > 1:
        results["combined"] = {}
        for s in splits:
            print(f"  evaluating combined {s}...")
            results["combined"][s] = _eval_single(
                model=model,
                datasets=datasets,
                split=s,
                eval_args=eval_args,
                fps_aug=fps_aug,
                velocity=velocity,
                device=device,
            )

    return results


def check_regression(new_metrics: dict, repo_id: str, threshold: float) -> tuple[str, dict | None]:
    """Compare new metrics against the latest tagged model.

    Returns (status, baseline_metrics) where status is "pass", "fail", or "no_baseline".
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    latest_tag = get_latest_version(repo_id=repo_id)
    if latest_tag is None:
        print("No version tags found — skipping regression check")
        return "no_baseline", None
    try:
        print(f"Comparing against latest tag: {latest_tag}")
        baseline_path = api.hf_hub_download(
            repo_id=repo_id,
            filename="eval_results.json",
            revision=latest_tag,
        )
        with open(baseline_path) as f:
            prod_metrics = json.load(f)
    except HfHubHTTPError as e:
        # only a missing baseline file (404) is "no baseline"; auth/network errors must surface.
        if e.response.status_code == 404:
            print(f"No eval_results.json at {latest_tag} — skipping regression check")
            return "no_baseline", None
        raise

    # extract flat test metrics for comparison
    new_test = _get_test_metrics(new_metrics)
    old_test = _get_test_metrics(prod_metrics)

    # check key metrics for regression
    regressions = []
    for key in ("sign_IoU", "sentence_IoU", "hm_IoU"):
        old_val = old_test.get(key, 0.0)
        new_val = new_test.get(key, 0.0)
        if new_val < old_val - threshold:
            regressions.append(f"  {key}: {old_val:.4f} -> {new_val:.4f} (delta: {new_val - old_val:+.4f})")

    if regressions:
        print("REGRESSION DETECTED:")
        for r in regressions:
            print(r)
        return "fail", prod_metrics

    print("Regression check passed")
    return "pass", prod_metrics


def promote(repo_id: str, tag: str, revision: str) -> None:
    """Tag a revision to mark it as promoted."""
    from huggingface_hub import HfApi

    api = HfApi()

    # resolve the revision to a commit sha
    refs = api.list_repo_refs(repo_id=repo_id)
    target_sha = None
    for ref_tag in refs.tags:
        if ref_tag.name == revision:
            target_sha = ref_tag.target_commit
            break
    if target_sha is None:
        for branch in refs.branches:
            if branch.name == revision:
                target_sha = branch.target_commit
                break
    if target_sha is None:
        raise ValueError(f"Could not resolve revision {revision!r} to a commit on {repo_id}")

    api.create_tag(repo_id=repo_id, tag=tag, revision=target_sha)
    print(f"Promoted: tagged '{tag}' at {target_sha[:8]}")
