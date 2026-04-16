"""Utility functions for model publishing: conversion, evaluation, regression, model card."""
import json
import re
from datetime import datetime, UTC
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors

_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "model_card_template.md"

_ARCH_KEYS = ["hidden_dim", "encoder_depth", "attn_nhead", "attn_ff_mult",
              "attn_dropout", "num_frames", "pose_dims", "num_classes"]
_TRAIN_KEYS = ["learning_rate", "optimizer", "dice_loss_weight",
               "fps_aug", "frame_dropout"]


def convert_to_safetensors(checkpoint_path: str, output_dir: Path) -> dict:
    """Convert .ckpt to safetensors + config.json. Returns hyper_parameters dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sd_bf16 = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in ckpt["state_dict"].items()
    }
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


def _parse_version(tag_name: str) -> tuple[int, int, int] | None:
    """Parse a vMAJOR.MINOR.PATCH tag. Returns None if not a version tag."""
    m = _VERSION_RE.match(tag_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_latest_version(repo_id: str) -> str | None:
    """Get the latest semver tag from the HF repo. Returns None if no version tags exist."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        refs = api.list_repo_refs(repo_id=repo_id)
    except Exception:
        return None
    versions = []
    for tag in refs.tags:
        parsed = _parse_version(tag.name)
        if parsed:
            versions.append((parsed, tag.name))
    if not versions:
        return None
    versions.sort()
    return versions[-1][1]


def get_next_version(repo_id: str, bump: str = "patch") -> str:
    """Determine the next semver tag by bumping the latest version.

    bump: "major", "minor", or "patch".
    """
    latest = get_latest_version(repo_id=repo_id)
    if latest is None:
        return "v1.0.0"
    major, minor, patch = _parse_version(latest)
    if bump == "major":
        return f"v{major + 1}.0.0"
    if bump == "minor":
        return f"v{major}.{minor + 1}.0"
    return f"v{major}.{minor}.{patch + 1}"


def run_evaluation(checkpoint_path: str, datasets: str, split: str,
                   corpus: str, poses: str, annotations_path: str,
                   device: str, split_manifest: dict | None = None) -> dict:
    """Run model evaluation and return metrics dict."""
    from sign_language_segmentation.evaluate import evaluate_model
    from sign_language_segmentation.datasets.common import Split, build_datasets, collate_fn
    from sign_language_segmentation.model.model import PoseTaggingModel
    from torch.utils.data import DataLoader

    model = PoseTaggingModel.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device, strict=False)
    model = model.to(device)

    fps_aug = getattr(model.hparams, "fps_aug", False)
    velocity = getattr(model.hparams, "velocity", True)

    # extract quality_percentile from manifest if available
    quality_percentile = 1.0
    if split_manifest:
        for m in split_manifest.get("manifests", []):
            qp = m.get("quality_percentile")
            if qp is not None:
                quality_percentile = qp
                break

    # build a namespace matching evaluate.py's expected args
    class EvalArgs:
        pass
    eval_args = EvalArgs()
    eval_args.datasets = datasets
    eval_args.corpus = corpus
    eval_args.poses = poses
    eval_args.annotations_path = annotations_path
    eval_args.target_fps = None
    eval_args.quality_percentile = quality_percentile

    dataset = build_datasets(
        names=datasets,
        split=Split(split),
        args=eval_args,
        num_frames=999999,
        fps_aug=fps_aug,
        velocity=velocity,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)
    results = evaluate_model(model=model, dataloader=dataloader, device=device)

    # compute harmonic mean of sign and sentence IoU
    sign_iou = results.get("sign_IoU", 0)
    sentence_iou = results.get("sentence_IoU", 0)
    if sign_iou > 0 and sentence_iou > 0:
        results["hm_IoU"] = 2 * sign_iou * sentence_iou / (sign_iou + sentence_iou)

    return results


def check_regression(new_metrics: dict, repo_id: str,
                     threshold: float) -> tuple[str, dict | None]:
    """Compare new metrics against the latest tagged model.

    Returns (status, baseline_metrics) where status is "pass", "fail", or "no_baseline".
    """
    from huggingface_hub import HfApi

    api = HfApi()
    latest_tag = get_latest_version(repo_id=repo_id)
    if latest_tag is None:
        print("No version tags found — skipping regression check")
        return "no_baseline", None
    try:
        print(f"Comparing against latest tag: {latest_tag}")
        baseline_path = api.hf_hub_download(
            repo_id=repo_id, filename="eval_results.json", revision=latest_tag,
        )
        with open(baseline_path) as f:
            prod_metrics = json.load(f)
    except Exception:
        print(f"Could not download eval_results.json from {latest_tag} — skipping regression check")
        return "no_baseline", None

    # check key metrics for regression
    regressions = []
    for key in ("sign_IoU", "sentence_IoU"):
        old_val = prod_metrics.get(key, 0.0)
        new_val = new_metrics.get(key, 0.0)
        if new_val < old_val - threshold:
            regressions.append(f"  {key}: {old_val:.4f} -> {new_val:.4f} (delta: {new_val - old_val:+.4f})")

    # TODO: add slack notifications
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
        target_sha = revision

    api.create_tag(repo_id=repo_id, tag=tag, revision=target_sha)
    print(f"Promoted: tagged '{tag}' at {target_sha[:8]}")


def _build_table_rows(config: dict, keys: list[str]) -> str:
    return "\n".join(f"| {k} | {config[k]} |" for k in keys if k in config)


def _build_model_index(eval_results: dict) -> str:
    lines = ["model-index:", "  - name: sign-language-segmentation", "    results:",
             "      - task:", "          type: other",
             "          name: Sign Language Segmentation", "        metrics:"]
    for key, value in eval_results.items():
        display_name = key.replace("_", " ").title()
        lines.append(f"          - name: {display_name}")
        lines.append(f"            type: {key}")
        lines.append(f"            value: {value:.4f}")
    return "\n".join(lines)


def _build_eval_section(eval_results: dict) -> str:
    rows = "\n".join(
        f"| {k.replace('_', ' ').title()} | {v:.4f} |"
        for k, v in eval_results.items()
    )
    return f"## Evaluation Results\n\n| Metric | Value |\n|--------|-------|\n{rows}"


def _build_dataset_section(split_manifest: dict) -> str:
    datasets = split_manifest.get("datasets", "unknown")
    created_at = split_manifest.get("created_at", "unknown")
    return f"## Dataset\n\n- **Datasets:** {datasets}\n- **Created at:** {created_at}"


def generate_model_card(config: dict, eval_results: dict | None,
                        regression_status: str, tag: str,
                        split_manifest: dict | None) -> str:
    """Generate a HuggingFace model card from template."""
    template = _TEMPLATE_PATH.read_text()

    replacements = {
        "{{published_at}}": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "{{tag}}": tag,
        "{{regression_status}}": regression_status,
        "{{architecture_rows}}": _build_table_rows(config=config, keys=_ARCH_KEYS),
        "{{training_rows}}": _build_table_rows(config=config, keys=_TRAIN_KEYS),
        "{{model_index}}": _build_model_index(eval_results=eval_results) if eval_results else "",
        "{{eval_section}}": _build_eval_section(eval_results=eval_results) if eval_results else "",
        "{{dataset_section}}": _build_dataset_section(split_manifest=split_manifest) if split_manifest else "",
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    return template
