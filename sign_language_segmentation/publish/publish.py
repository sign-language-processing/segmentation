"""Publish a model checkpoint to HuggingFace Hub.

Converts a PyTorch Lightning .ckpt to safetensors, optionally evaluates it,
runs regression checks against the current production model, generates a
model card, and pushes to the 'weekly' branch on HuggingFace Hub.
A date-based tag (vYYYY.MM.DD) is only created on promotion
(regression pass or explicit --promote).

Usage:
    publish_model --checkpoint path/to/best.ckpt --repo org/model-name
    publish_model --repo org/model-name --promote
"""
import argparse
import json
import os
import tempfile
from datetime import datetime, UTC
from pathlib import Path



from sign_language_segmentation.publish.utils import (
    convert_to_safetensors,
    find_split_manifest,
    run_evaluation,
    check_regression,
    generate_model_card,
    promote,
    get_next_version,
)


def publish(
    checkpoint: str,
    repo_id: str,
    tag: str,
    datasets: str,
    corpus: str,
    poses: str,
    device: str,
    skip_eval: bool,
    metrics_json: str | None,
    regression_threshold: float,
    no_promote: bool
) -> None:
    """Main publish workflow."""
    from huggingface_hub import HfApi

    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. convert to safetensors
        print(f"Converting {checkpoint} to safetensors...")
        config = convert_to_safetensors(checkpoint_path=checkpoint, output_dir=tmp_path)
        print(f"  model.safetensors + config.json written to {tmp_path}")

        # 2. load split manifest (needed for eval quality_percentile)
        manifest = find_split_manifest(checkpoint_path=checkpoint)

        # 3. evaluation
        eval_results = None
        if not skip_eval:
            if metrics_json:
                print(f"Loading pre-computed metrics from {metrics_json}")
                with open(metrics_json) as f:
                    eval_results = json.load(f)
            else:
                print(f"Evaluating on {datasets} dev+test sets...")
                eval_results = run_evaluation(
                    checkpoint_path=checkpoint, datasets=datasets,
                    corpus=corpus, poses=poses,
                    device=device, split_manifest=manifest,
                )

        # save eval results
        if eval_results:
            with open(tmp_path / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)

        # 4. regression check
        regression_status = "skipped"
        if eval_results and not skip_eval:
            regression_status, _ = check_regression(
                new_metrics=eval_results, repo_id=repo_id, threshold=regression_threshold,
            )
        # 5. save split manifest
        if manifest:
            with open(tmp_path / "split_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        # 6. generate model card
        model_card = generate_model_card(
            config=config, eval_results=eval_results,
            regression_status=regression_status, tag=tag,
            split_manifest=manifest,
        )
        with open(tmp_path / "README.md", "w") as f:
            f.write(model_card)

        # 7. push to weekly branch
        print(f"Pushing to {repo_id} branch 'weekly'...")
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.create_branch(repo_id=repo_id, branch="weekly", exist_ok=True)
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            revision="weekly",
            commit_message=f"publish {tag} (regression: {regression_status})",
        )

        # 8. promote if regression passed — creates a date tag on the weekly branch
        if regression_status == "fail":
            print("NOT promoting — regression check failed")
        elif no_promote:
            print("Skipping promotion (--no-promote)")
        else:
            promote(repo_id=repo_id, tag=tag, revision="weekly")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Publish a model checkpoint to HuggingFace Hub",
    )
    parser.add_argument("--checkpoint", type=str,
                        help="path to .ckpt checkpoint to publish")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID (e.g. org/model-name)")
    parser.add_argument("--tag", type=str, default=None,
                        help="version tag (default: vYYYY.MM.DD based on today's date)")

    # evaluation
    parser.add_argument("--datasets", type=str, default="dgs",
                        help="comma-separated dataset names for evaluation")
    parser.add_argument("--corpus", type=str,
                        default="/mnt/nas/GCS/sign-external-datasets/dgs-corpus")
    parser.add_argument("--poses", type=str,
                        default="/mnt/nas/GCS/sign-mediapipe-holistic-poses")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip evaluation and regression check")
    parser.add_argument("--metrics-json", type=str,
                        help="path to pre-computed metrics JSON (alternative to running eval)")

    # regression / promotion
    parser.add_argument("--regression-threshold", type=float, default=0.02,
                        help="IoU drop tolerance for regression check (default: 0.02)")
    parser.add_argument("--no-promote", action="store_true",
                        help="push without tagging or promoting")
    parser.add_argument("--promote", action="store_true",
                        help="tag the current weekly branch (no upload)")

    args = parser.parse_args()

    # resolve version tag
    if args.tag is None:
        args.tag = get_next_version(repo_id=args.repo)
    print(f"Version: {args.tag}")

    # standalone promote mode
    if args.promote:
        promote(repo_id=args.repo, tag=args.tag, revision="weekly")
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless using --promote)")

    publish(
        checkpoint=args.checkpoint,
        repo_id=args.repo,
        tag=args.tag,
        datasets=args.datasets,
        corpus=args.corpus,
        poses=args.poses,
        device=args.device,
        skip_eval=args.skip_eval,
        metrics_json=args.metrics_json,
        regression_threshold=args.regression_threshold,
        no_promote=args.no_promote,
    )


if __name__ == "__main__":
    main()
