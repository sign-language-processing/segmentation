"""Publish a model checkpoint to HuggingFace Hub.

Converts a PyTorch Lightning .ckpt to safetensors, optionally evaluates it,
runs regression checks against the current production model, generates a
model card, and pushes everything to HuggingFace Hub with versioning tags.

Usage:
    publish_model --checkpoint path/to/best.ckpt --repo org/model-name
    publish_model --checkpoint path/to/best.ckpt --repo org/model-name --skip-eval
    publish_model --repo org/model-name --promote-tag weekly-2026-04-08
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
)


def publish(checkpoint: str, repo_id: str, tag: str,
            datasets: str, split: str, corpus: str, poses: str,
            annotations_path: str, device: str,
            skip_eval: bool, metrics_json: str | None,
            regression_threshold: float, no_promote: bool) -> None:
    """Main publish workflow."""
    from huggingface_hub import HfApi

    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. convert to safetensors
        print(f"Converting {checkpoint} to safetensors...")
        config = convert_to_safetensors(checkpoint_path=checkpoint, output_dir=tmp_path)
        print(f"  model.safetensors + config.json written to {tmp_path}")

        # 2. evaluation
        eval_results = None
        if not skip_eval:
            if metrics_json:
                print(f"Loading pre-computed metrics from {metrics_json}")
                with open(metrics_json) as f:
                    eval_results = json.load(f)
            else:
                print(f"Evaluating on {datasets} {split} set...")
                eval_results = run_evaluation(
                    checkpoint_path=checkpoint, datasets=datasets, split=split,
                    corpus=corpus, poses=poses, annotations_path=annotations_path,
                    device=device,
                )
                for key, value in eval_results.items():
                    print(f"  {key}: {value:.4f}")

        # save eval results
        if eval_results:
            with open(tmp_path / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)

        # 3. regression check
        regression_status = "skipped"
        if eval_results and not skip_eval:
            regression_status, _ = check_regression(
                new_metrics=eval_results, repo_id=repo_id, threshold=regression_threshold,
            )

        # 4. copy split manifest if available
        manifest = find_split_manifest(checkpoint_path=checkpoint)
        if manifest:
            with open(tmp_path / "split_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        # 5. generate model card
        model_card = generate_model_card(
            config=config, eval_results=eval_results,
            regression_status=regression_status, tag=tag,
            split_manifest=manifest,
        )
        with open(tmp_path / "README.md", "w") as f:
            f.write(model_card)

        # 6. push to hub
        print(f"Pushing to {repo_id}...")
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            commit_message=f"publish {tag} (regression: {regression_status})",
        )

        # 7. tag the commit
        api.create_tag(repo_id=repo_id, tag=tag, revision="main")
        print(f"Tagged as '{tag}'")

        # 8. promote to production if regression passed
        if regression_status == "fail":
            print("NOT promoting to production — regression check failed")
        elif no_promote:
            print("Skipping promotion (--no-promote)")
        else:
            promote(repo_id=repo_id, revision="main")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Publish a model checkpoint to HuggingFace Hub",
    )
    parser.add_argument("--checkpoint", type=str,
                        help="path to .ckpt checkpoint to publish")
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace repo ID (e.g. org/model-name)")
    parser.add_argument("--tag", type=str,
                        default=f"weekly-{datetime.now(tz=UTC).strftime('%Y-%m-%d')}",
                        help="version tag (default: weekly-YYYY-MM-DD)")

    # evaluation
    parser.add_argument("--datasets", type=str, default="dgs",
                        help="comma-separated dataset names for evaluation")
    parser.add_argument("--split", type=str, default="test",
                        help="evaluation split (default: test)")
    parser.add_argument("--corpus", type=str,
                        default="/mnt/nas/GCS/sign-external-datasets/dgs-corpus")
    parser.add_argument("--poses", type=str,
                        default="/mnt/nas/GCS/sign-mediapipe-holistic-poses")
    parser.add_argument("--annotations_path", type=str,
                        default="sign_language_segmentation/datasets/annotation_platform/annotations_cache.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip evaluation and regression check")
    parser.add_argument("--metrics-json", type=str,
                        help="path to pre-computed metrics JSON (alternative to running eval)")

    # regression / promotion
    parser.add_argument("--regression-threshold", type=float, default=0.02,
                        help="IoU drop tolerance for regression check (default: 0.02)")
    parser.add_argument("--no-promote", action="store_true",
                        help="push and tag but do not move the production tag")
    parser.add_argument("--promote-tag", type=str,
                        help="promote an existing tag to production (no upload, just moves the tag)")

    args = parser.parse_args()

    # standalone promote mode
    if args.promote_tag:
        promote(repo_id=args.repo, revision=args.promote_tag)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless using --promote-tag)")

    publish(
        checkpoint=args.checkpoint,
        repo_id=args.repo,
        tag=args.tag,
        datasets=args.datasets,
        split=args.split,
        corpus=args.corpus,
        poses=args.poses,
        annotations_path=args.annotations_path,
        device=args.device,
        skip_eval=args.skip_eval,
        metrics_json=args.metrics_json,
        regression_threshold=args.regression_threshold,
        no_promote=args.no_promote,
    )


if __name__ == "__main__":
    main()
