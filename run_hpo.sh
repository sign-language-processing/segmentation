#!/usr/bin/env bash
set -e

# sync annotations from production Convex DB (skips if cache exists)
uv run --python 3.11 --extra train \
  python -m sign_language_segmentation.datasets.annotation_platform.sync \
  --project_ids \
    ms76x02ycvvvp44k9796hs92m98192r5 \
    ms7a40x27xemrk45bxkf6tx9yh81r822 \
    ms77arjjhgfdc7tyqk197xeb6x81s3mm \
    ms775tp9gsqfpgf7gsetb0376181rerq

# run HPO sweep
uv run --python 3.11 --extra dev --extra train \
  python -m sign_language_segmentation.train \
  --dataset platform \
  --device gpu \
  --gpus 1 \
  --quality_percentile 0.8 \
  --finetune_from sign_language_segmentation/dist/2026/best.ckpt \
  --optuna sign_language_segmentation/optuna.yaml \
  --optuna_trials 50 \
  --run_name lr_scale_backbone_optuna
