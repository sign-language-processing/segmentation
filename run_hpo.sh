#!/usr/bin/env bash
uv run --python 3.11 --extra dev \
  python -m sign_language_segmentation.train \
  --dataset platform \
  --annotations_path sign_language_segmentation/datasets/annotation_platform/annotations_cache.json \
  --device gpu \
  --gpus 1 \
  --lr_scale_backbone 0.1 \
  --score \
  --quality_percentile 0.8 \
  --finetune_from sign_language_segmentation/dist/2026/best.ckpt \
  --optuna sign_language_segmentation/optuna.yaml \
  --optuna_trials 30
  --run_name lr_scale_backbone
