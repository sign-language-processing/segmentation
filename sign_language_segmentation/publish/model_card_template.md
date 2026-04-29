---
tags:
  - sign-language
  - segmentation
  - pose
  - pytorch
  - pytorch-lightning
library_name: pytorch
pipeline_tag: other
{{model_index}}
---

# Sign Language Segmentation

CNN-medium-attn model with RoPE for sign language segmentation.
Jointly trained on sign (gloss) and phrase (sentence) BIO tagging.

**Published:** {{published_at}}
**Tag:** `{{tag}}`
**Regression status:** {{regression_status}}

## Architecture

{{architecture_rows}}

{{eval_section}}

## Training Config

{{training_rows}}

{{dataset_section}}

## Usage

```bash
pip install sign-language-segmentation
export HF_MODEL_REPO={{repo_id}} HF_MODEL_REVISION={{tag}}
pose_to_segments --pose input.pose --elan output.eaf
```
