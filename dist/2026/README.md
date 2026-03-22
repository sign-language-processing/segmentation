# 2026 Models

Improved sign language segmentation models trained on DGS Corpus 3.0.0-uzh-document.
Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md).

## Files

- `best.ckpt` — best checkpoint (PyTorch Lightning `.ckpt`, to be added after final training)

## Usage

```bash
# Inference
python -m sign_language_segmentation.bin \
    --pose input.pose \
    --elan output.eaf

# Training
python -m sign_language_segmentation.train \
    --corpus /path/to/dgs-corpus \
    --poses /path/to/mediapipe-poses \
    --hidden_dim 384 --encoder_depth 6 --attn_nhead 8 \
    --velocity --no_face --fps_aug \
    --body_part_dropout 0.1 --frame_dropout 0.15 \
    --dice_loss_weight 1.0 \
    --batch_size 8 --num_frames 1024 \
    --epochs 200 --patience 10
```

## Architecture

**CNN-medium-attn with RoPE**

- Stage 1: Two-stage UNet CNN — spatial compression over joints, then temporal context
- Stage 2: N-layer pre-norm transformer with Rotary Position Embedding (RoPE)
- Two output heads: sign (gloss) BIO and phrase (sentence) BIO
- ~7.8M parameters (hidden_dim=384, depth=6, nhead=8)

RoPE encodes relative position — attention scores depend on frame *offset*, not
absolute index. This allows chunked inference (training-window-sized chunks) that
correctly handles the full-length dev videos (3k–51k frames).

## What Helped

| Technique | Effect |
|-----------|--------|
| **CNN-medium-attn + RoPE** (vs LSTM, TCN, local-attn) | +6pp Sign IoU over LSTM |
| **Dice loss** (weight=1.0) | +2pp Sign IoU; 3× faster convergence |
| **fps_aug** (random 25–50fps per clip) | ESSENTIAL — disabling drops Sign IoU from 0.58→0.49 |
| **body_part_dropout=0.1** | +0.9pp S25, +10.5pp Phrase25 (zeroes each hand independently) |
| **frame_dropout=0.15** | ESSENTIAL regularization — without it, phrase head overfits at long training |
| **velocity features** | +1–2pp Sign IoU |
| **no_face** (exclude face landmarks) | Cleaner signal; improves convergence |
| **hidden_dim=384** (vs 256) | +2pp Sign IoU |
| **encoder_depth=6** (vs 4) | Marginal improvement |
| **HM(sign, phrase) validation metric** | Prevents sign over-optimisation at expense of phrase |
| **Inference chunk_size=num_frames** | +12.8pp Phrase IoU at 2048-frame models (bug fix) |

## What Did Not Help

| Technique | Why |
|-----------|-----|
| **Attention padding mask** | Consistently hurt Sign25 by ~7pp; training dynamics unfavorable |
| **B-frame Dice loss** | Added complexity, no consistent IoU improvement |
| **Focal loss** | No benefit over plain NLL + Dice in our setup |
| **Label smoothing** | Marginal/no effect |
| **Per-head weighted NLL** | Dice loss already rewards B/I frames; extra weighting redundant |
| **Speed augmentation** | Hurts Sign IoU; model over-rotates sign positions |
| **Acceleration features** | No consistent improvement over velocity alone |
| **Frame curriculum** (grow num_frames during training) | Marginal at best; adds complexity |
| **fd=0 (no frame_dropout)** | Severe phrase head overfitting after ~50 epochs |
| **2048-frame context** | Higher Sign50 (+1pp) but worse Phrase25 without long training |
| **Longer fine-tuning** (beyond early stopping) | Hurts IoU; val_loss is a flawed proxy — trust early stopping |

## Best Results (dev split, 50fps evaluation)

| Model | Sign IoU@50 | Sign IoU@25 | Phrase IoU@50 | Phrase IoU@25 | HM |
|-------|-------------|-------------|---------------|---------------|----|
| E1s (2023 paper) | 0.440 | — | — | — | — |
| E145 (1024fr, body_dropout=0.1) | 0.595 | 0.569 | 0.907 | 0.880 | **0.705** |
| E147 (2048fr) | 0.617 | 0.573 | 0.739 | — | 0.675 |
| E160 (2048fr + bug fixes) | 0.607 | 0.563 | 0.867 | — | 0.721* |

*E160 evaluated before HM metric was adopted for early stopping.

HM = harmonic mean of Sign IoU@50 and Phrase IoU@50.

## Key Bugs Fixed

1. **Hardcoded 1024-frame inference chunks**: All dev videos are >3k frames and use
   chunked inference. Old code always chunked at 1024 frames regardless of `num_frames`.
   2048-frame trained models were split at 1024 during eval → phrases spanning chunk
   boundaries were missed → Phrase IoU collapsed (0.739 vs 0.907).
   **Fix**: `chunk_size = self.hparams.num_frames`

2. **Attention padding mask**: Old code built a causal-style padding mask from sequence
   lengths and passed it to all attention layers. This changed training dynamics in a way
   that degraded Sign IoU at 25fps by ~7pp across all experiments that used it.
   **Fix**: removed mask — layers called without `attn_mask`.
