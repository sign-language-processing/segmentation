# 2026 Models

Improved sign language segmentation models trained on DGS Corpus 3.0.0-uzh-document.

## Files

- `best.ckpt` — best checkpoint (PyTorch Lightning `.ckpt`)

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
    --hidden_dim 384 --encoder_depth 4 --attn_nhead 8 \
    --batch_size 8 --num_frames 1024 \
    --dice_loss_weight 1.5 \
    --epochs 500 --patience 100
```

## Architecture

**CNN-medium-attn with RoPE**

- Stage 1: Two-stage UNet CNN — spatial compression over joints (skip connections at each resolution), then temporal context
- Stage 2: N-layer pre-norm transformer with Rotary Position Embedding (RoPE) and residual connections
- Two output heads: sign (gloss) BIO and phrase (sentence) BIO, each a two-layer MLP (linear → GELU → linear)
- ~5.5M parameters (hidden_dim=384, depth=4, nhead=8)

RoPE encodes relative time — attention scores depend on the *time difference* in
seconds between frames, not absolute position. This allows chunked inference
(training-window-sized chunks) that correctly handles the full-length dev videos
(3k–51k frames), because each chunk sees the same relative-time distribution as
during training.

## What Helped

| Technique | Effect |
|-----------|--------|
| **CNN-medium-attn + RoPE** (vs LSTM, TCN, local-attn) | +6pp Sign IoU over LSTM |
| **Dice loss** (weight=1.5) | +2pp Sign IoU; 3× faster convergence |
| **fps_aug** (random 25–50fps per clip) | ESSENTIAL — disabling drops Sign IoU from 0.58→0.49 |
| **body_part_dropout=0.1** | +0.9pp S25, +10.5pp Phrase25 (zeroes each hand independently) |
| **frame_dropout=0.15** | ESSENTIAL regularization — without it, phrase head overfits at long training |
| **velocity features** | +1–2pp Sign IoU |
| **no_face** (exclude face landmarks) | Cleaner signal; improves convergence |
| **hidden_dim=384** (vs 256) | +2pp Sign IoU |
| **encoder_depth=4** | More training time with depth=4 beats depth=6 at equal budget |
| **Inference chunk_size=num_frames** | +12.8pp Phrase IoU at 2048-frame models (bug fix) |

## What Did Not Help

| Technique | Why |
|-----------|-----|
| **Attention padding mask** | Consistently hurt Sign25 by ~7pp — training uses padded sequences but inference does not, so mask changes training distribution in a way that does not match inference |
| **Threshold-based decoding** | Likeliest (argmax) wins on well-trained models; threshold overfits dev sign IoU at the expense of phrase IoU |
| **B-frame Dice loss** | Added complexity, no consistent IoU improvement |
| **Focal loss** | No benefit over plain NLL + Dice in our setup |
| **Label smoothing** | Marginal/no effect |
| **Per-head weighted NLL** | Dice loss already rewards B/I frames; extra weighting redundant |
| **Speed augmentation** | Hurts Sign IoU; model over-rotates sign positions |
| **Acceleration features** | No consistent improvement over velocity alone |
| **Frame curriculum** (grow num_frames during training) | Marginal at best; adds complexity |
| **fd=0 (no frame_dropout)** | Severe phrase head overfitting after ~50 epochs |
| **2048-frame context** | Marginal sign gain, worse phrase without long training; 1024fr more efficient |
| **Longer fine-tuning** (beyond early stopping) | Hurts IoU; val_loss is a flawed proxy — trust early stopping |

## Best Results (dev split, 50fps evaluation)

| Model | Sign IoU@50 | Phrase IoU@50 | HM |
|-------|-------------|---------------|----|
| E1s (2023 paper) | 0.440 | — | — |
| E166 (depth=4, 1024fr, 3h) | 0.641 | 0.900 | 0.748 |
| E167 (depth=4, 1536fr, 3h) | 0.645 | 0.908 | 0.754 |
| E169 (depth=4, 1024fr, 6h) | 0.657 | 0.910 | 0.763 |
| **Efinal** (depth=4, 1024fr, 12h) | — | — | — |

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
