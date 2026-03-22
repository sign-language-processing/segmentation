# 2023 Models

Models from the EMNLP 2023 paper: "Automatic Segmentation of Sign Language into Signs".

## Files

- `model_E1s-1.pth` — BiLSTM 4-layer, 25fps, no optical flow (paper E1s config)
- `model_E4s-1.pth` — BiLSTM 1-layer, optical flow + 3D hand normalisation (paper E4s config)
- `model.pth` — default model (same as E1s)

These are TorchScript (`.pth`) files loaded via `torch.jit.load`.

## Usage

```bash
python -m sign_language_segmentation.old.bin \
    --pose input.pose \
    --elan output.eaf \
    --model model_E1s-1.pth
```

## Performance (re-evaluated on DGS Corpus 3.0.0-uzh-document dev split)

| Model | Sign IoU | Sign Seg F1 |
|-------|----------|-------------|
| E1s   | 0.4398   | 0.9265      |
| E4s   | 0.5129   | 0.9690      |

Note: both models use IO-fallback decoding (B-class probability is unreliable
at inference time on our pose format). Predicted count is ~1.5–2× gold count.
