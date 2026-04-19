"""Strip optimizer states and convert weights to bfloat16 for deployment.

Usage:
    uv run python -m sign_language_segmentation.slim_checkpoint \
        path/to/input.ckpt [path/to/output.ckpt]

If no output path is given, overwrites the input file in-place.
Reduces checkpoint size ~6x (e.g. 66MB → 11MB) by:
  1. Removing AdamW optimizer states (m/v tensors)
  2. Converting float32 model weights to bfloat16
"""

import sys

import torch


def slim(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    sd_bf16 = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
    slim_ckpt = {
        "state_dict": sd_bf16,
        "hyper_parameters": ckpt["hyper_parameters"],
        "hparams_name": ckpt["hparams_name"],
        "pytorch-lightning_version": ckpt["pytorch-lightning_version"],
    }
    torch.save(slim_ckpt, dst)

    import os

    before_mb = os.path.getsize(src) / 1e6 if src != dst else None
    after_mb = os.path.getsize(dst) / 1e6
    if before_mb:
        print(f"{src} → {dst}  ({before_mb:.1f} MB → {after_mb:.1f} MB)")
    else:
        print(f"Saved {dst}  ({after_mb:.1f} MB)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    slim(src, dst)


if __name__ == "__main__":
    main()
