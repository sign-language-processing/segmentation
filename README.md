# Sign Language Segmentation

Pose segmentation model for sign language — signs and sentences — using CNN + Transformer with RoPE.

## Usage

```bash
# Install
pip install git+https://github.com/sign-language-processing/segmentation

# Acquire a MediaPipe Holistic pose file
wget -O example.pose https://sign-lanugage-datasets.sign-mt.cloud/poses/holistic/dgs_corpus/1413451-11105600-11163240_a.pose

# Run the model
pose_to_segments --pose example.pose --elan output.eaf [--video example.mp4]
```

The model reads a `.pose` file and writes an ELAN (`.eaf`) annotation file with SIGN and SENTENCE tiers.

```python
from pose_format import Pose
from sign_language_segmentation.bin import segment_pose

with open("example.pose", "rb") as f:
    pose = Pose.read(f)

eaf, tiers = segment_pose(pose)
# tiers["SIGN"] and tiers["SENTENCE"] are lists of {"start": int, "end": int} frame dicts
```

## Server

```bash
# Build and run the inference server
docker build -t segmentation-serve .
docker run -p 8080:8080 -e PORT=8080 segmentation-serve

# Segment a pose file (input/output are file paths or gs:// URIs)
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"input": "/path/to/input.pose", "output": "/path/to/output.eaf"}'

# Health check
curl http://localhost:8080/health
```

## Training

### Prerequisites

Requires the DGS Corpus and MediaPipe Holistic poses (internal datasets).

### Docker (recommended)

```bash
# Build the training image
docker build -f Dockerfile.train -t segmentation-train .

# Train
docker run --rm --gpus all \
  -v /path/to/dgs-corpus:/data/dgs-corpus:ro \
  -v /path/to/mediapipe-poses:/data/poses:ro \
  -v $(pwd)/models:/app/models \
  segmentation-train \
  python -m sign_language_segmentation.train \
    --corpus /data/dgs-corpus \
    --poses /data/poses \
    --hidden_dim 384 --encoder_depth 4 --attn_nhead 8 \
    --batch_size 8 --num_frames 1024 \
    --dice_loss_weight 1.5 \
    --epochs 500 --patience 100

# Evaluate on dev split
docker run --rm --gpus all \
  -v /path/to/dgs-corpus:/data/dgs-corpus:ro \
  -v /path/to/mediapipe-poses:/data/poses:ro \
  -v $(pwd)/models:/app/models \
  segmentation-train \
  python -m sign_language_segmentation.evaluate \
    --checkpoint /app/models/<run_name>/best.ckpt \
    --corpus /data/dgs-corpus \
    --poses /data/poses \
    --split dev
```

Best hyperparameters and architecture details: [`dist/2026/README.md`](dist/2026/README.md).

### Local (development)

```bash
conda create --name segmentation python=3.12 -y
conda activate segmentation
pip install ".[dev]"
python -m sign_language_segmentation.train --corpus /path/to/dgs-corpus --poses /path/to/poses
```

## Architecture

CNN-medium-attn + RoPE (2026):
- Stage 1: Two-stage UNet CNN — spatial compression over joints, then temporal context
- Stage 2: N-layer pre-norm Transformer with Rotary Position Embedding (RoPE)
- Two output heads: sign (gloss) BIO and phrase (sentence) BIO

See [`dist/2026/README.md`](dist/2026/README.md) for what worked, what didn't, and key bug fixes.

## Citation

```bibtex
@inproceedings{moryossef-etal-2023-linguistically,
    title = "Linguistically Motivated Sign Language Segmentation",
    author = {Moryossef, Amit  and Jiang, Zifan  and M{\"u}ller, Mathias  and Ebling, Sarah  and Goldberg, Yoav},
    booktitle = "Findings of EMNLP 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.findings-emnlp.846",
}
```
