import random
from argparse import ArgumentParser

import numpy as np
import torch

parser = ArgumentParser()

# wandb
parser.add_argument('--no_wandb', action='store_true', default=False)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--wandb_dir', type=str, default='.')
parser.add_argument('--wandb_entity', type=str, default='nagish')
parser.add_argument('--wandb_project', type=str, default='segmentation')

# Training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='gpu')
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_frames', type=int, default=1024, help='frames per training sample')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--max_time', type=str, default="00:00:30:00",
                    help='max wall time DD:HH:MM:SS (default: 30 min)')
parser.add_argument('--optimizer',
                    choices=['adam', 'adamw', 'adamw-onecycle', 'cosine', 'constant'],
                    default='adamw-onecycle')
parser.add_argument('--finetune_from', type=str, default=None,
                    help='checkpoint to fine-tune from')

# Data
parser.add_argument('--dataset', choices=['dgs', 'platform', 'combined'], default='dgs',
                    help='dataset to train on')
parser.add_argument('--corpus', default='/mnt/nas/GCS/sign-external-datasets/dgs-corpus')
parser.add_argument('--poses', default='/mnt/nas/GCS/sign-mediapipe-holistic-poses')
parser.add_argument('--annotations_path', type=str, default=None,
                    help='path to annotations_cache.json (required for platform/combined)/')
parser.add_argument('--quality_percentile', type=float, default=1.0,
                    help='keep top X of platform annotations by quality score (1.0=all, 0.8=top 80%%)')
parser.add_argument('--velocity', action='store_true', default=True,
                    help='append fps-normalised velocity to pose features')
parser.add_argument('--fps_aug', action='store_true', default=True,
                    help='randomly sample fps 25-50 per clip (fps-invariance augmentation)')
parser.add_argument('--frame_dropout', type=float, default=0.15,
                    help='max frame dropout rate (0=off, 0.15=drop 0-15%% of frames)')
parser.add_argument('--body_part_dropout', type=float, default=0.1,
                    help='per-hand zeroing probability (0=off, 0.1=10%% per hand)')

# Model
parser.add_argument('--hidden_dim', type=int, default=384)
parser.add_argument('--encoder_depth', type=int, default=4)
parser.add_argument('--attn_nhead', type=int, default=8)
parser.add_argument('--attn_ff_mult', type=int, default=2)
parser.add_argument('--attn_dropout', type=float, default=0.1)

# Loss
parser.add_argument('--dice_loss_weight', type=float, default=1.5,
                    help='Dice loss weight for binary sign mask (0=off, 1.5=recommended)')

args = parser.parse_args()

print('Arguments:', args)

if args.seed == 0:
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
