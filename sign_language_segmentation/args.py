import random
from argparse import ArgumentParser

import numpy as np
import torch

parser = ArgumentParser()

# wandb
parser.add_argument('--no_wandb', action='store_true', default=True, help='ignore wandb?')
parser.add_argument('--run_name', type=str, default=None, help='name of wandb run')
parser.add_argument('--wandb_dir', type=str, default='.', help='where to store wandb data')

# Training Arguments
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=str, default='gpu', help='device to use, cpu or gpu')
parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
parser.add_argument('--epochs', type=int, default=100, help='how many epochs')
parser.add_argument('--patience', type=int, default=20, help='how many epochs as the patience for early stopping')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_frames_per_item', type=int, default=2 ** 10, help='batch size')
parser.add_argument('--batch_size_devtest', type=int, default=20,
                    help='batch size for dev and test (by default run all in one batch)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--steps_per_epoch', type=int, default=100, help='steps per epoch')

# Data Arguments
parser.add_argument('--dataset', default='/tmp/segmentation', help='which dataset to use?')

# Model Arguments
parser.add_argument('--hidden_dim', type=int, default=256, help='encoder hidden dimension')

parser.add_argument('--save_jit', action="store_true", default=False, help='whether to save model without code?')

# Prediction args
parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--pred_output', type=str, default=None, metavar='PATH', help="Path for saving prediction files")

args = parser.parse_args()

print('Agruments:', args)

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

