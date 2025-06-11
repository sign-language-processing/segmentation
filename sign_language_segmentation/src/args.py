import random
from argparse import ArgumentParser
from os import path

import numpy as np
import torch


def boolean_string(s):
    lower_s = s.lower()
    if lower_s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

# wandb
parser.add_argument('--no_wandb', type=boolean_string, default=False, help='ignore wandb?')
parser.add_argument('--run_name', type=str, default=None, help='name of wandb run')
parser.add_argument('--wandb_dir', type=str, default='.', help='where to store wandb data')

# Training Arguments
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--device', type=str, default='gpu', help='device to use, cpu or gpu')
parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
parser.add_argument('--epochs', type=int, default=100, help='how many epochs')
parser.add_argument('--patience', type=int, default=20, help='how many epochs as the patience for early stopping')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--batch_size_devtest', type=int, default=20,
                    help='batch size for dev and test (by default run all in one batch)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--lr_scheduler', type=str, default='none', help='optimizer learning rate scheduler')

# Data Arguments
parser.add_argument('--dataset', choices=['dgs_corpus', 'mediapi_skel', 'bobsl_cslr', 'bslcp'],
                    default='dgs_corpus', help='which dataset to use?')
parser.add_argument('--data_dir', help='which dir to store the dataset?')
parser.add_argument('--data_dev', type=boolean_string, default=False,
                    help='whether to use dev set as training data for fast debugging?')
parser.add_argument('--fps', type=int, default=25, help='fps to load')
parser.add_argument('--pose', choices=['holistic', 'openpose'], default='holistic', help='which pose estimation')
parser.add_argument(
    '--pose_components',
    nargs='+',
    default=["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
    help='what pose components to use?')
parser.add_argument('--pose_reduce_face', type=bool, default=False, help='Should we reduce the face keypoints?')
parser.add_argument('--hand_normalization', type=boolean_string, default=False,
                    help='Should we perform 3D normalization on hands?')
parser.add_argument('--optical_flow', type=boolean_string, default=False, help='Should we use optical flow?')
parser.add_argument('--only_optical_flow', type=boolean_string, default=False, help='Should we use only optical flow?')
parser.add_argument('--classes', choices=['bio', 'io'], default="bio", help='Should we use BIO tagging or IO tagging?')

# Model Arguments
parser.add_argument('--pose_projection_dim', type=int, default=256, help='pose projection dimension')
parser.add_argument('--hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--encoder_depth', type=int, default=4, help='number of layers for the encoder')
parser.add_argument('--encoder_bidirectional', type=boolean_string, default=True,
                    help='should use a bidirectional encoder?')
parser.add_argument('--encoder_autoregressive', type=boolean_string, default=False,
                    help='should use a autoregressive encoder?')
parser.add_argument('--weighted_loss', type=boolean_string, default=True, help='should use a class weighted loss?')

# Decoding Algorithm
parser.add_argument('--b_threshold', type=int, default=50, help='b_threshold')
parser.add_argument('--o_threshold', type=int, default=50, help='o_threshold')
parser.add_argument('--threshold_likeliest', type=boolean_string, default=False,
                    help='should use the likeliest class for decoding?')

# Testing Arguments
parser.add_argument('--train', type=boolean_string, default=True, help='whether to train')
parser.add_argument('--test', type=boolean_string, default=True, help='whether to test after training finishes?')
parser.add_argument('--save_jit', type=boolean_string, default=False, help='whether to save model without code?')

# Prediction args
parser.add_argument('--checkpoint', type=str, default=None, metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--pred_output', type=str, default=None, metavar='PATH', help="Path for saving prediction files")
parser.add_argument('--ffmpeg_path', type=str, default=None, metavar='PATH', help="Path for ffmpeg executable")

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

# conda update -n base -c defaults conda
