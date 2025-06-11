#!/bin/bash

#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=32                   # Number of CPU cores per task
#SBATCH --mem=256gb                          # Job memory request
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=compute                     # Partition (compute (default) / gpu)
# -------------------------------

srun videos_to_poses --format mediapipe --num-workers 32 --directory /users/zifan/BSL-Corpus/derivatives/videos --additional-config "model_complexity=2,refine_face_landmarks=True"