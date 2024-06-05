#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH -A ccameras

#SBATCH --partition=gpu
#SBATCH --time=0:30:00  # walltime, timeout (if script runs longer than specified, it will timeout). Setting it higher results in lower priority on HPC
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --cpus-per-task=8  # Number of CPUs required per task.
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem-per-cpu=16G # memory per CPU core
#SBATCH -J "firenet_trial"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

# Needed to load miniconda
source ~/.bashrc
# module avail
module load cuda/11.8
# module load gcc/9.2.0
# module load clang/16.0.4

conda activate fire

export CUDA_VISIBLE_DEVICES=0

cd ~/FireNet

# NOTE(@kai): the training_vqgan script already uses cuda by default

python3 inference.py \
    --checkpoint_path /central/groups/ccameras/event_camera/results/models/firenet/0603_011604/checkpoint-epoch970.pth \
    --events_file_path /central/groups/ccameras/event_camera/Davis/slider_depth \
    --loader_type Davis \
    --output_folder results/inference/slider_depth/best