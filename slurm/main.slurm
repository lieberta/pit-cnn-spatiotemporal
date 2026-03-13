#!/bin/bash

#SBATCH --job-name=None		  # create a short name for your job

#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --gres=gpu:1 			# blocks one gpu for the task
#SBATCH --cpus-per-task=36		# 72
#SBATCH --partition=small_gpu

#SBATCH --time=23:59:00                 # total run time limit (HH:MM:SS)

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lieberta@hsu-hh.de


#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
 
module purge

module load anaconda3/2021.11

module load cuda/11.7.0

eval "$(conda shell.bash hook)"

conda activate /beegfs/home/l/lieberta/venv/penn-venv/

conda info --envs

python main.py --config "${TRAIN_CONFIG:-configs/pitcnn_dynamic_config.py}"
