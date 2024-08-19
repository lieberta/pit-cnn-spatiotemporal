#!/bin/bash

#SBATCH --job-name=PECNN_small_a=0		  # create a short name for your job

#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --gres=gpu:1 			# blocks one gpu for the task
#SBATCH --cpus-per-task=36		# 72
#SBATCH --partition=small_gpu

#SBATCH --time=23:59:00                 # total run time limit (HH:MM:SS)

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lieberta@hsu-hh.de


#SBATCH --output=terminal_output_main.log
 
module purge

module load anaconda3/2021.11

module load cuda/11.7.0

eval "$(conda shell.bash hook)"

conda activate /beegfs/home/l/lieberta/venv/penn-venv/

# source ~/.virtualenvs/geospatial-time-series/bin/activate

conda info --envs
 
python main.py
