#!/bin/bash

#SBATCH --job-name=lieberta_pecnn_code:yellow		  # create a short name for your job

#SBATCH --nodes=1                         # node count

#SBATCH --ntasks=1                        # total number of tasks across all nodes

#SBATCH --partition=cpu

#SBATCH --time=23:59:00                   # total run time limit (HH:MM:SS)

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lieberta@hsu-hh.de


#SBATCH --output=terminal_output_evaluate.log
 
module purge

module load anaconda3/2021.11

module load cuda/11.7.0

eval "$(conda shell.bash hook)"

conda activate /beegfs/home/l/lieberta/venv/penn-venv/

conda info --envs
 
python evaluate.py
