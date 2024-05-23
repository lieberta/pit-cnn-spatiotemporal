#!/bin/bash

#SBATCH --job-name=lieberta_heatsim_code:yellow		  # create a short name for your job

#SBATCH --nodes=1                         # node count

#SBATCH --ntasks=1                        # total number of tasks across all nodes

#SBATCH --partition=small_gpu

#SBATCH --time=23:59:00                   # total run time limit (HH:MM:SS)

#SBATCH --mail-type=ALL

#SBATCH --mail-user=lieberta@hsu-hh.de


#SBATCH --output=terminal_output_sim.log
 
module purge

module load anaconda3/2021.11

module load cuda

eval "$(conda shell.bash hook)"

source ~/.virtualenvs/geospatial-time-series/bin/activate

conda info --envs
 
python heat_sim_class.py
