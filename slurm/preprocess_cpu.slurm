#!/bin/bash
#SBATCH --job-name=preprocess_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=small
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p logs

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

echo "Running on host: $(hostname)"
echo "Python: $(which python)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m simulation.preprocess
