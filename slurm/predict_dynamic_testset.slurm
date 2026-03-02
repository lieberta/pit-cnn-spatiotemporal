#!/bin/bash

#SBATCH --job-name=predict_dynamic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --partition=small_gpu
#SBATCH --time=23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lieberta@hsu-hh.de
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

: "${RUN_ID:?Set RUN_ID, e.g. dynamic_20260223-161602_d6c999}"
: "${TESTSET_PATH:?Set TESTSET_PATH, e.g. ./data/TESTSET_DT=0.001_T=15.0}"

cmd=(
  python -m evaluation.predict_dynamic_testset
  --run "${RUN_ID}"
  --runs-root "${RUNS_ROOT:-./runs/dynamic}"
  --testset-path "${TESTSET_PATH}"
  --out-root "${OUT_ROOT:-./evaluation/predictions}"
  --device "${DEVICE:-auto}"
  --time-stride "${TIME_STRIDE:-1}"
  --experiment-offset "${EXPERIMENT_OFFSET:-0}"
)

if [[ -n "${MAX_EXPERIMENTS:-}" ]]; then
  cmd+=(--max-experiments "${MAX_EXPERIMENTS}")
fi

"${cmd[@]}"
