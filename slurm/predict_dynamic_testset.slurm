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
#SBATCH --chdir=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn
#SBATCH --output=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/logs/%x_%j.out
#SBATCH --error=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/beegfs/home/l/lieberta/projects/physics-enhanced-cnn}"
cd "${REPO_ROOT}"

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

# Make sure Python can import this repository's modules when the job starts from SLURM.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Accept either a full run directory path or a run id and normalize both to one variable.
if [[ -n "${MODEL_DIR:-}" ]]; then
  RUN_TARGET="${MODEL_DIR}"
elif [[ -n "${RUN_ID:-}" ]]; then
  RUN_TARGET="${RUN_ID}"
else
  echo "Set either MODEL_DIR or RUN_ID." >&2
  exit 1
fi

if [[ -n "${TESTSET_PATH:-}" ]]; then
  RESOLVED_TESTSET_PATH="${TESTSET_PATH}"
elif [[ -n "${TESTSET_NAME:-}" ]]; then
  if [[ -n "${DATA_ROOT:-}" ]]; then
    RESOLVED_TESTSET_PATH="${DATA_ROOT%/}/${TESTSET_NAME}"
  else
    RESOLVED_TESTSET_PATH="${REPO_ROOT}/data/${TESTSET_NAME}"
  fi
else
  echo "Set either TESTSET_PATH or TESTSET_NAME." >&2
  exit 1
fi

cmd=(
  python -m evaluation.predict_dynamic_testset
  --run "${RUN_TARGET}"
  --runs-root "${RUNS_ROOT:-${REPO_ROOT}/runs/dynamic}"
  --testset-path "${RESOLVED_TESTSET_PATH}"
  --device "${DEVICE:-auto}"
  --time-stride "${TIME_STRIDE:-1}"
  --experiment-offset "${EXPERIMENT_OFFSET:-0}"
)

if [[ -n "${MAX_EXPERIMENTS:-}" ]]; then
  cmd+=(--max-experiments "${MAX_EXPERIMENTS}")
fi

printf 'Running command:\n'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
