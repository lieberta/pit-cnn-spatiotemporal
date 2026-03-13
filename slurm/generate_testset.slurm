#!/bin/bash
#SBATCH --job-name=gen_testset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=small
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

# Tunables via env vars (all optional)
OUT_ROOT="${OUT_ROOT:-./data/testset_20s}"
FIRES_MIN="${FIRES_MIN:-1}"
FIRES_MAX="${FIRES_MAX:-10}"
EXPERIMENTS_PER_FIRE="${EXPERIMENTS_PER_FIRE:-1}"
SIM_TIME_SECONDS="${SIM_TIME_SECONDS:-20}"
NT="${NT:-20000}"
SAVE_EVERY="${SAVE_EVERY:-100}"
NORMALIZE="${NORMALIZE:-1}"   # 1 => --normalize, 0 => skip

CMD=(
  python -m simulation.generate_testset_timed
  --out-root "$OUT_ROOT"
  --fires-min "$FIRES_MIN"
  --fires-max "$FIRES_MAX"
  --experiments-per-fire "$EXPERIMENTS_PER_FIRE"
  --sim-time-seconds "$SIM_TIME_SECONDS"
  --nt "$NT"
  --save-every "$SAVE_EVERY"
  --device cpu
)

if [ "$NORMALIZE" = "1" ]; then
  CMD+=(--normalize)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
