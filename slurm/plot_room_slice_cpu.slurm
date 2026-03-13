#!/bin/bash
#SBATCH --job-name=plot_room_slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p logs

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

BASE_PATH="${BASE_PATH:-./data/testdummy}"
EXPERIMENT="${EXPERIMENT:-experiment_15_20260223_140219}"
STEP_EVERY="${STEP_EVERY:-10}"
VMAX_CLIP="${VMAX_CLIP:-500.0}"
NORMALIZED="${NORMALIZED:-0}"

CMD=(
  python -m evaluation.plot_room_slice
  --base-path "$BASE_PATH"
  --experiment "$EXPERIMENT"
  --step-every "$STEP_EVERY"
  --vmax-clip "$VMAX_CLIP"
)

if [ "$NORMALIZED" = "1" ]; then
  CMD+=(--normalized)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
