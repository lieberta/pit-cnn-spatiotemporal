#!/bin/bash
#SBATCH --job-name=viz_testset_3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=small
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

BASE_PATH="${BASE_PATH:-./data/testset_20s}"
OUT_ROOT="${OUT_ROOT:-./plots/testset_20s_3d}"
NORMALIZED="${NORMALIZED:-0}"           # 1 => --normalized
FPS="${FPS:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-2}"
MAX_FRAMES="${MAX_FRAMES:-0}"           # 0 => all
DOWNSAMPLE="${DOWNSAMPLE:-2}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.90}"
STYLE="${STYLE:-cloud}"
MAX_CLOUD_POINTS="${MAX_CLOUD_POINTS:-12000}"
CLOUD_GAMMA="${CLOUD_GAMMA:-2.2}"
SOURCE_MARKER_MODE="${SOURCE_MARKER_MODE:-all}"
HEAT_THRESHOLD="${HEAT_THRESHOLD:-0.04}"
ELEV="${ELEV:-25}"
AZIM="${AZIM:--45}"
SAVE_FRAMES_EVERY="${SAVE_FRAMES_EVERY:-10}"   # 0 disables PNG frame export
EXPERIMENT="${EXPERIMENT:-}"

CMD=(
  python -m evaluation.visualize_testset_3d
  --base-path "$BASE_PATH"
  --out-root "$OUT_ROOT"
  --fps "$FPS"
  --frame-stride "$FRAME_STRIDE"
  --max-frames "$MAX_FRAMES"
  --downsample "$DOWNSAMPLE"
  --threshold-quantile "$THRESHOLD_QUANTILE"
  --style "$STYLE"
  --max-cloud-points "$MAX_CLOUD_POINTS"
  --cloud-gamma "$CLOUD_GAMMA"
  --source-marker-mode "$SOURCE_MARKER_MODE"
  --heat-threshold "$HEAT_THRESHOLD"
  --elev "$ELEV"
  --azim "$AZIM"
  --save-frames-every "$SAVE_FRAMES_EVERY"
)

if [ "$NORMALIZED" = "1" ]; then
  CMD+=(--normalized)
fi

if [ -n "$EXPERIMENT" ]; then
  CMD+=(--experiment "$EXPERIMENT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
