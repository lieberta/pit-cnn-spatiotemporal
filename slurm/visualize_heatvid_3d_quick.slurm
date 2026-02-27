#!/bin/bash
#SBATCH --job-name=viz_quick_3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=small
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load anaconda3/2021.11

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

BASE_PATH="${BASE_PATH:-./data/new_detailed_heat_sim_f64}"
OUT_ROOT="${OUT_ROOT:-./plots/new_detailed_heat_sim_f64_3d}"
NORMALIZED="${NORMALIZED:-0}"
FPS="${FPS:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-1}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.50}"
STYLE="${STYLE:-voxel}"
MAX_CLOUD_POINTS="${MAX_CLOUD_POINTS:-12000}"
CLOUD_GAMMA="${CLOUD_GAMMA:-2.2}"
SOURCE_MARKER_MODE="${SOURCE_MARKER_MODE:-none}"
HEAT_THRESHOLD="${HEAT_THRESHOLD:-0.01}"
VIZ_VMAX="${VIZ_VMAX:-5000}"
VIZ_VMIN="${VIZ_VMIN:-}"
VIZ_GAMMA="${VIZ_GAMMA:-1.6}"
CMAP="${CMAP:-YlOrRd}"
ASSUME_DT="${ASSUME_DT:-0.001}"
ELEV="${ELEV:-25}"
AZIM="${AZIM:--45}"
SAVE_FRAMES_EVERY="${SAVE_FRAMES_EVERY:-0}"
EXPERIMENT="${EXPERIMENT:-}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-}"
LAST_FRAME_ONLY="${LAST_FRAME_ONLY:-1}"

if [ -z "$EXPERIMENT" ] && [ -z "$EXPERIMENT_PATH" ]; then
  echo "ERROR: quick mode requires EXPERIMENT or EXPERIMENT_PATH to avoid bulk rendering."
  exit 1
fi

CMD=(
  python -m evaluation.visualize_heatvid_3d
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
  --viz-vmax "$VIZ_VMAX"
  --viz-gamma "$VIZ_GAMMA"
  --cmap "$CMAP"
  --assume-dt "$ASSUME_DT"
  --elev "$ELEV"
  --azim "$AZIM"
  --save-frames-every "$SAVE_FRAMES_EVERY"
)

if [ "$NORMALIZED" = "1" ]; then
  CMD+=(--normalized)
fi

if [ -n "$VIZ_VMIN" ]; then
  CMD+=(--viz-vmin "$VIZ_VMIN")
fi

if [ -n "$EXPERIMENT" ]; then
  CMD+=(--experiment "$EXPERIMENT")
fi

if [ -n "$EXPERIMENT_PATH" ]; then
  CMD+=(--experiment-path "$EXPERIMENT_PATH")
fi

if [ "$LAST_FRAME_ONLY" = "1" ]; then
  CMD+=(--last-frame-only)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
