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

BASE_PATH="${BASE_PATH:-./data/new_detailed_heat_sim_f64}"
OUT_ROOT="${OUT_ROOT:-./plots/new_detailed_heat_sim_f64_3d}"
NORMALIZED="${NORMALIZED:-0}"           # 1 => --normalized
FPS="${FPS:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"           # 0 => all
DOWNSAMPLE="${DOWNSAMPLE:-1}"
THRESHOLD_QUANTILE="${THRESHOLD_QUANTILE:-0.90}"
STYLE="${STYLE:-voxel}"
MAX_CLOUD_POINTS="${MAX_CLOUD_POINTS:-12000}"
CLOUD_GAMMA="${CLOUD_GAMMA:-2.2}"
SOURCE_MARKER_MODE="${SOURCE_MARKER_MODE:-none}"
HEAT_THRESHOLD="${HEAT_THRESHOLD:-0.04}"
VIZ_VMAX="${VIZ_VMAX:-5000}"
VIZ_VMIN="${VIZ_VMIN:-}"
VIZ_GAMMA="${VIZ_GAMMA:-1.6}"
VOXEL_ALPHA_BASE="${VOXEL_ALPHA_BASE:-0.06}"
VOXEL_ALPHA_SCALE="${VOXEL_ALPHA_SCALE:-0.86}"
VOXEL_ALPHA_GAMMA="${VOXEL_ALPHA_GAMMA:-0.70}"
CMAP="${CMAP:-YlOrRd}"
ASSUME_DT="${ASSUME_DT:-0.001}"
ELEV="${ELEV:-25}"
AZIM="${AZIM:--45}"
SAVE_FRAMES_EVERY="${SAVE_FRAMES_EVERY:-0}"   # 0 disables PNG frame export
EXPERIMENT="${EXPERIMENT:-}"
EXPERIMENT_PATH="${EXPERIMENT_PATH:-}"
LAST_FRAME_ONLY="${LAST_FRAME_ONLY:-0}" # 1 => only one PNG at final timestep (no mp4)
ALLOW_ALL_EXPERIMENTS="${ALLOW_ALL_EXPERIMENTS:-0}" # 1 => allow processing all experiment_* folders

if [ -z "$EXPERIMENT" ] && [ -z "$EXPERIMENT_PATH" ] && [ "$ALLOW_ALL_EXPERIMENTS" != "1" ]; then
  echo "ERROR: set EXPERIMENT or EXPERIMENT_PATH. For full-bulk rendering set ALLOW_ALL_EXPERIMENTS=1 explicitly."
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
  --voxel-alpha-base "$VOXEL_ALPHA_BASE"
  --voxel-alpha-scale "$VOXEL_ALPHA_SCALE"
  --voxel-alpha-gamma "$VOXEL_ALPHA_GAMMA"
  --cmap "$CMAP"
  --assume-dt "$ASSUME_DT"
  --elev "$ELEV"
  --azim "$AZIM"
  --save-frames-every "$SAVE_FRAMES_EVERY"
)

if [ -n "$VIZ_VMIN" ]; then
  CMD+=(--viz-vmin "$VIZ_VMIN")
fi

if [ "$NORMALIZED" = "1" ]; then
  CMD+=(--normalized)
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
