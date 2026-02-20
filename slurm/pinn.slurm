#!/bin/bash
#SBATCH --job-name=pinn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --partition=small_gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load anaconda3/2021.11
# Keep optional CUDA module non-fatal across nodes/module trees.
module load cuda/11.7.0 || true

eval "$(conda shell.bash hook)"
conda activate /beegfs/home/l/lieberta/venv/penn-venv/

BASE_PATH="${BASE_PATH:-./data/laplace_convolution}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
POINTS_PER_SAMPLE="${POINTS_PER_SAMPLE:-8192}"
MODULO="${MODULO:-1}"
ALPHA="${ALPHA:-0.0257}"
LAMBDA_DATA="${LAMBDA_DATA:-1.0}"
LAMBDA_PDE="${LAMBDA_PDE:-1.0}"
HIDDEN_FEATURES="${HIDDEN_FEATURES:-128}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-6}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-./runs/pinn}"
SOURCE_THRESHOLD_RAW="${SOURCE_THRESHOLD_RAW:-1000.0}"
SOURCE_INTENSITY_RAW="${SOURCE_INTENSITY_RAW:-100000.0}"

CMD=(
  python -m training.train_pinn
  --base-path "$BASE_PATH"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --points-per-sample "$POINTS_PER_SAMPLE"
  --modulo "$MODULO"
  --alpha "$ALPHA"
  --lambda-data "$LAMBDA_DATA"
  --lambda-pde "$LAMBDA_PDE"
  --hidden-features "$HIDDEN_FEATURES"
  --hidden-layers "$HIDDEN_LAYERS"
  --val-split "$VAL_SPLIT"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --runs-root "$RUNS_ROOT"
  --source-threshold-raw "$SOURCE_THRESHOLD_RAW"
  --source-intensity-raw "$SOURCE_INTENSITY_RAW"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"
