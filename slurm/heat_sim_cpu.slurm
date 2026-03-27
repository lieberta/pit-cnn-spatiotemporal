#!/bin/bash
#SBATCH --job-name=heat_sim_cpu
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

SIM_CONFIG="${SIM_CONFIG:-configs/heat_sim_config.py}"
echo "SIM_CONFIG=${SIM_CONFIG}"

python -m simulation.heat_sim --config "${SIM_CONFIG}"

# After dataset generation, automatically render a room-slice plot for the first
# experiment folder. This gives a fast visual sanity check without requiring a
# separate manual plotting job after every simulation run.
DATASET_OUT_ROOT="$(python - "${SIM_CONFIG}" <<'PY'
import importlib.util
import os
import sys

config_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("heat_sim_runtime_config", config_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load config module from: {config_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

data_root = str(getattr(module, "DATA_ROOT", "./data"))
dataset_name = str(getattr(module, "DATASET_NAME", "heat_sim"))
print(os.path.join(data_root, dataset_name))
PY
)"

echo "DATASET_OUT_ROOT=${DATASET_OUT_ROOT}"

FIRST_EXPERIMENT="$(find "${DATASET_OUT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'experiment_*' | sort | head -n 1)"

if [ -n "${FIRST_EXPERIMENT}" ]; then
  FIRST_EXPERIMENT_NAME="$(basename "${FIRST_EXPERIMENT}")"
  echo "Rendering room-slice plot for first experiment: ${FIRST_EXPERIMENT_NAME}"
  python -m evaluation.plot_room_slice \
    --base-path "${DATASET_OUT_ROOT}" \
    --experiment "${FIRST_EXPERIMENT_NAME}" \
    --step-every 1 \
    --vmin-clip 0.0 \
    --vmax-clip 50.0 \
    --subtract-ambient \
    --output-subdir plots
else
  echo "No experiment_* folder found in ${DATASET_OUT_ROOT}; skipping automatic plot generation."
fi
