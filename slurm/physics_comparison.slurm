#!/bin/bash
#SBATCH --job-name=pit_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --partition=small_gpu
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module purge
export PATH="/beegfs/home/l/lieberta/venv/penn-venv/bin:$PATH"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pitcnn-matplotlib-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
output_dir="${1:?Campaign output directory required}"
export PITCNN_GIT_PROVENANCE_FILE="$output_dir/source_provenance.json"
mode="${2:?Expected calibration or training}"

if [[ "$mode" == calibration ]]; then
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name())'
    python -m unittest discover -s tests -p test_physics_residual.py -v
    for seed in 42 43 44; do
        python -m evaluation.calibrate_physics --seed "$seed" --device cuda \
            --output "$output_dir/scales_seed${seed}.json"
    done
    python - "$output_dir" <<'PY'
import json
import math
from pathlib import Path
import statistics
import sys

out = Path(sys.argv[1])
reports = [json.loads((out / f'scales_seed{s}.json').read_text()) for s in (42, 43, 44)]
weights = [r['scales']['soft_physics']['candidate_weight'] for r in reports]
if any(w is None or not math.isfinite(w) or w <= 0 for w in weights):
    raise ValueError('Invalid calibration; dependent trainings must not start.')
choice = dict(seeds=[42, 43, 44], per_seed_candidates=weights,
              common_soft_weight=statistics.median(weights),
              rule='median of training-only initial gradient-ratio candidates')
(out / 'calibration_choice.json').write_text(json.dumps(choice, indent=2))
print(json.dumps(choice, indent=2))
PY
elif [[ "$mode" == training ]]; then
    index="${SLURM_ARRAY_TASK_ID:?Training requires an array task}"
    if (( index < 0 || index > 14 )); then
        echo "Array index must be between 0 and 14" >&2
        exit 1
    fi
    variants=(A B C D E)
    export PITCNN_SEED=$((42 + index / 5))
    export PITCNN_VARIANT="${variants[index % 5]}"
    export PITCNN_SOFT_WEIGHT
    PITCNN_SOFT_WEIGHT=$(python - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys
print(json.loads((Path(sys.argv[1]) / 'calibration_choice.json').read_text())['common_soft_weight'])
PY
)
    echo "Starting variant=$PITCNN_VARIANT seed=$PITCNN_SEED job=$SLURM_JOB_ID"
    python main.py --config configs/pitcnn_steel_comparison.py
    echo "Completed variant=$PITCNN_VARIANT seed=$PITCNN_SEED job=$SLURM_JOB_ID"
else
    echo "Unknown mode: $mode" >&2
    exit 1
fi
