# Agent Memory

Last updated: 2026-02-20

## Project Snapshot
- Repository trains CNN/PINN surrogates for 3D transient heat-transfer data.
- Main training families:
  - Static CNN: `PICNN_static`
  - Dynamic CNN: `PITCNN_*` variants with explicit time input
  - PINN: `models/pinn.py` via separate training script
- Main orchestration entrypoint: `main.py`.

## Training Entrypoints
- Static training loop: `training/train_picnn_static.py` (`BaseModel.train_model`).
- Dynamic training loop: `training/train_pitcnn_dynamic.py` (`BaseModel_dynamic.train_model`).
- PINN training script: `training/train_pinn.py`.

## Config and Dtype Conventions
- Central dtype source: `configs/train_config.py` (`TRAIN_DTYPE`).
- `TRAIN_DTYPE` is imported directly in:
  - `main.py`
  - `dataset.py`
  - `training/train_picnn_static.py`
  - `training/train_pitcnn_dynamic.py`
  - `training/train_pinn.py`
- Run/job params (epochs, model class/name, a-list, comments) come from run config files passed via `--config` (e.g. `configs/pitcnn_dynamic_config.py`).
- `main.py` no longer expects `TRAIN_DTYPE` inside run configs.
- `configs/` is now a package (`configs/__init__.py`) to support imports.

## Dataset Layout and Loaders
- Dataset module: `dataset.py`.
- Expected file per experiment folder:
  - `normalized_heat_equation_solution.npz` with key `temperature`
- Static loader:
  - `HeatEquationMultiDataset`
  - uses `t=0` as input and `predicted_time*10` as target index.
- Dynamic loader:
  - `HeatEquationMultiDataset_dynamic`
  - returns `((input_t0, predicted_time_tensor), target_t)`.
  - keeps a simple one-file cache (`data_cache`).
- PINN loader:
  - `HeatEquationPINNDataset`
  - samples random collocation points from full spatiotemporal fields.
- Dataset numpy casting follows `TRAIN_DTYPE` via `NP_DTYPE` mapping (`float32`/`float64`).

## Loss and Physics Notes
- Loss definitions: `training/loss.py`.
- `CombinedLoss_dynamic` temporal derivative now uses finite difference between two model outputs:
  - `u(t)` and `u(t_past)`
  - derivative `(u(t) - u(t_past)) / (t - t_past)`
- Dynamic trainer currently sets `t_past = t - 0.1` and computes `output_past = self(input, t_past)` in both train and val loops.
- Source term in `CombinedLoss_dynamic` remains normalization-aware:
  - `source_intensity / 27353.34765625`
  - fire threshold `(1000.0 - 20.0) / 27353.34765625`
- Laplacian is implemented as fixed 3D conv kernel and cast to input dtype/device in forward.

## Checkpointing and Metrics
- Utilities: `training/train_utils.py`.
- `load_checkpoint` supports dict checkpoints and plain state dicts.
- On resume, optimizer floating-point state is recast to current model dtype/device.
- Metrics are appended to per-run CSV via `append_metrics_row`.

## Run/Experiment Structure
- Runs under `./runs/static`, `./runs/dynamic`, `./runs/pinn`.
- Typical run artifacts:
  - `config.json`
  - `metrics.csv`
  - model checkpoints (`epoch_*.pth`, `<run_id>.pth`, or PINN `last/best`)

## Main/SLURM Behavior
- `main.py` chooses static vs dynamic based on `model_class_name` from run config.
- Default run config path in `main.py`: `configs/pitcnn_dynamic_config.py` (override via `--config` or `TRAIN_CONFIG` env).
- SLURM default in `slurm/main.slurm` matches the same config path fallback.

## Docs Status
- `README.md` contains current workflow, config usage, model overview, and dynamic physics-loss notes.
- Legacy `readme.txt` has been removed after migration of relevant content.

## Pitfalls to Remember
- Keep dtype aligned end-to-end (model, dataset tensors, loss computations) using `configs/train_config.py`.
- For cluster runs, avoid editing configs while jobs are queued/running unless job scheduling state is controlled (`hold/release`) to ensure reproducibility.
- When resuming checkpoints after dtype changes, ensure optimizer-state recasting remains intact (already handled in `train_utils.py`).
