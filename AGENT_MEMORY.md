# Agent Memory

Last updated: 2026-02-16

## Project Snapshot
- Repository trains CNN/PINN models for heat-equation data.
- Two main training families:
  - Static CNN: `PICNN_static`
  - Dynamic CNN: `PITCNN_*` variants with time input
- Main orchestration entrypoint: `main.py`.

## Training Entrypoints
- Static training loop: `training/train_picnn_static.py` (`BaseModel.train_model`).
- Dynamic training loop: `training/train_pitcnn_dynamic.py` (`BaseModel_dynamic.train_model`).
- PINN training script: `train_pinn.py`.

## Dataset Layout and Loaders
- Dataset module: `dataset.py`.
- Expected simulation files per experiment folder:
  - `normalized_heat_equation_solution.npz`
  - key: `temperature`
- Static loader:
  - `HeatEquationMultiDataset`
  - uses t=0 as input and `predicted_time*10` frame as target.
- Dynamic loader:
  - `HeatEquationMultiDataset_dynamic`
  - returns `((input_t0, predicted_time_tensor), target_t)`.
  - includes a simple one-file cache (`data_cache`).
- PINN loader:
  - `HeatEquationPINNDataset`
  - samples random collocation points from full spatiotemporal fields.

## Loss and Physics Notes
- Loss definitions: `training/loss.py`.
- `CombinedLoss_dynamic` uses normalization-aware source term constants:
  - `source_intensity / 27353.34765625`
  - fire threshold `(1000.0 - 20.0) / 27353.34765625`
- Laplacian implemented via fixed 3D convolution kernel (`Laplacian3DLayer`).

## Checkpointing and Metrics
- Utilities: `training/train_utils.py`.
- `load_checkpoint` supports both dict checkpoints and plain state dicts.
- Resume logic now recasts optimizer state tensors to current model dtype/device after loading.
- Metrics are appended to per-run CSV via `append_metrics_row`.

## Run/Experiment Structure
- Runs are grouped under `./runs/static`, `./runs/dynamic`, `./runs/pinn`.
- Each run typically stores:
  - `config.json`
  - `metrics.csv`
  - model checkpoints (`epoch_*.pth`, `<run_id>.pth` or `last/best` for PINN)

## Current Precision Convention
- Training code is currently set to `float32`.
- Dtype control points exist in:
  - `main.py` (`TRAIN_DTYPE`)
  - `training/train_picnn_static.py` (`TRAIN_DTYPE`)
  - `training/train_pitcnn_dynamic.py` (`TRAIN_DTYPE`)
  - `train_pinn.py` (`TRAIN_DTYPE`)
  - `dataset.py` tensor and numpy casts
  - `training/loss.py` Laplacian kernel dtype

## Operational Notes
- `main.py` has `run_mode` switch (`static` or `dynamic`) and optional resume ID auto-collection.
- Dynamic model choice is controlled by `model_class_dynamic` in `main.py`.
- SLURM scripts exist in `slurm/` for cluster execution.

## Pitfalls to Remember
- Keep model dtype, dataset dtype, and loss kernel dtype aligned to avoid implicit casts.
- When resuming old checkpoints after dtype changes, optimizer-state casting is required.
- Avoid changing user-specific experiment naming in `main.py` unless explicitly requested.
