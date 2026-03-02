# Agent Memory

Last updated: 2026-02-26

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
- Dynamic trainer currently sets `t_past = t - 0.001` (1 simulation step) and computes `output_past = self(input, t_past)` in both train and val loops.
- Source term in `CombinedLoss_dynamic` remains normalization-aware:
  - `source_intensity / 27353.34765625`
  - fire threshold uses `source_threshold=500.0` (normalized with dataset min/max)
- Dynamic physics residual is now enforced only on interior cells (`[:, :, 1:-1, 1:-1, 1:-1]`).
- Laplacian in `Laplacian3DLayer` is now scaled like the simulation stencil:
  - center `-2*(1/dx^2 + 1/dy^2 + 1/dz^2)`
  - 6-neighbor weights `1/dx^2`, `1/dy^2`, `1/dz^2`
  - defaults use simulation geometry (`Lx=6.3, Ly=3.1, Lz=1.5, Nx=64, Ny=32, Nz=16`)
- Physics residual now uses `laplacian(output_past)` (explicit-Euler aligned with simulator step).

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
- `README.md` contains current workflow, config usage, model overview, dynamic physics-loss notes, and the V0.2 physics-alignment update details.
- Legacy `readme.txt` has been removed after migration of relevant content.

## Model Boundary Notes
- `models/pitcnn_latenttime.py` had a z=0 boundary assignment bug where a single scalar was broadcast to the full plane.
- Fixed in both `PITCNN_dynamic` and `PITCNN_dynamic_latenttime1` by copying full z=0 plane from input.
- Remaining mismatch: model still copies boundaries from `u0`, while simulator enforces constant Dirichlet ambient each timestep.

## Config/Run Notes (2026-02-26)
- `configs/pitcnn_dynamic_config.py` and `configs/pitcnn_timefirst_config.py` were aligned to V0.2 defaults:
  - `epochs = 10`
  - `model_name = model_class_name + "_f32_V0.2"`
  - `resume_run_ids_dynamic = []`
  - updated run comments reflecting physics/laplacian/boundary fixes
- `configs/pitcnn_dynamic_config.py` now uses V0.3 naming/comment and `a_list = [0.1, 1]`.
- Dynamic run directories were archived under `runs/dynamic/pre-normsource-incident/` except `runs/dynamic/PITCNN_dynamic_f32_a=0`.

## Simulation/Data Notes (2026-02-26)
- Simulation file for current dataset generation: `simulation/heat_sim_initial.py` (formerly `new_heat_sim_class.py`, not `new_sim_class.py`).
- Source placement was adjusted to avoid boundaries:
  - fireplaces keep one-cell distance to x/y boundaries
  - initial hot zone and source term moved from `:2` to interior `z=1:3`
- PDE scheme in simulator: explicit Euler FD
  - `u[n+1] = u[n] + alpha*dt*laplacian(u[n]) + dt*source`
  - Dirichlet boundaries are set after each step.

## Debug Findings (a=0 vs a=1)
- `loss_components.csv` columns used in both runs:
  - `run_id, epoch, train_mse_loss, train_physics_loss, val_mse_loss, val_physics_loss, a`
- `total_loss` is not logged in `loss_components.csv`; it is in `metrics.csv` as:
  - `total = mse + a * physics`
- Observed scale snapshot from inspected runs:
  - data-only (`a=0`, 50 epochs): train/val MSE reaches around `1e-7 ... 1e-5`; physics is hard-zero by construction.
  - physics (`a=1`, 10 epochs): train/val MSE roughly `1e-4 ... 1e-3`, physics term roughly `1e-5 ... 1e-2`.
- Interpretation used for debugging:
  - smaller `a=0` numbers are not contradictory, because only one objective is optimized.
  - MSE and PDE residual are not naturally unit-comparable; direct magnitude comparison can be misleading.
  - the compared runs were not strictly apples-to-apples (`epochs`, seed, and code state differed).

## Physics Alignment Checklist (2026-02-26)
- Simulator (`simulation/heat_sim_initial.py`) currently uses:
  - explicit Euler: `u[n+1] = u[n] + alpha*dt*laplacian(u[n]) + dt*source`
  - geometry/time: `Nx,Ny,Nz=64,32,16`, `Lx,Ly,Lz=6.3,3.1,1.5`, so `dx=dy=dz=0.1`, `dt=0.001`.
  - constant Dirichlet boundaries set every step on all six faces.
- Physics loss (`training/loss.py`) currently uses:
  - temporal derivative from two forward passes: `(u(t)-u(t-dt))/dt`
  - Laplacian evaluated at past state: `laplacian(output_past)` (explicit-Euler aligned)
  - residual only on interior voxels `1:-1` in x/y/z.
- Remaining known mismatch:
  - model forward (`models/pitcnn_latenttime.py`) still clamps boundaries to `u0` values, while simulator enforces constant ambient boundaries each timestep.

## Visualization Notes (2026-02-26)
- Visualization script renamed:
  - from `evaluation.visualize_testset_3d`
  - to `evaluation.visualize_heatvid_3d`
- Added zarr support and robust time-axis handling (`--assume-dt`) for datasets with index-like time arrays.
- Added fixed visualization controls (`--viz-vmin`, `--viz-vmax`, `--viz-gamma`, `--cmap`) and collision-safe video naming.

## Pitfalls to Remember
- Keep dtype aligned end-to-end (model, dataset tensors, loss computations) using `configs/train_config.py`.
- For cluster runs, avoid editing configs while jobs are queued/running unless job scheduling state is controlled (`hold/release`) to ensure reproducibility.
- When resuming checkpoints after dtype changes, ensure optimizer-state recasting remains intact (already handled in `train_utils.py`).
- User preference: avoid rare fallback paths as the default solution; prefer the direct, regular workflow unless a fallback is explicitly requested or necessary.

## Working Preferences

### Config-First, Explicit Behavior
- Treat config files as the source of truth for critical training and runtime settings.
- Do not add hidden code defaults for values that should come from project config.
- Prefer explicit defaults in shared config files over hardcoded runtime defaults when users may tune the behavior.
- Required config fields should be validated early and fail clearly if missing or invalid.
- Validate config values as they are extracted, including type and range where relevant.
- Keep behavior traceable to config, CLI, or explicit function arguments.
- When merging CLI overrides into config-derived values, only apply explicitly provided CLI values; absent optional CLI args must not overwrite config values.
- Keep top-level config ownership explicit across files to avoid silent key collisions.
- Config-loading helpers should validate the expected shape and fail clearly on wrong structures.
- Keep machine-local paths and environment-specific binaries in `.env` or environment variables, not in shared project configs.
- If a CLI supports environment fallback for a required value, document precedence explicitly and fail with an error that names both sources.

### Keep It Minimal
- Implement the smallest correct solution that satisfies the requirement.
- Prefer extending existing code paths over adding parallel flows.
- Remove superseded code when replacing behavior.
- Do not add feature flags, compatibility shims, or alternate modes unless explicitly required.

### Work in Small Steps
- Do not change large parts of the training pipeline, config handling, and run artifacts all at once.
- Finish one coherent change at a time, validate it, then move to the next change.
- For larger refactors, keep phases small enough that each step can be checked against actual project artifacts such as `config.json`, `metrics.csv`, checkpoints, and SLURM submission flow.

### Validate Continuously
- With each meaningful code change, run the smallest relevant validation step available.
- Prefer targeted validation first, for example syntax checks, config loading, a focused script run, or inspection of produced run artifacts.
- Do not continue stacking changes on top of known failures; fix the failure before moving on.
- When changing artifact schema, CLI behavior, or config fields, update implementation and the relevant docs in the same change to avoid drift.

### Review and Simplify
- After making a change work, do a short simplification pass.
- Check for regressions, stale assumptions, edge cases, nondeterministic behavior, and resume-training compatibility issues.
- Compare the implemented behavior against the intended training workflow and close any obvious gaps before moving on.

## Coding Style and Comments
- Prefer clear, explicit code over clever shortcuts.
- Keep naming precise and behavior-oriented.
- Add comments where intent, invariants, or non-obvious behavior need explanation; do not add comments that merely restate syntax.
- Use docstrings for outward-facing helpers, public utilities, training/config helpers, and non-trivial classes when they benefit readability.
- Keep code comments and docstrings runtime-oriented; they should describe behavior and invariants, not planning phases or private workflow notes.
- Define abbreviations on first use in docs or code comments when the meaning may not be obvious in this project context.
- Add durable project terms or recurring abbreviations to `GLOSSARY.md` when that file exists and is actively used.
- Prefer normal top-level imports by default; use lazy imports only when there is a concrete benefit such as heavy optional dependencies, startup cost, or better testability.
