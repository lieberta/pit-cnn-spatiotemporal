# PIT-CNN

Surrogate modeling of transient 3D heat transfer using convolutional neural networks with physics-informed loss functions.
This repo contains tools to simulate heat transfer, preprocess data, train neural networks, and evaluate models.

# Workflow


# 1. Generate simulation data

Runs a 3D transient heat equation with random rectangular heat sources (“fireplaces”).

```bash
python -m simulation.heat_sim_class
```

Saves results in ./data/testset/experiment_* with: <br>

heat_equation_solution.npz → temperatures, grid, time <br>

fireplace_simulation_results.txt → metadata <br>


# 2. Preprocess experiments

Normalize temperature fields across all experiments.

```bash
python -m simulation.preprocess
```

Computes global min/max <br>

Saves normalized .npz files in each experiment folder <br>

Trained checkpoints saved in ./models/


# 3. Train models

Run static or dynamic CNNs with physics-informed loss.

```bash
python main.py --config configs/pitcnn_dynamic_config.py
```

- Static models → PICNN_static

- Dynamic models → PITCNN_dynamic

## 4. Train with config + `main.py`

Training is controlled by Python config files.

### Option A: Central dtype in `configs/train_config.py`

Set only the global dtype there:

- `TRAIN_DTYPE` (e.g. `torch.float32` or `torch.float64`)

`main.py` and the training modules import this dtype directly, independent from the run config file.

### Option B: Use a run config in `configs/`

Use one of these config files for epochs/model class/run naming:

```bash
python main.py --config configs/picnn_static_config.py
python main.py --config configs/pitcnn_dynamic_config.py
python main.py --config configs/pitcnn_timefirst_config.py
```
### Notes

- Static mode runs over `predicted_times` and `a_list`.
- Dynamic mode runs over `a_list`.
- Run artifacts are written to `runs/static/...` or `runs/dynamic/...` with a generated `run_id` and `config.json`.
- For resume training, set the corresponding `resume_run_ids_*` lists in `main.py` (or enable auto-collect flags).

## 5. Model Overview

The project currently supports these model classes (configured via `model_class_name`):

- `PICNN_static`: static model for a fixed prediction horizon (`predicted_time`).
- `PITCNN_dynamic`: dynamic model with time as an additional input in latent space.
- `PITCNN_dynamic_timefirst`: dynamic model that injects time earlier in the network.
- `PITCNN_dynamic_batchnorm` and `PITCNN_dynamic_latenttime1`: additional dynamic variants available in `main.py` registry.

All CNN model classes inherit from shared training base classes (`BaseModel` / `BaseModel_dynamic`).

## 6. Physics Loss (Dynamic)

For dynamic training (`CombinedLoss_dynamic`), the temporal derivative is computed via finite difference between two model predictions:

- `u(t)` from `self(input, t)`
- `u(t-Δt)` from `self(input, t_past)`
- derivative: `(u(t) - u(t-Δt)) / (t - t_past)`

This replaces the older coarse approximation based on `(output - input) / t`.
