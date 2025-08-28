# PIT-CNN

Surrogate modeling of transient 3D heat transfer using convolutional neural networks with physics-informed loss functions.
This repo contains tools to simulate heat transfer, preprocess data, train neural networks, and evaluate models.

## Setup
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt


# Workflow

# 1. Generate simulation data
Runs a 3D transient heat equation with random rectangular heat sources (“fireplaces”).

```bash
python heat_sim_class.py
```
Saves results in ./data/testset/experiment_* with:
heat_equation_solution.npz → temperatures, grid, time
fireplace_simulation_results.txt → metadata

# 2. Preprocess experiments

Normalize temperature fields across all experiments.
```bash
python preprocess.py
```
Computes global min/max
Saves normalized .npz files in each experiment folder


Trained checkpoints saved in ./models/

# 3. Train models
Run static or dynamic CNNs with physics-informed loss.
```bash
python main.py
```
Static models → PICNN_static
Dynamic models → PECNN_dynamic
