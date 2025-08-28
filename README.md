# PIT-CNN
This module generates synthetic temperature fields for a 3D room using a finite-difference explicit scheme. It supports multiple rectangular heat sources (“fireplaces”), saves time-sampled temperature tensors, and writes metadata for each experiment. The data can be used to train spatiotemporal surrogate models (e.g., PIT-CNN).

Features

3D Laplacian via conv3d (learnable-free kernel) for fast CPU/GPU execution

Explicit time stepping with constant source term

Randomized rectangular heat sources on the floor plane

Hard Dirichlet boundary at 20 °C on all faces

NPZ output (temperature, time, x, y, z) + a human-readable info file

Batch generation over multiple fire counts

Requirements

Python 3.9+

PyTorch

NumPy

pip install torch numpy

File Overview

heat_sim_class.py

Laplacian3D: fixed 3×3×3 kernel module implementing the 3D Laplacian.

HeatSimulation: sets up the grid, sources, IC/BC, integrates in time, and saves results.

run_experiment: convenience wrapper to run a single experiment.

__main__: example loop that generates experiments for fire counts [3,4,5,6,7,8,9].

Quickstart
1) Run the demo (uses GPU if available)
python heat_sim_class.py


This will create dated experiment folders under:

./data/testset/experiment_<num_fires>_<YYYYMMDD_HHMMSS>/
├── heat_equation_solution.npz
└── fireplace_simulation_results.txt

2) Use from Python
import torch
from heat_sim_class import HeatSimulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sim = HeatSimulation(
    num_fires=5,       # number of rectangular sources
    alpha=0.0257,      # thermal diffusivity [m^2/s]
    Lx=6.3, Ly=3.1, Lz=1.5,   # domain size [m]
    Nx=64, Ny=32, Nz=16,      # grid points
    T=10.0, Nt=10000,         # total time [s], steps
    device=device
)

sim.run_simulation()
sim.save_results()

