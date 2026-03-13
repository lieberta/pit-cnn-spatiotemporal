"""
Configuration for simulation.heat_sim_concrete.

The solver uses:
    Nt = round(T / DT)
and stores every 10th step in the output NPZ.

Example:
    T = 600.0, DT = 0.1 -> Nt = 6000 -> 6001 simulated points, 601 saved points.
"""

NUM_FIRES_LIST = [10, 20, 30]
EXPERIMENTS_PER_FIRE_COUNT = 1

DATA_ROOT = "./data"
DATASET_NAME = "concrete"

T = 600.0
DT = 0.1

DEVICE = "auto"  # "auto", "cpu", or "cuda"
