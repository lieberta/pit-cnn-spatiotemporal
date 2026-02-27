"""
Configuration for simulation.new_heat_sim_class.

The solver uses:
    Nt = round(T / DT)
and stores Nt + 1 time points (including t=0).

Example:
    T = 10.0, DT = 0.01  -> Nt = 1000 -> 1001 data points.
"""

NUM_FIRES_LIST = [5,6,7,8,9,10]
EXPERIMENTS_PER_FIRE_COUNT = 100

DATA_ROOT = "./data"
DATASET_NAME = "new_sim_10s_dt01"

T = 10.0
DT = 0.1

DEVICE = "auto"  # "auto", "cpu", or "cuda"
INCLUDE_AXES = True
CHUNK_T = 1
