"""
Configuration for simulation.heat_sim_initial.

The solver uses:
    Nt = round(T / DT)
and stores Nt + 1 time points (including t=0).

Example:
    T = 10.0, DT = 0.01  -> Nt = 1000 -> 1001 data points.
"""

NUM_FIRES_LIST = [10,20,30]
EXPERIMENTS_PER_FIRE_COUNT = 1

DATA_ROOT = "./data"
DATASET_NAME = "TESTSET_DT=0.001_T=15.0"

T = 15.0
DT = 0.001

DEVICE = "auto"  # "auto", "cpu", or "cuda"
INCLUDE_AXES = True
CHUNK_T = 1
