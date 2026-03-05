"""
Configuration for simulation.heat_sim_initial.

The solver uses:
    Nt = round(T / DT)
and stores Nt + 1 time points (including t=0).

This testset config creates exactly 3 experiments total:
    1 experiment with 5 fires,
    1 experiment with 15 fires,
    1 experiment with 30 fires.
"""

NUM_FIRES_LIST = [5, 15, 30]
EXPERIMENTS_PER_FIRE_COUNT = 1

DATA_ROOT = "./data"
DATASET_NAME = "TEST_DATASET_15s_dt0.01"

T = 15.0
DT = 0.01

DEVICE = "auto"  # "auto", "cpu", or "cuda"
INCLUDE_AXES = True
CHUNK_T = 1


# command: sbatch -J "testset_15s" --export=ALL,SIM_CONFIG=configs/testset_15s_dt0.01_config.py slurm/new_heat_sim_class_cpu.slurm
