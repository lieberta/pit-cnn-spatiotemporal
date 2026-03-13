"""
Configuration for simulation.heat_sim_initial.

The solver uses:
    Nt = round(T / DT)
and stores Nt + 1 time points (including t=0).

Example:
    T = 10.0, DT = 0.01  -> Nt = 1000 -> 1001 data points.
"""

NUM_FIRES_LIST = [15]
EXPERIMENTS_PER_FIRE_COUNT = 100

DATA_ROOT = "./data"
DATASET_NAME = "Equilibriumset_30s_dt0.001"

T = 30.0
DT = 0.001

DEVICE = "auto"  # "auto", "cpu", or "cuda"
INCLUDE_AXES = True
CHUNK_T = 1


# command: sbatch -J "new_sim_cpu" --export=ALL,SIM_CONFIG=configs/sim_config.py slurm/new_heat_sim_class_cpu.slurm