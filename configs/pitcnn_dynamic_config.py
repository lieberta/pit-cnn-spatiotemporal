import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
epochs = 50
lr = 0.0001
lp_weight_list = [1.0]
mse_weight = 1.0  # set to 0.0 for physics-only training (MSE disabled)
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f32_lr=0.0001_V0.3"
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    "V0.3: keeps V0.2 fixes (interior physics residual, simulation-scaled Laplacian, corrected z=0 boundary copy) "
    "and additionally uses laplacian(output_past) in physics loss to match explicit Euler FD step; "
    "simulation source/initial hot zone moved to interior z layers (1:3) and fireplace x/y placement keeps one-cell "
    "distance from Dirichlet boundaries."
)


resume_run_ids_dynamic = []
auto_collect_dynamic = False


# command:
# sbatch -J "dynamic32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_config.py slurm/main.slurm
