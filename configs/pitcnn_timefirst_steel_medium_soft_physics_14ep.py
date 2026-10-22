from configs.reproducibility import seed, split_seed, split_manifest_path, training_source_counts

import torch

# Training config for the dynamic PITCNN model.
# This run uses the "soft_physics" residual:
#   (output_n - output_{n-1}) / dt - alpha * laplace(input_t0)
# with input_t0 interpreted as the initial temperature U0.

train_dtype = torch.float32
epochs = 14
lr = 0.001
loss_weight_schedule = [
    {"epochs": 14, "lp_weight": 1.0, "mse_weight": 1.0},
]
physics_residual_mode = "soft_physics"
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_STEEL_MEDIUM_BOTH_soft_physics_14ep_lp1"

data_path = "./data/steel_train_medium/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0

run_comment = (
    "Train on steel_train_medium for 14 epochs with combined MSE + soft_physics loss using lp_weight=1.0. The soft_physics residual uses (output_n - output_{n-1}) / dt - alpha * laplace(input_t0) without a source term."
)

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# command:
# sbatch -J "tf_soft14" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_steel_medium_soft_physics_14ep.py slurm/main.slurm
