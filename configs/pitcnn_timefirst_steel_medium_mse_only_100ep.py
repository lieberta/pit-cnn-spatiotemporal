from configs.reproducibility import seed, split_seed, split_manifest_path, training_source_counts

import torch

# Training config for the dynamic PITCNN model.
# This run disables the physics term completely and trains with pure MSE.
# physics_residual_mode is still set explicitly because the dynamic training
# entrypoint requires it, but it is irrelevant when lp_weight = 0.0.

train_dtype = torch.float32
epochs = 20
lr = 0.001
loss_weight_schedule = [
    {"epochs": 20, "lp_weight": 0.0, "mse_weight": 1.0},
]
physics_residual_mode = "gt_future"
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_STEEL_MEDIUM_MSE_ONLY_20ep_resume"

data_path = "./data/steel_train_medium/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0

run_comment = (
    "Continue the steel_train_medium pure-MSE run from epoch 14 to epoch 20 with no active physics term."
)

resume_run_ids_dynamic = ["dynamic_20260323-104107_7e2ea6"]
auto_collect_dynamic = False

# command:
# sbatch -J "tf_mse20r" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_steel_medium_mse_only_100ep.py slurm/main.slurm
