from configs.reproducibility import seed, split_seed, split_manifest_path, training_source_counts

import torch

# Follow-up training config for the dynamic PITCNN model.
# Branch from the pure-MSE steel_medium run after 20 epochs and continue
# for 10 more epochs with gt_past physics supervision in a new run folder.

train_dtype = torch.float32
epochs = 30
lr = 0.001
loss_weight_schedule = [
    {"epochs": 20, "lp_weight": 0.0, "mse_weight": 1.0},
    {"epochs": 10, "lp_weight": 1.0, "mse_weight": 1.0},
]
physics_residual_mode = "gt_past"
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_STEEL_MEDIUM_FROM_MSE20_TO_gt_past_10ep_lp1"

data_path = "./data/steel_train_medium/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0

run_comment = (
    "Initialize from the 20-epoch steel_train_medium pure-MSE run and continue to epoch 30 with gt_past physics loss at lp_weight=1.0 in a separate run folder."
)

resume_run_ids_dynamic = []
initialize_run_ids_dynamic = ["dynamic_20260323-104107_7e2ea6"]
auto_collect_dynamic = False

# command:
# sbatch -J "tf_gpast30b" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_steel_medium_from_mse20_to_gt_past_10ep.py slurm/main.slurm
