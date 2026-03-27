import torch

# Training config for the dynamic PITCNN model.
# This run uses the "gt_past" physics residual:
#   (output_n - target_{n-1}) / dt - alpha * laplace(target_{n-1}) - source

train_dtype = torch.float32
epochs = 14
lr = 0.001
loss_weight_schedule = [
    {"epochs": 14, "lp_weight": 0.01, "mse_weight": 1.0},
]
physics_residual_mode = "gt_past"
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_STEEL_MEDIUM_BOTH_gt_past_14ep_lp001"

data_path = "./data/steel_train_medium/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0

run_comment = (
    "Train on steel_train_medium for 14 epochs with combined MSE + physics loss using physics_residual_mode=gt_past and lp_weight=0.01."
)

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# command:
# sbatch -J "tf_gpast14" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_steel_medium_gt_past_100ep.py slurm/main.slurm
