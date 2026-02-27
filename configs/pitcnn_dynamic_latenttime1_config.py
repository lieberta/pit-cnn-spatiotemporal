import torch

# This file contains training configuration parameters for the model PITCNN_dynamic_latenttime1
epochs = 100
lr = 0.001
lp_weight_list = [1.0]
mse_weight = 0.0  # default for non-scheduled runs
loss_weight_schedule = [
    {"epochs": 50, "lp_weight": 1.0, "mse_weight": 0.0},
    {"epochs": 50, "lp_weight": 0.0, "mse_weight": 1.0},
]
model_class_name = "PITCNN_dynamic_latenttime1"
model_name = model_class_name + "_physics_only_f32_V0.3"
run_comment = (
    "Two-phase run: first 50 epochs physics-only, next 50 epochs mse-only."
)
resume_run_ids_dynamic = []
auto_collect_dynamic = False


# command:
# sbatch -J "latenttime1_32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_latenttime1_config.py slurm/main.slurm
