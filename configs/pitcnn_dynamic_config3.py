import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
train_dtype = torch.float32
epochs = 100
lr = 0.001
lp_weight_list = [0.0]
mse_weight = 0.0  # set to 0.0 for physics-only training (MSE disabled)
loss_weight_schedule = [
    {"epochs": 50, "lp_weight": 0.0, "mse_weight": 1.0},
    {"epochs": 50, "lp_weight": 1.0, "mse_weight": 0.0}
]
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_DATADRIVENONLY_100EP_f32_lr=0.001_V0.4"
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    " "
)


resume_run_ids_dynamic = []
auto_collect_dynamic = False


# command:
# sbatch -J "dynamic32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_config3.py slurm/main.slurm
