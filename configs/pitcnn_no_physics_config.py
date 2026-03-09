import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
train_dtype = torch.float32
epochs = 50
lr = 0.001
loss_weight_schedule = [
    {"epochs": 50, "lp_weight": 0.0, "mse_weight": 1.0},
]
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f32_a=0"
# Data subset controls for dataset-size sweeps:
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = "fixed an error in physics loss -> laplacian(output) instead of laplacian(input) + f32"

resume_run_ids_dynamic = []
auto_collect_dynamic = False
