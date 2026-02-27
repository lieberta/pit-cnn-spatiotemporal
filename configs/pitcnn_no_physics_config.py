import torch

# This file contains training configuration parameters for the model PITCNN_dynamic
epochs = 50
lr = 0.001
lp_weight_list = [0.0]
mse_weight = 1.0
model_class_name = "PITCNN_dynamic"
model_name = model_class_name + "_f32_a=0"
# Data subset controls for dataset-size sweeps:
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = "fixed an error in physics loss -> laplacian(output) instead of laplacian(input) + f32"
