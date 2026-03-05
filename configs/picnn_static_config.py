import torch

train_dtype = torch.float64
epochs = 10
lr = 0.001
a_list = [0, 1]
model_class_name = "PICNN_static"
model_name = model_class_name + "_f64_normsource"
# Data subset controls (used by dynamic mode; kept here for config consistency):
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = "float64 + normalized Heat  Source; PICNN_static run from configs/picnn_static.py"
