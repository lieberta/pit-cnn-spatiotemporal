import torch

epochs = 50
lr = 0.001
lp_weight_list = [0.0]
mse_weight = 1.0
loss_weight_schedule = [
    {"epochs": 50, "lp_weight": 0.0, "mse_weight": 1.0},
]
model_class_name = "PITCNN_dynamic_timefirst"
model_name = model_class_name + "_f32_lp0.0_mse1.0_50ep_V0.3"
# Data subset controls for dataset-size sweeps:
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    "50 epochs with constant lp_weight=0.0 and mse_weight=1.0."
)

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# command:
# sbatch -J "timefirst32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_config.py slurm/main.slurm
