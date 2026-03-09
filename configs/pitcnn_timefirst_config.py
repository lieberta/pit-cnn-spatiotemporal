import torch

train_dtype = torch.float32
epochs = 100
lr = 0.001
loss_weight_schedule = [
    {"epochs": 100, "lp_weight": 1.0, "mse_weight": 1.0},
]
model_class_name = "PITCNN_dynamic_timefirst"
model_name = model_class_name + "_BOTH_100ep_V0.3"
# Data subset controls for dataset-size sweeps:
data_path = "./data/TRAIN_DATASET_10s_dt0.01/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    ""
)

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# command:
# sbatch -J "timefirst32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_config.py slurm/main.slurm