import torch

train_dtype = torch.float32
epochs = 100
lr = 0.001

lp_weight_list = [0.0]
mse_weight = 1.0
loss_weight_schedule = [
    {"epochs": 100, "lp_weight": 0.0, "mse_weight": 1.0},
]

model_class_name = "PITCNN_dynamic"
model_name = "PITCNN_dynamic_MSEONLY_quick10_mod10_trainset10s"

data_path = "./data/TRAIN_DATASET_10s_dt0.01/"
data_modulo = 10
data_max_experiments = None
data_experiment_offset = 0

run_comment = "Quick MSE-only test on TRAIN_DATASET_10s_dt0.01 (epochs=100, modulo=10, lp=0.0, mse=1.0)."

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# Local:
# python main.py --config configs/pitcnn_dynamic_mse_only_quick10_trainset.py
#
# Slurm:
# sbatch -J "mse10_mod10" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_mse_only_quick10_trainset.py slurm/main.slurm
