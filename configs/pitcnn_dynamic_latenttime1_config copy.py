import torch

# This file contains training configuration parameters for the model PITCNN_dynamic_latenttime1
train_dtype = torch.float32
epochs = 20
lr = 0.001
loss_weight_schedule = [
    {"epochs": 20, "lp_weight": 1.0, "mse_weight": 1.0},
]
model_class_name = "PITCNN_dynamic_latenttime1"
model_name = model_class_name + "_f32_lp1.0_mse1.0_20ep_V0.3"
# Data subset controls for dataset-size sweeps:
data_path = "./data/laplace_convolution_zarr/"
run_dataset_name = "laplace_convolution_zarr"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    "20 epochs with constant lp_weight=1.0 and mse_weight=1.0 on laplace_convolution_zarr."
)
resume_run_ids_dynamic = []
auto_collect_dynamic = False


# command:
# sbatch -J "latenttime1_32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_latenttime1_config.py slurm/main.slurm
