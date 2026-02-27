import torch

# This file contains training configuration parameters for the model PITCNN_dynamic_latenttime1
epochs = 50
lr = 0.001
a_list = [1.0]
mse_weight = 0.0  # set to 0.0 for physics-only training (MSE disabled)
model_class_name = "PITCNN_dynamic_latenttime1"
model_name = model_class_name + "_physics_only_f32_V0.3"
run_comment = (
    "50 epochs without mse only physics"
)
resume_run_ids_dynamic = []
auto_collect_dynamic = False


# command:
# sbatch -J "latenttime1_32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_latenttime1_config.py slurm/main.slurm
