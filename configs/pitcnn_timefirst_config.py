import torch

epochs = 10
lr = 0.001
lp_weight_list = [1.0]
mse_weight = 1.0
model_class_name = "PITCNN_dynamic_timefirst"
model_name = model_class_name + "_f32_V0.3"
# Data subset controls for dataset-size sweeps:
data_path = "./data/new_detailed_heat_sim_f64/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
run_comment = (
    "V0.2: physics loss now enforced on interior cells only; Laplacian kernel scaled like simulation "
    "(dx,dy,dz from L/N); fixed z=0 boundary assignment in PITCNN_dynamic and PITCNN_dynamic_latenttime1 "
    "to copy the full boundary plane."
)

resume_run_ids_dynamic = []
auto_collect_dynamic = False

# command:
# sbatch -J "timefirst32" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_config.py slurm/main.slurm
