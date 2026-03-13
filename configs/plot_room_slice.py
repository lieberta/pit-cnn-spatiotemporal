base_path = "./data/new_detailed_heat_sim_f64"
experiment = "experiment_15_20260223_140219"
normalized = False
step_every = 10
vmax_clip = 500.0
output_subdir = "plots_room_slice"

sbatch -J "plotting" \
  --export=ALL,BASE_PATH=./data/new_detailed_heat_sim_f64_TESTFOLDER,EXPERIMENT=experiment_25_20260223_145422,NORMALIZED=0,STEP_EVERY=10,VMAX_CLIP=500.0 \
  slurm/plot_room_slice_cpu.slurm

# /beegfs/home/l/lieberta/projects/physics-enhanced-cnn/data/laplace_convolution_zarr/experiment_3_20240223_180822

