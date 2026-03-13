~~~Git commands~~~

git status
git pull
git fetch --all
git switch <branch>
git switch -c <new-branch>
git add <file-or-dir>
git add file1 file2 file3  # mehrere Dateien auf einmal
git add -A
git commit -m "message"
git log --oneline --decorate -n 20
git diff
git push


~~~Slurm (sbatch)~~~

Simulation (heat_sim_initial):

sbatch -J "new_sim_cpu" slurm/new_heat_sim_class_cpu.slurm
sbatch -J "new_sim_cpu" --export=ALL,SIM_CONFIG=configs/new_sim_config.py slurm/new_heat_sim_class_cpu.slurm

Training: 

sbatch -J "dyn" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_config.py slurm/main.slurm
sbatch -J "timefirst" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_config.py slurm/main.slurm
sbatch -J "static" --export=ALL,TRAIN_CONFIG=configs/picnn_static_config.py slurm/main.slurm

sbatch -J "pinn" slurm/pinn.slurm


Testsetgeneration:

sbatch -J "gen20s" \
  --export=ALL,OUT_ROOT=./data/testset_20s,FIRES_MIN=1,FIRES_MAX=10,EXPERIMENTS_PER_FIRE=1,SIM_TIME_SECONDS=20,NT=20000,SAVE_EVERY=100,NORMALIZE=1 \
  slurm/generate_testset.slurm

Predictions:

sbatch -J "pred_runid" \
  --export=ALL,RUN_ID=dynamic_20260302-162917_6b06f9,TESTSET_NAME=testset_20s \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_modeldir" \
  --export=ALL,MODEL_DIR=./runs/dynamic/PITCNN_dynamic_f32_lr=0.0001_V0.3/dynamic_20260302-162917_6b06f9,TESTSET_NAME=testset_20s \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_custom_data_root" \
  --export=ALL,MODEL_DIR=./runs/dynamic/PITCNN_dynamic_f32_lr=0.0001_V0.3/dynamic_20260302-162917_6b06f9,DATA_ROOT=./data,TESTSET_NAME=testset_20s,TIME_STRIDE=10 \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_smollds_161748" \
  --export=ALL,RUN_ID=dynamic_20260227-161748_d1a417,TESTSET_NAME=TESTSET_DT=0.001_T=15.0 \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_smollds_184422" \
  --export=ALL,RUN_ID=dynamic_20260227-184422_1b1203,TESTSET_NAME=TESTSET_DT=0.001_T=15.0 \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_little_161748" \
  --export=ALL,RUN_ID=dynamic_20260227-161748_d1a417,TESTSET_NAME=testsetLITTLE \
  slurm/predict_dynamic_testset.slurm

sbatch -J "pred_little_184422" \
  --export=ALL,RUN_ID=dynamic_20260227-184422_1b1203,TESTSET_NAME=testsetLITTLE \
  slurm/predict_dynamic_testset.slurm

Visualisierung:

sbatch -J "slice15" \
  --export=ALL,BASE_PATH=./data/testdummy,EXPERIMENT=experiment_15_20260223_140219,NORMALIZED=0 \
  slurm/plot_room_slice_cpu.slurm

sbatch -J "viz_newsim3d" --export=ALL,ALLOW_ALL_EXPERIMENTS=1,OUT_ROOT=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/plots/new_detailed_heat_sim_f64_3d slurm/visualize_heatvid_3d.slurm

sbatch -J "viz_lastframe" --export=ALL,OUT_ROOT=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/plots/new_detailed_heat_sim_f64_3d,LAST_FRAME_ONLY=1 slurm/visualize_heatvid_3d.slurm
sbatch -J "viz_25_lastframe" --export=ALL,EXPERIMENT=experiment_25_20260223_145422,OUT_ROOT=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/plots/new_detailed_heat_sim_f64_3d,LAST_FRAME_ONLY=1,THRESHOLD_QUANTILE=0.50,HEAT_THRESHOLD=0.01,VIZ_VMAX=5000 slurm/visualize_heatvid_3d.slurm  # output: .../plots/new_detailed_heat_sim_f64_3d/experiment_25_.../last_frame.png
sbatch -J "viz_quick_25" --export=ALL,EXPERIMENT=experiment_25_20260223_145422,OUT_ROOT=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/plots/new_detailed_heat_sim_f64_3d slurm/visualize_heatvid_3d_quick.slurm

sbatch -J "viz9_voxel_full" \
  --export=ALL,EXPERIMENT=experiment_9_20260220_134208_000,STYLE=voxel,SOURCE_MARKER_MODE=none,DOWNSAMPLE=1,THRESHOLD_QUANTILE=0.60,HEAT_THRESHOLD=0.04,FRAME_STRIDE=1,MAX_FRAMES=0,SAVE_FRAMES_EVERY=10 \
  slurm/visualize_heatvid_3d.slurm


sbatch -J "viz_25_video_opaque" --export=ALL,EXPERIMENT=experiment_25_20260223_145422,OUT_ROOT=/beegfs/home/l/lieberta/projects/physics-enhanced-cnn/plots/new_detailed_heat_sim_f64_3d,LAST_FRAME_ONLY=1,THRESHOLD_QUANTILE=0.50,HEAT_THRESHOLD=0.01,VIZ_VMAX=5000,VOXEL_ALPHA_BASE=0.18,VOXEL_ALPHA_SCALE=0.82,VOXEL_ALPHA_GAMMA=0.55 slurm/visualize_heatvid_3d.slurm

General:

squeue -u $USER
scancel <jobid>
scontrol show job <jobid> | grep -i Command
squeue -p small_gpu -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" # check small gpu occupation

# Note: without --export TRAIN_CONFIG, main.py uses train_config.py by default.



~~~Linux~~~

conda activate /beegfs/home/l/lieberta/venv/penn-venv/  #activates conda env
tree -L 2   #shows folder tree of length 2
