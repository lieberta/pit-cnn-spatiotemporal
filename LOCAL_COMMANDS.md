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

Training: 

sbatch -J "dyn64" --export=ALL,TRAIN_CONFIG=configs/pitcnn_dynamic_config.py slurm/main.slurm
sbatch -J "timefirst64" --export=ALL,TRAIN_CONFIG=configs/pitcnn_timefirst_config.py slurm/main.slurm
sbatch -J "static64" --export=ALL,TRAIN_CONFIG=configs/picnn_static_config.py slurm/main.slurm

sbatch -J "pinn" slurm/pinn.slurm


Testsetgeneration:

sbatch -J "gen20s" \
  --export=ALL,OUT_ROOT=./data/testset_20s,FIRES_MIN=1,FIRES_MAX=10,EXPERIMENTS_PER_FIRE=1,SIM_TIME_SECONDS=20,NT=20000,SAVE_EVERY=100,NORMALIZE=1 \
  slurm/generate_testset.slurm

Visualisierung:

sbatch -J "viz20s3d" slurm/visualize_testset_3d.slurm

sbatch -J "viz9_voxel_full" \
  --export=ALL,EXPERIMENT=experiment_9_20260220_134208_000,STYLE=voxel,SOURCE_MARKER_MODE=none,DOWNSAMPLE=1,THRESHOLD_QUANTILE=0.60,HEAT_THRESHOLD=0.04,FRAME_STRIDE=1,MAX_FRAMES=0,SAVE_FRAMES_EVERY=10 \
  slurm/visualize_testset_3d.slurm

General:

squeue -u $USER
scancel <jobid>
scontrol show job <jobid> | grep -i Command
squeue -p small_gpu -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" # check small gpu occupation

# Note: without --export TRAIN_CONFIG, main.py uses train_config.py by default.



~~~Linux~~~

conda activate /beegfs/home/l/lieberta/venv/penn-venv/  #activates conda env
tree -L 2   #shows folder tree of length 2

