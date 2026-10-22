"""Controlled A–E comparison; invoked by scripts/physics_comparison.py."""
import math
import os

import torch

from configs.reproducibility import seed, split_seed, training_source_counts

variant = os.environ.get("PITCNN_VARIANT", "A")
weights = {"A": 0.0, "B": 0.001, "C": 0.01, "D": 0.1}
if variant not in {*weights, "E"}:
    raise ValueError("PITCNN_VARIANT must be A, B, C, D or E.")
weight = float(os.environ["PITCNN_SOFT_WEIGHT"]) if variant == "E" else weights[variant]
if not math.isfinite(weight) or weight < 0:
    raise ValueError("Physics weight must be finite and nonnegative.")
physics_residual_mode = "soft_physics" if variant == "E" else "model_past"
train_dtype = torch.float32
epochs = 30
lr = 0.001
batch = 128
channels = 16
loss_weight_schedule = [dict(epochs=epochs, lp_weight=weight, mse_weight=1.0)]
model_class_name = "PITCNN_dynamic"
model_name = f"PITCNN_STEEL_COMPARISON_{variant}_seed{seed}_lp{weight:.8g}"
run_comment = "Controlled A–E comparison: same split, initialization seed and 30 epochs."
data_path = "./data/steel_train_medium/"
data_modulo = 1
data_max_experiments = None
data_experiment_offset = 0
split_manifest_path = "splits/steel_train_medium.json"
resume_run_ids_dynamic = [os.environ["PITCNN_RESUME_RUN_ID"]] if os.environ.get("PITCNN_RESUME_RUN_ID") else []
auto_collect_dynamic = False
