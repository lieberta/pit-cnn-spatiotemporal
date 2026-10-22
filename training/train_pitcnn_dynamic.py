import math
import os
import time
import json
import random

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import CombinedLoss_dynamic, physics_residual
from data import prepare_experiment_split
from .train_utils import atomic_torch_save, EpochBatchSampler, TrainingBudget, loss_gradient_scales, seed_everything, seed_worker, git_provenance, state_dict_hash, append_metrics_row, fallback_loss_history, load_checkpoint, load_loss_history_from_metrics, accumulate_training_duration, update_run_config

def _normalized_model_times(step_idx, dt, prediction_horizon, dtype):
    """Convert integer steps to normalized model times, keeping dt in seconds."""
    t_seconds = step_idx.to(dtype=dtype) * dt
    # Subtract before casting so the first past step is exactly zero.
    t_past_seconds = (step_idx - 1).to(dtype=dtype) * dt
    return t_seconds / prediction_horizon, t_past_seconds / prediction_horizon


def _anchor_initial_state(input_t0, output_past, step_idx):
    """Use the known temperature, excluding source channels, at step zero."""
    is_first_step = (step_idx == 1).view(-1, 1, 1, 1, 1)
    return torch.where(is_first_step, input_t0[:, :1, ...], output_past)


class BaseModel_dynamic(nn.Module):
    def __init__(self):
        super(BaseModel_dynamic, self).__init__()

    def train_model(
        self,
        loss_weight_schedule,
        dataset,
        num_epochs,
        batch_size,
        learning_rate,
        model_name,
        save_path,
        run_id=None,
        channels=None,
        seed=None,
        resume_checkpoint_path=None,
        physics_residual_mode=None,
        split_manifest_path=None,
        split_seed=42,
        training_source_counts=None,
        loader_workers=4,
        mmap_cache_dir=None,
        max_runtime_seconds=None,
    ):
        tic = time.perf_counter()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_dtype = torch.get_default_dtype()
        print("Device = " + device)
        self.to(device=device, dtype=train_dtype)
        
        train_losses = []
        val_losses = []

        if seed is None:
            raise ValueError("An explicit model seed is required for reproducible training.")
        seed = int(seed)
        seed_everything(seed)
        model_dir = os.path.join(save_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, "config.json")
        subsets, manifest = prepare_experiment_split(
            dataset, split_manifest_path, split_seed, training_source_counts
        )
        protocol = {
            "version": 2, "split": manifest,
            "time_units": {"model": "t_seconds / prediction_horizon", "derivative_dt": "seconds"},
            "first_step": "known_temperature_channel_0", "checkpoint_selection": "validation_mse",
            "diagnostic_residual": "model_past_explicit_euler_interior",
            "physics": {key: float(getattr(dataset, key)) for key in (
                "dt", "prediction_horizon", "alpha", "dx", "dy", "dz", "rho", "cp", "ambient_temp"
            )},
        }
        if resume_checkpoint_path:
            source_manifest = os.path.join(os.path.dirname(resume_checkpoint_path), "protocol.json")
            if not os.path.isfile(source_manifest):
                raise ValueError("Legacy checkpoint has no experiment-split protocol. Train a new baseline first.")
            with open(source_manifest) as f:
                if json.load(f) != protocol:
                    raise ValueError("Checkpoint split, normalization or time protocol differs from this run.")
        with open(os.path.join(model_dir, "protocol.json"), "w") as f:
            json.dump(protocol, f, indent=2)
        with open(os.path.join(model_dir, "experiment_split.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self.experiment_protocol = protocol
        self.experiment_seed = seed
        train_generator = torch.Generator()
        val_generator = torch.Generator().manual_seed(seed)
        if mmap_cache_dir:
            dataset.prepare_mmap_cache(mmap_cache_dir, set(manifest['train'] + manifest['validation']))
        batch_sampler = EpochBatchSampler(len(subsets['train']), batch_size, seed)
        loader_options = dict(num_workers=loader_workers, pin_memory=True,
                              worker_init_fn=seed_worker, persistent_workers=loader_workers > 0)
        train_loader = DataLoader(subsets['train'], batch_sampler=batch_sampler,
                                  generator=train_generator, **loader_options)
        val_loader = DataLoader(subsets['validation'], shuffle=False, batch_size=batch_size,
                                generator=val_generator, **loader_options)
        print("Experiment split: " + ", ".join(f"{k}={len(manifest[k])}" for k in ('train','validation','test')))

        if not loss_weight_schedule:
            raise ValueError("loss_weight_schedule must be provided and non-empty.")
        if physics_residual_mode is None:
            raise ValueError("physics_residual_mode must be provided explicitly for dynamic training.")

        # Dataset metadata is the single source of truth for all physical and
        # normalization constants used by the loss. No default fallbacks are
        # allowed because silent mismatches are hard to debug later.
        required_dataset_attributes = (
            "min_temp",
            "max_temp",
            "dt",
            "prediction_horizon",
            "alpha",
            "dx",
            "dy",
            "dz",
            "ambient_temp",
        )
        missing_attributes = [name for name in required_dataset_attributes if not hasattr(dataset, name)]
        if missing_attributes:
            raise ValueError(
                f"Dataset is missing required attributes for dynamic training: {missing_attributes}"
            )

        min_temp = float(dataset.min_temp)
        max_temp = float(dataset.max_temp)
        dt = float(dataset.dt)
        prediction_horizon = float(dataset.prediction_horizon)
        alpha = float(dataset.alpha)
        dx = float(dataset.dx)
        dy = float(dataset.dy)
        dz = float(dataset.dz)
        ambient_temp = float(dataset.ambient_temp)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Invalid dataset dt: {dt}")
        if not math.isfinite(prediction_horizon) or prediction_horizon <= 0.0:
            raise ValueError(f"Invalid dataset prediction_horizon: {prediction_horizon}")

        initial_lp_weight = float(loss_weight_schedule[0]["lp_weight"])
        initial_mse_weight = float(loss_weight_schedule[0]["mse_weight"])
        criterion = CombinedLoss_dynamic(
            lp_weight=initial_lp_weight,
            mse_weight=initial_mse_weight,
            device=device,
            alpha=alpha,
            min_temp=min_temp,
            max_temp=max_temp,
            dx=dx,
            dy=dy,
            dz=dz,
            dt=dt,
            ambient_temp=ambient_temp,
            source_region_weight=1.0,
            physics_residual_mode=physics_residual_mode,
        ).to(device=device, dtype=train_dtype)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_val_mse = None
        best_val_physics = None
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    run_config = json.load(f)
                if run_config.get("best_val_mse_loss") is not None:
                    best_val_mse = float(run_config["best_val_mse_loss"])
                if run_config.get("best_val_physics_loss") is not None:
                    best_val_physics = float(run_config["best_val_physics_loss"])
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                best_val_mse = None
                best_val_physics = None

        start_epoch = 0
        resume_progress = {}
        if resume_checkpoint_path:
            checkpoint_metadata = torch.load(resume_checkpoint_path, map_location="cpu")
            if checkpoint_metadata.get("protocol") != protocol:
                raise ValueError("Checkpoint itself has incompatible experiment provenance.")
            if checkpoint_metadata.get("seed") != seed:
                raise ValueError("Continuation must use the same seed as its source checkpoint.")
            resume_progress = checkpoint_metadata.get("progress") or {}
            del checkpoint_metadata
            start_epoch = load_checkpoint(self, optimizer, resume_checkpoint_path, device)
            print(f"Resuming from checkpoint: {resume_checkpoint_path} (start_epoch={start_epoch})")

        if not resume_checkpoint_path or not os.path.exists(os.path.join(model_dir, "initial_state.pth")):
            torch.save({"model_state_dict": self.state_dict(), "seed": seed},
                       os.path.join(model_dir, "initial_state.pth"))
        initial_model_state = torch.load(os.path.join(model_dir, "initial_state.pth"), map_location="cpu")
        update_run_config(config_path, {
            "git": git_provenance(), "protocol": protocol,
            "training_complete": False, "loader_workers": loader_workers,
            "loader": "job_local_mmap" if mmap_cache_dir else "zarr_lru",
            "sampler": "seed_plus_epoch_randperm_v1",
            "loss_weight_schedule": loss_weight_schedule,
            "physics_residual_mode": physics_residual_mode,
            "epochs": num_epochs, "batch": batch_size, "lr": learning_rate,
            "dataset_metadata_sha256": manifest['metadata_sha256'],
            "normalization": manifest['normalization'],
            "initial_state_sha256": state_dict_hash(initial_model_state['model_state_dict']), "seed": seed,
            "train_samples": len(subsets['train']), "validation_samples": len(subsets['validation']),
            "test_samples": len(subsets['test']),
        })
        lp_by_epoch = []
        mse_by_epoch = []
        for phase in loss_weight_schedule:
            phase_epochs = int(phase["epochs"])
            lp_by_epoch.extend([float(phase["lp_weight"])] * phase_epochs)
            mse_by_epoch.extend([float(phase["mse_weight"])] * phase_epochs)
        if len(lp_by_epoch) != num_epochs:
            raise ValueError(
                f"loss_weight_schedule must cover exactly num_epochs={num_epochs}, got {len(lp_by_epoch)} epochs."
            )

        remaining_budget = (max(0, max_runtime_seconds - (time.perf_counter() - tic))
                            if max_runtime_seconds is not None else None)
        with TrainingBudget(remaining_budget) as control:
            for epoch in range(start_epoch, num_epochs):
                train_generator.manual_seed(seed + epoch)
                batch_sampler.epoch = epoch
                batch_sampler.start_batch = int(resume_progress.get("next_batch", 0)) if epoch == start_epoch else 0
                criterion.lp_weight = float(lp_by_epoch[epoch])
                criterion.mse_weight = float(mse_by_epoch[epoch])
                train_loss = 0.0
                train_mse = 0.0
                train_physics = 0.0
                train_objective_physics = 0.0
                val_loss = 0.0
                val_mse = 0.0
                val_physics = 0.0
                val_objective_physics = 0.0
                train_samples = val_samples = 0
                val_ic_squared_error = 0.0
                val_ic_elements = 0
                val_ic_max_abs_error = 0.0

                if epoch == start_epoch and resume_progress:
                    train_loss = resume_progress['train_loss']
                    train_mse = resume_progress['train_mse']
                    train_physics = resume_progress['train_physics']
                    train_objective_physics = resume_progress['train_objective_physics']
                    train_samples = resume_progress['train_samples']
                next_batch = batch_sampler.start_batch
                last_progress_save = time.monotonic()

                def save_progress():
                    self.save_model(epoch, model_name, save_path, optimizer, progress=dict(
                        next_batch=next_batch, train_loss=train_loss, train_mse=train_mse,
                        train_physics=train_physics, train_objective_physics=train_objective_physics,
                        train_samples=train_samples, sampler='seed_plus_epoch_randperm_v1'))

                def pause_if_needed():
                    if control.expired():
                        save_progress()
                        update_run_config(config_path, {"training_complete": False,
                            "pause_epoch": epoch, "pause_next_batch": next_batch})
                        accumulate_training_duration(config_path, time.perf_counter() - tic)
                        print(f'Saved resumable checkpoint: epoch={epoch}, next_batch={next_batch}', flush=True)
                        raise SystemExit(75)

                pause_if_needed()
                self.train()
                loop = tqdm(train_loader, total=len(train_loader), leave=True)
                for i, (input_tuple, target) in enumerate(loop):
                    # Dynamic batches carry T0, target past state, and the explicit
                    # source representation in addition to the integer target step. The
                    # dataloader intentionally provides both past and future ground
                    # truth frames so different physics residual variants can be
                    # tested on the same sample set.
                    input_t0, step_idx, target_past, target_future, source_mask, source_field = input_tuple
                    input_t0 = input_t0.to(device, dtype=train_dtype)
                    step_idx = step_idx.to(device)
                    tau, tau_past = _normalized_model_times(
                        step_idx, dt, prediction_horizon, train_dtype
                    )
                    target_past = target_past.to(device, dtype=train_dtype)
                    target_future = target_future.to(device, dtype=train_dtype)
                    source_mask = source_mask.to(device, dtype=train_dtype)
                    source_field = source_field.to(device, dtype=train_dtype)
                    target = target.to(device, dtype=train_dtype)

                    output = self(input_t0, tau)

                    # Measure the same model-based residual for every variant.
                    # Only model-based objectives need gradients through the past.
                    if criterion.physics_residual_mode in {"model_past", "soft_physics"}:
                        diagnostic_past = self(input_t0, tau_past)
                    else:
                        with torch.no_grad():
                            diagnostic_past = self(input_t0, tau_past)
                    diagnostic_past = _anchor_initial_state(input_t0, diagnostic_past, step_idx)
                    output_past = diagnostic_past if criterion.physics_residual_mode in {
                        "model_past", "soft_physics"
                    } else None

                    loss, mse_loss, physics_loss = criterion.compute_components(
                        input_t0=input_t0,
                        output=output,
                        output_past=output_past,
                        target_past=target_past,
                        target=target,
                        target_future=target_future,
                        source_mask=source_mask,
                        source_field=source_field,
                    )

                    common_residual = physics_residual(
                        output, diagnostic_past, dt, alpha, criterion.laplacian_layer, source_field
                    )
                    if batch_sampler.start_batch + i == 0:
                        # Probe the selected regularizer even for the MSE control.
                        _, _, measured_objective = criterion.compute_components(
                            input_t0=input_t0, output=output, output_past=output_past,
                            target_past=target_past, target=target, target_future=target_future,
                            source_mask=source_mask, source_field=source_field, measure_physics=True
                        )
                        scales = loss_gradient_scales(self, mse_loss, measured_objective, criterion.lp_weight)
                        scales.update(common_physics_loss=common_residual.detach().square().mean().item())
                        scale_row = dict(run_id=run_id or model_name, epoch=epoch + 1,
                                         samples=target.shape[0], lp_weight=criterion.lp_weight,
                                         mse_weight=criterion.mse_weight, **scales)
                        append_metrics_row(os.path.join(model_dir, "loss_gradient_scales.csv"),
                                           list(scale_row), scale_row)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_samples = target.shape[0]
                    train_samples += batch_samples
                    train_loss += loss.item() * batch_samples
                    train_mse += mse_loss.item() * batch_samples
                    train_physics += common_residual.detach().square().mean().item() * batch_samples
                    train_objective_physics += physics_loss.item() * batch_samples
                    next_batch = batch_sampler.start_batch + i + 1
                    if time.monotonic() - last_progress_save >= 900:
                        save_progress()
                        last_progress_save = time.monotonic()
                    pause_if_needed()

                    loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                    loop.set_postfix(trainloss=train_loss / train_samples)

                avg_train_loss = train_loss / train_samples
                train_losses.append(avg_train_loss)

                self.eval()
                with torch.no_grad():
                    for input_tuple, target in val_loader:
                        pause_if_needed()
                        input_t0, step_idx, target_past, target_future, source_mask, source_field = input_tuple
                        input_t0 = input_t0.to(device, dtype=train_dtype)
                        step_idx = step_idx.to(device)
                        tau, tau_past = _normalized_model_times(
                            step_idx, dt, prediction_horizon, train_dtype
                        )
                        target_past = target_past.to(device, dtype=train_dtype)
                        target_future = target_future.to(device, dtype=train_dtype)
                        source_mask = source_mask.to(device, dtype=train_dtype)
                        source_field = source_field.to(device, dtype=train_dtype)
                        target = target.to(device, dtype=train_dtype)
                        output = self(input_t0, tau)
                        diagnostic_past = _anchor_initial_state(
                            input_t0, self(input_t0, tau_past), step_idx
                        )
                        output_past = diagnostic_past if criterion.physics_residual_mode in {
                            "model_past", "soft_physics"
                        } else None
                        loss, mse_loss, physics_loss = criterion.compute_components(
                            input_t0=input_t0,
                            output=output,
                            output_past=output_past,
                            target_past=target_past,
                            target=target,
                            target_future=target_future,
                            source_mask=source_mask,
                            source_field=source_field,
                        )
                        common_residual = physics_residual(
                            output, diagnostic_past, dt, alpha, criterion.laplacian_layer, source_field
                        )
                        batch_samples = target.shape[0]
                        val_samples += batch_samples
                        val_loss += loss.item() * batch_samples
                        val_mse += mse_loss.item() * batch_samples
                        val_physics += common_residual.square().mean().item() * batch_samples
                        val_objective_physics += physics_loss.item() * batch_samples

                        # Diagnose the model's own t=0 output independently of the
                        # anchored residual, for every mode including MSE-only.
                        output_initial = self(input_t0, torch.zeros_like(tau))
                        initial_error = output_initial - input_t0[:, :1, ...]
                        val_ic_squared_error += initial_error.square().sum().item()
                        val_ic_elements += initial_error.numel()
                        val_ic_max_abs_error = max(
                            val_ic_max_abs_error, initial_error.abs().max().item()
                        )

                avg_val_loss = val_loss / val_samples
                val_losses.append(avg_val_loss)
                avg_train_mse = train_mse / train_samples
                avg_train_physics = train_physics / train_samples
                avg_val_mse = val_mse / val_samples
                avg_val_physics = val_physics / val_samples

                os.makedirs(model_dir, exist_ok=True)
                metrics_path = os.path.join(model_dir, "metrics.csv")
                components_path = os.path.join(model_dir, "loss_components.csv")
                header = ["run_id", "epoch", "train_loss", "val_loss", "lr", "lp_weight", "mse_weight", "channels", "batch", "seed"]
                components_header = [
                    "run_id",
                    "epoch",
                    "train_mse_loss",
                    "train_physics_loss",
                    "val_mse_loss",
                    "val_physics_loss",
                    "lp_weight",
                    "mse_weight",
                    "physics_residual_mode",
                    "val_objective_physics_loss",
                    "train_objective_physics_loss",
                ]
                row = {
                    "run_id": run_id or model_name,
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "lr": learning_rate,
                    "lp_weight": float(criterion.lp_weight),
                    "mse_weight": float(criterion.mse_weight),
                    "channels": channels,
                    "batch": batch_size,
                    "seed": seed,
                }
                components_row = {
                    "run_id": run_id or model_name,
                    "epoch": epoch + 1,
                    "train_mse_loss": avg_train_mse,
                    "train_physics_loss": avg_train_physics,
                    "val_mse_loss": avg_val_mse,
                    "val_physics_loss": avg_val_physics,
                    "lp_weight": float(criterion.lp_weight),
                    "mse_weight": float(criterion.mse_weight),
                    "physics_residual_mode": criterion.physics_residual_mode,
                    "val_objective_physics_loss": val_objective_physics / val_samples if criterion.lp_weight != 0 else None,
                    "train_objective_physics_loss": train_objective_physics / train_samples if criterion.lp_weight != 0 else None,
                }
                append_metrics_row(metrics_path, header, row)
                append_metrics_row(components_path, components_header, components_row)
                # Keep diagnostics separate so resuming older runs does not change
                # the column layout of their existing metrics CSVs.
                val_ic_mse = val_ic_squared_error / val_ic_elements
                append_metrics_row(
                    os.path.join(model_dir, "initial_condition_metrics.csv"),
                    ["run_id", "epoch", "val_ic_mse", "val_ic_max_abs_error"],
                    {
                        "run_id": run_id or model_name,
                        "epoch": epoch + 1,
                        "val_ic_mse": val_ic_mse,
                        "val_ic_max_abs_error": val_ic_max_abs_error,
                    },
                )
                print(
                    f"Initial condition (validation, normalized): "
                    f"MSE={val_ic_mse:.6e}, max_abs={val_ic_max_abs_error:.6e}"
                )
                is_best = best_val_mse is None or avg_val_mse < best_val_mse
                best_val_mse = avg_val_mse if is_best else best_val_mse
                best_val_physics = avg_val_physics if best_val_physics is None else min(best_val_physics, avg_val_physics)
                update_run_config(
                    config_path,
                    {
                        "best_val_mse_loss": float(best_val_mse),
                        "best_val_physics_loss": float(best_val_physics),
                    },
                )
                self.save_model(epoch, model_name, save_path, optimizer, is_best=is_best,
                                validation_mse=avg_val_mse)

        update_run_config(config_path, {"training_complete": True})
        metrics_path = os.path.join(save_path, model_name, "metrics.csv")
        epochs_all, train_losses_all, val_losses_all = load_loss_history_from_metrics(metrics_path)
        if not epochs_all:
            epochs_all, train_losses_all, val_losses_all = fallback_loss_history(num_epochs, train_losses, val_losses)

        self.save_loss_plot(model_name, epochs_all, train_losses_all, val_losses_all, save_path)
        proc_time = self.save_proc_time(model_name, tic, save_path)
        accumulate_training_duration(config_path, proc_time)

    def save_model(self, epoch, model_name, save_path, optimizer, is_best=False, validation_mse=None, progress=None):
        model_dir = os.path.join(save_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"epoch_{epoch}.pth")

        checkpoint = {
            "epoch": epoch, "epoch_complete": progress is None, "progress": progress,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_mse": validation_mse,
            "protocol": self.experiment_protocol,
            "seed": self.experiment_seed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": (lambda state: (state[0], state[1].tolist(), *state[2:]))(np.random.get_state()),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        if progress is None:
            atomic_torch_save(checkpoint, model_path)

        latest_model_path = os.path.join(model_dir, f"{model_name}.pth")
        atomic_torch_save(checkpoint, latest_model_path)
        if is_best:
            atomic_torch_save(checkpoint, os.path.join(model_dir, "best.pth"))

    def save_loss_plot(self, model_name, epochs, train_losses, val_losses, save_path):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Train Loss", color="blue")
        plt.plot(epochs, val_losses, label="Validation Loss", color="red")
        plt.title("Training and Validation Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        model_dir = os.path.join(save_path, model_name)
        losses_plot_filename = os.path.join(model_dir, f"losses_plot_{model_name}.png")
        plt.savefig(losses_plot_filename)
        plt.show()

    def save_proc_time(self, model_name, start_time, save_path):
        end_time = time.perf_counter()
        proc_time = end_time - start_time
        model_dir = os.path.join(save_path, model_name)
        proc_time_filename = os.path.join(model_dir, f"proc_time_{model_name}.txt")
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(proc_time))
        with open(proc_time_filename, "w") as file:
            file.write(f"Training process duration: {formatted_time}")
        return proc_time
