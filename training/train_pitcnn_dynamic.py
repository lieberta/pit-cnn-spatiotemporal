import math
import os
import time
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import CombinedLoss_dynamic
from .train_utils import append_metrics_row, fallback_loss_history, load_checkpoint, load_loss_history_from_metrics, accumulate_training_duration, update_run_config

SIM_TOTAL_SECONDS = 10.0
SIM_STEPS_PER_SECOND = 1000.0
SECONDS_PER_STEP = 1.0 / SIM_STEPS_PER_SECOND


class BaseModel_dynamic(nn.Module):
    def __init__(self):
        super(BaseModel_dynamic, self).__init__()

    def train_model(
        self,
        lp_weight,
        mse_weight,
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
        loss_weight_schedule=None,
    ):
        tic = time.perf_counter()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_dtype = torch.get_default_dtype()
        print("Device = " + device)
        self.to(device=device, dtype=train_dtype)
        
        train_losses = []
        val_losses = []

        shuffle = True
        pin_memory = True
        num_workers = 16

        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [math.ceil(len(dataset) * 0.8), math.floor(len(dataset) * 0.2)],
        )
        train_loader = DataLoader(
            dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            dataset=val_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )

        min_temp = getattr(dataset, "min_temp", 20.0)
        max_temp = getattr(dataset, "max_temp", 27373.34765625)
        criterion = CombinedLoss_dynamic(
            lp_weight=lp_weight,
            mse_weight=mse_weight,
            device=device,
            min_temp=min_temp,
            max_temp=max_temp,
        ).to(device=device, dtype=train_dtype)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        model_dir = os.path.join(save_path, model_name)
        config_path = os.path.join(model_dir, "config.json")

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
        if resume_checkpoint_path:
            start_epoch = load_checkpoint(self, optimizer, resume_checkpoint_path, device)
            print(f"Resuming from checkpoint: {resume_checkpoint_path} (start_epoch={start_epoch})")

        if loss_weight_schedule:
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
        else:
            lp_by_epoch = [float(lp_weight)] * num_epochs
            mse_by_epoch = [float(mse_weight)] * num_epochs

        for epoch in range(start_epoch, num_epochs):
            criterion.lp_weight = float(lp_by_epoch[epoch])
            criterion.mse_weight = float(mse_by_epoch[epoch])
            train_loss = 0.0
            train_mse = 0.0
            train_physics = 0.0
            val_loss = 0.0
            val_mse = 0.0
            val_physics = 0.0

            self.train()
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for i, (input_tuple, target) in enumerate(loop):
                input, t = input_tuple # input shape: (B, C_in, D, H, W), t shape: (B,1)
                input = input.to(device, dtype=train_dtype)
                t = t.to(device, dtype=train_dtype)
                target = target.to(device, dtype=train_dtype)

                # creates a tensor for delta_t to match tensor t
                delta_t = torch.full_like(t, SECONDS_PER_STEP, device=device, dtype=train_dtype)
                t_past = t - delta_t
                output = self(input, t)
                first_step_mask = torch.isclose(t, delta_t, rtol=0.0, atol=1e-6)  # creates bool-mask per batch-element (shape: (B,1)), True where t ≈ delta_t, so where t_past would be ≈ 0 (with a tolerance of 1e-6 to account for floating point precision issues)

                output_past = self(input, t_past) # computes output_past for all samples

                # For samples where t_past ≈ 0, we set output_past := input, because for the first simulation step the "past output" is just the initial condition (input).
                if first_step_mask.any():  # check if there are any samples in the batch where t_past ≈ 0
                    # Reshape mask from (B,1) to (B,1,1,1,1), so one bool controls one full sample volume.
                    mask_5d = first_step_mask.view(t.shape[0], 1, 1, 1, 1)
                    # torch.where picks input where mask is True, otherwise keeps model output_past.
                    output_past = torch.where(mask_5d, input, output_past)

                loss, mse_loss, physics_loss = criterion.compute_components(
                    input, output, output_past, t, t_past, target
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_mse += mse_loss.item()
                train_physics += physics_loss.item()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(trainloss=train_loss / (i + 1))

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            with torch.no_grad():
                for input_tuple, target in val_loader:
                    input, t = input_tuple
                    input = input.to(device, dtype=train_dtype)
                    t = t.to(device, dtype=train_dtype)
                    target = target.to(device, dtype=train_dtype)
                    # creates a tensor for delta_t to match tensor t
                    delta_t = torch.full_like(t, SECONDS_PER_STEP, device=device, dtype=train_dtype)
                    t_past = t - delta_t
                    output = self(input, t)

                    first_step_mask = torch.isclose(t, delta_t, rtol=0.0, atol=1e-6)  # creates bool-mask per batch-element (shape: (B,1)), True where t ≈ delta_t, so where t_past would be ≈ 0 (with a tolerance of 1e-6 to account for floating point precision issues)

                    output_past = self(input, t_past) # computes output_past for all samples

                    # For samples where t_past ≈ 0, we set output_past := input, because for the first simulation step the "past output" is just the initial condition (input).
                    if first_step_mask.any():  # check if there are any samples in the batch where t_past ≈ 0
                        # Reshape mask from (B,1) to (B,1,1,1,1), so one bool controls one full sample volume.
                        mask_5d = first_step_mask.view(t.shape[0], 1, 1, 1, 1)
                        # torch.where picks input where mask is True, otherwise keeps model output_past.
                        output_past = torch.where(mask_5d, input, output_past)
                    loss, mse_loss, physics_loss = criterion.compute_components(
                        input, output, output_past, t, t_past, target
                    )
                    val_loss += loss.item()
                    val_mse += mse_loss.item()
                    val_physics += physics_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            avg_train_mse = train_mse / len(train_loader)
            avg_train_physics = train_physics / len(train_loader)
            avg_val_mse = val_mse / len(val_loader)
            avg_val_physics = val_physics / len(val_loader)

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
            }
            append_metrics_row(metrics_path, header, row)
            append_metrics_row(components_path, components_header, components_row)
            best_val_mse = avg_val_mse if best_val_mse is None else min(best_val_mse, avg_val_mse)
            best_val_physics = avg_val_physics if best_val_physics is None else min(best_val_physics, avg_val_physics)
            update_run_config(
                config_path,
                {
                    "best_val_mse_loss": float(best_val_mse),
                    "best_val_physics_loss": float(best_val_physics),
                },
            )
            self.save_model(epoch, model_name, save_path, optimizer)

        metrics_path = os.path.join(save_path, model_name, "metrics.csv")
        epochs_all, train_losses_all, val_losses_all = load_loss_history_from_metrics(metrics_path)
        if not epochs_all:
            epochs_all, train_losses_all, val_losses_all = fallback_loss_history(num_epochs, train_losses, val_losses)

        self.save_loss_plot(model_name, epochs_all, train_losses_all, val_losses_all, save_path)
        proc_time = self.save_proc_time(model_name, tic, save_path)
        accumulate_training_duration(config_path, proc_time)

    def save_model(self, epoch, model_name, save_path, optimizer):
        model_dir = os.path.join(save_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        base_model_path = os.path.join(model_dir, f"epoch_{epoch}.pth")
        if os.path.exists(base_model_path):
            i = 1
            new_model_path = os.path.join(model_dir, f"epoch_{epoch + i}.pth")
            while os.path.exists(new_model_path):
                i += 1
                new_model_path = os.path.join(model_dir, f"epoch_{epoch + i}.pth")
            model_path = new_model_path
        else:
            model_path = base_model_path

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, model_path)

        latest_model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(checkpoint, latest_model_path)

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
