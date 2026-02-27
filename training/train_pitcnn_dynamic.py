import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import CombinedLoss_dynamic
from .train_utils import append_metrics_row, fallback_loss_history, load_checkpoint, load_loss_history_from_metrics, accumulate_training_duration
from configs.train_config import TRAIN_DTYPE

SIM_TOTAL_SECONDS = 10.0
SIM_STEPS_PER_SECOND = 1000.0
SECONDS_PER_STEP = 1.0 / SIM_STEPS_PER_SECOND


class BaseModel_dynamic(nn.Module):
    def __init__(self):
        super(BaseModel_dynamic, self).__init__()

    def train_model(
        self,
        a,
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
    ):
        tic = time.perf_counter()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device = " + device)
        self.to(device=device, dtype=TRAIN_DTYPE)
        
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
            a=a,
            mse_weight=mse_weight,
            device=device,
            min_temp=min_temp,
            max_temp=max_temp,
        ).to(device=device, dtype=TRAIN_DTYPE)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        start_epoch = 0
        if resume_checkpoint_path:
            start_epoch = load_checkpoint(self, optimizer, resume_checkpoint_path, device)
            print(f"Resuming from checkpoint: {resume_checkpoint_path} (start_epoch={start_epoch})")

        for epoch in range(start_epoch, num_epochs):
            train_loss = 0.0
            train_mse = 0.0
            train_physics = 0.0
            val_loss = 0.0
            val_mse = 0.0
            val_physics = 0.0

            self.train()
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for i, (input_tuple, target) in enumerate(loop):
                input, t = input_tuple
                input = input.to(device, dtype=TRAIN_DTYPE)
                t = t.to(device, dtype=TRAIN_DTYPE)
                target = target.to(device, dtype=TRAIN_DTYPE)

                delta_t = torch.full_like(t, SECONDS_PER_STEP, device=device, dtype=TRAIN_DTYPE)
                t_past = t - delta_t
                output = self(input, t)
                
                output_past = self(input, t_past)

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
                    input = input.to(device, dtype=TRAIN_DTYPE)
                    t = t.to(device, dtype=TRAIN_DTYPE)
                    target = target.to(device, dtype=TRAIN_DTYPE)

                    delta_t = torch.full_like(t, SECONDS_PER_STEP, device=device, dtype=TRAIN_DTYPE)
                    t_past = t - delta_t
                    output = self(input, t)
                    output_past = self(input, t_past)
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

            model_dir = os.path.join(save_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            metrics_path = os.path.join(model_dir, "metrics.csv")
            components_path = os.path.join(model_dir, "loss_components.csv")
            header = ["run_id", "epoch", "train_loss", "val_loss", "lr", "a", "channels", "batch", "seed"]
            components_header = [
                "run_id",
                "epoch",
                "train_mse_loss",
                "train_physics_loss",
                "val_mse_loss",
                "val_physics_loss",
                "a",
            ]
            row = {
                "run_id": run_id or model_name,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": learning_rate,
                "a": a,
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
                "a": a,
            }
            append_metrics_row(metrics_path, header, row)
            append_metrics_row(components_path, components_header, components_row)
            self.save_model(epoch, model_name, save_path, optimizer)

        metrics_path = os.path.join(save_path, model_name, "metrics.csv")
        epochs_all, train_losses_all, val_losses_all = load_loss_history_from_metrics(metrics_path)
        if not epochs_all:
            epochs_all, train_losses_all, val_losses_all = fallback_loss_history(num_epochs, train_losses, val_losses)

        self.save_loss_plot(model_name, epochs_all, train_losses_all, val_losses_all, save_path)
        proc_time = self.save_proc_time(model_name, tic, save_path)
        config_path = os.path.join(save_path, model_name, "config.json")
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
