import math
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import CombinedLoss_dynamic
from .train_utils import append_metrics_row, fallback_loss_history, load_checkpoint, load_loss_history_from_metrics


class BaseModel_dynamic(nn.Module):
    def __init__(self):
        super(BaseModel_dynamic, self).__init__()

    def train_model(
        self,
        a,
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
        self.to(device)
        self.double()

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

        criterion = CombinedLoss_dynamic(a=a, device=device).to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        start_epoch = 0
        if resume_checkpoint_path:
            start_epoch = load_checkpoint(self, optimizer, resume_checkpoint_path, device)
            print(f"Resuming from checkpoint: {resume_checkpoint_path} (start_epoch={start_epoch})")

        for epoch in range(start_epoch, num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            self.train()
            loop = tqdm(train_loader, total=len(train_loader), leave=True)
            for i, (input_tuple, target) in enumerate(loop):
                input, t = input_tuple
                input = input.to(device)
                t = t.to(device)
                target = target.to(device)
                output = self(input.double(), t)

                loss = criterion(input, t, output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(trainloss=train_loss / (i + 1))

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.eval()
            with torch.no_grad():
                for input_tuple, target in val_loader:
                    input, t = input_tuple
                    input = input.to(device)
                    t = t.to(device)
                    target = target.to(device)
                    output = self(input.double(), t)
                    loss = criterion(input, t, output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            model_dir = os.path.join(save_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            metrics_path = os.path.join(model_dir, "metrics.csv")
            header = ["run_id", "epoch", "train_loss", "val_loss", "lr", "a", "channels", "batch", "seed"]
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
            append_metrics_row(metrics_path, header, row)
            self.save_model(epoch, model_name, save_path, optimizer)

        metrics_path = os.path.join(save_path, model_name, "metrics.csv")
        epochs_all, train_losses_all, val_losses_all = load_loss_history_from_metrics(metrics_path)
        if not epochs_all:
            epochs_all, train_losses_all, val_losses_all = fallback_loss_history(num_epochs, train_losses, val_losses)

        self.save_loss_plot(model_name, epochs_all, train_losses_all, val_losses_all, save_path)
        self.save_proc_time(model_name, tic, save_path)

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
