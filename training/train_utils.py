import csv
import os

import torch


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        return start_epoch
    model.load_state_dict(checkpoint)
    return 0


def append_metrics_row(csv_path, header, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_loss_history_from_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return [], [], []

    by_epoch = {}
    with open(metrics_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            by_epoch[epoch] = (train_loss, val_loss)

    epochs = sorted(by_epoch.keys())
    train_losses = [by_epoch[e][0] for e in epochs]
    val_losses = [by_epoch[e][1] for e in epochs]
    return epochs, train_losses, val_losses


def fallback_loss_history(num_epochs, train_losses, val_losses):
    count = min(len(train_losses), len(val_losses))
    epochs = list(range(1, count + 1))
    return epochs, train_losses[:count], val_losses[:count]
