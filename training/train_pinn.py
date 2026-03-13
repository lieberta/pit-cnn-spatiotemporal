import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import HeatEquationPINNDataset
from models.pinn import PINN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_id(prefix="pinn"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def append_metrics_row(csv_path, row):
    header = ["epoch", "train_total", "train_data", "train_pde", "val_total", "val_data", "val_pde", "lr"]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(model, loader, device, optimizer, alpha, lambda_data, lambda_pde, train):
    train_dtype = torch.get_default_dtype()
    if train:
        model.train()
    else:
        model.eval()

    total_sum = 0.0
    data_sum = 0.0
    pde_sum = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 3:
            coords, target, source = batch
        else:
            coords, target = batch
            source = None

        coords = coords.to(device=device, dtype=train_dtype)
        target = target.to(device=device, dtype=train_dtype)
        if source is not None:
            source = source.to(device=device, dtype=train_dtype)

        coords = coords.reshape(-1, 4).contiguous()
        target = target.reshape(-1, 1).contiguous()
        if source is not None:
            source = source.reshape(-1, 1).contiguous()
        coords.requires_grad_(True)

        if train:
            optimizer.zero_grad()

        pred = model(coords)
        data_loss = F.mse_loss(pred, target)
        pde_residual = model.heat_residual(coords, alpha=alpha, source=source)
        pde_loss = torch.mean(pde_residual ** 2)
        loss = lambda_data * data_loss + lambda_pde * pde_loss

        if train:
            loss.backward()
            optimizer.step()

        total_sum += float(loss.detach().cpu().item())
        data_sum += float(data_loss.detach().cpu().item())
        pde_sum += float(pde_loss.detach().cpu().item())
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_sum / n_batches, data_sum / n_batches, pde_sum / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train a PINN on normalized heat-equation fields.")
    parser.add_argument("--base-path", type=str, default="./data/laplace_convolution/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--points-per-sample", type=int, default=8192)
    parser.add_argument("--modulo", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.0257)
    parser.add_argument("--source-threshold-raw", type=float, default=1000.0)
    parser.add_argument("--source-intensity-raw", type=float, default=100000.0)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=6)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-root", type=str, default="./runs/pinn")
    parser.add_argument("--train-dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    tic = time.perf_counter()

    set_seed(args.seed)
    torch.set_default_dtype(torch.float64 if args.train_dtype == "float64" else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dtype = torch.get_default_dtype()
    print(f"Device: {device}")

    dataset = HeatEquationPINNDataset(
        base_path=args.base_path,
        points_per_sample=args.points_per_sample,
        modulo=args.modulo,
        source_threshold_raw=args.source_threshold_raw,
        source_intensity_raw=args.source_intensity_raw,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No normalized experiments found in: {args.base_path}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PINN(
        in_features=4,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        out_features=1,
    ).to(device=device, dtype=train_dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_id = make_run_id("pinn")
    run_dir = os.path.join(args.runs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "model_class": "PINN",
                "name": "PINN",
                "base_path": args.base_path,
                "epochs": args.epochs,
                "batch": args.batch_size,
                "lr": args.lr,
                "points_per_sample": args.points_per_sample,
                "modulo": args.modulo,
                "alpha": args.alpha,
                "lambda_data": args.lambda_data,
                "lambda_pde": args.lambda_pde,
                "hidden_features": args.hidden_features,
                "hidden_layers": args.hidden_layers,
                "val_split": args.val_split,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "runs_root": args.runs_root,
                "source_threshold_raw": args.source_threshold_raw,
                "source_intensity_raw": args.source_intensity_raw,
                "training_dtype": str(train_dtype),
            },
            f,
            indent=2,
        )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_total, train_data, train_pde = run_epoch(
            model, train_loader, device, optimizer, args.alpha, args.lambda_data, args.lambda_pde, train=True
        )
        val_total, val_data, val_pde = run_epoch(
            model, val_loader, device, optimizer, args.alpha, args.lambda_data, args.lambda_pde, train=False
        )

        append_metrics_row(
            metrics_path,
            {
                "epoch": epoch,
                "train_total": train_total,
                "train_data": train_data,
                "train_pde": train_pde,
                "val_total": val_total,
                "val_data": val_data,
                "val_pde": val_pde,
                "lr": args.lr,
            },
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(run_dir, "last.pth"))
        if val_total < best_val:
            best_val = val_total
            torch.save(checkpoint, os.path.join(run_dir, "best.pth"))

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train(total={train_total:.6f}, data={train_data:.6f}, pde={train_pde:.6f}) "
            f"val(total={val_total:.6f}, data={val_data:.6f}, pde={val_pde:.6f})"
        )

    proc_time = float(time.perf_counter() - tic)
    with open(os.path.join(run_dir, f"proc_time_{run_id}.txt"), "w") as f:
        f.write(f"Training process duration: {time.strftime('%H:%M:%S', time.gmtime(proc_time))}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["training_duration_seconds"] = proc_time
    cfg["training_duration_hms"] = time.strftime("%H:%M:%S", time.gmtime(proc_time))
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
