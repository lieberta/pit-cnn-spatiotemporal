import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from simulation.heat_sim_initial import HeatSimulation
from simulation.preprocess import norm_values, normalization


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_experiment(
    out_root: Path,
    num_fires: int,
    experiment_idx: int,
    sim_time_seconds: float,
    nt: int,
    save_every: int,
    device: torch.device,
):
    sim = HeatSimulation(num_fires=num_fires, T=sim_time_seconds, Nt=nt, device=device)

    wall_tic = time.perf_counter()
    sim.run_simulation()
    wall_tac = time.perf_counter()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"experiment_{num_fires}_{now}_{experiment_idx:03d}"
    exp_dir = out_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    temperature_np = sim.u[::save_every].detach().cpu().numpy()
    time_axis = np.linspace(0.0, sim_time_seconds, nt + 1)[::save_every]

    np.savez(
        exp_dir / "heat_equation_solution.npz",
        time=time_axis,
        x=np.linspace(0, sim.Lx, sim.Nx),
        y=np.linspace(0, sim.Ly, sim.Ny),
        z=np.linspace(0, sim.Lz, sim.Nz),
        temperature=temperature_np,
    )

    fireplaces = []
    for i, (x_start, y_start, x_size, y_size) in enumerate(sim.fireplaces):
        fireplaces.append(
            {
                "fireplace_idx": i + 1,
                "x_start": int(x_start),
                "y_start": int(y_start),
                "x_size": int(x_size),
                "y_size": int(y_size),
            }
        )

    runtime_seconds = float(wall_tac - wall_tic)
    sim_steps_per_second = float(nt / runtime_seconds) if runtime_seconds > 0 else float("inf")

    meta = {
        "experiment_name": exp_name,
        "num_fires": int(num_fires),
        "experiment_idx": int(experiment_idx),
        "device": str(device),
        "sim_time_seconds": float(sim_time_seconds),
        "nt": int(nt),
        "dt": float(sim.dt),
        "saved_every_n_steps": int(save_every),
        "saved_frames": int(temperature_np.shape[0]),
        "saved_dt_seconds": float(sim.dt * save_every),
        "runtime_seconds": runtime_seconds,
        "sim_steps_per_second": sim_steps_per_second,
        "temperature_min": float(sim.min_temp),
        "temperature_max": float(sim.max_temp),
        "grid": {
            "Lx": float(sim.Lx),
            "Ly": float(sim.Ly),
            "Lz": float(sim.Lz),
            "Nx": int(sim.Nx),
            "Ny": int(sim.Ny),
            "Nz": int(sim.Nz),
        },
        "fireplaces": fireplaces,
    }
    write_json(exp_dir / "metadata.json", meta)

    # Keep the old-style text summary for compatibility/human readability.
    summary_lines = []
    for fp in fireplaces:
        summary_lines.append(
            f"Fireplace {fp['fireplace_idx']}: x={fp['x_start']}, y={fp['y_start']}, "
            f"width={fp['x_size']}, depth={fp['y_size']}"
        )
    summary_lines.extend(
        [
            f"Simulated Time={sim_time_seconds}s, dt={sim.dt}s, Nt={nt}",
            f"Saved every {save_every} steps => saved_dt={sim.dt * save_every}s, frames={temperature_np.shape[0]}",
            f"Lx={sim.Lx}, Ly={sim.Ly}, Lz={sim.Lz}, Nx={sim.Nx}, Ny={sim.Ny}, Nz={sim.Nz}",
            f"Min Temp={sim.min_temp}, Max Temp={sim.max_temp}",
            f"Runtime (wall)={runtime_seconds}s",
        ]
    )
    (exp_dir / "fireplace_simulation_results.txt").write_text("\n".join(summary_lines) + "\n")

    return {
        "experiment_name": exp_name,
        "num_fires": int(num_fires),
        "experiment_idx": int(experiment_idx),
        "sim_time_seconds": float(sim_time_seconds),
        "nt": int(nt),
        "dt": float(sim.dt),
        "save_every": int(save_every),
        "saved_frames": int(temperature_np.shape[0]),
        "saved_dt_seconds": float(sim.dt * save_every),
        "runtime_seconds": runtime_seconds,
        "sim_steps_per_second": sim_steps_per_second,
        "temperature_min": float(sim.min_temp),
        "temperature_max": float(sim.max_temp),
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate timed heat-simulation testset with configurable fire counts and horizon."
    )
    parser.add_argument("--out-root", default="./data/testset_20s", help="target folder for new dataset")
    parser.add_argument("--fires-min", type=int, default=1)
    parser.add_argument("--fires-max", type=int, default=10)
    parser.add_argument("--experiments-per-fire", type=int, default=1)
    parser.add_argument("--sim-time-seconds", type=float, default=20.0)
    parser.add_argument("--nt", type=int, default=20000, help="number of simulation steps")
    parser.add_argument("--save-every", type=int, default=100, help="save every n-th simulated step")
    parser.add_argument("--device", default=None, help="cpu|cuda; default auto")
    parser.add_argument("--normalize", action="store_true", help="compute info.json and normalized npz files")
    args = parser.parse_args()

    if args.fires_min < 1 or args.fires_max < args.fires_min:
        raise ValueError("Invalid fire range. Need 1 <= fires_min <= fires_max.")
    if args.experiments_per_fire < 1:
        raise ValueError("experiments-per-fire must be >= 1")
    if args.nt < 1 or args.save_every < 1:
        raise ValueError("nt and save-every must be >= 1")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[info] out_root={out_root}")
    print(f"[info] fire counts={args.fires_min}..{args.fires_max}, experiments_per_fire={args.experiments_per_fire}")
    print(f"[info] sim_time={args.sim_time_seconds}s, nt={args.nt}, dt={args.sim_time_seconds / args.nt}s")
    print(f"[info] save_every={args.save_every} -> saved_dt={(args.sim_time_seconds / args.nt) * args.save_every}s")
    print(f"[info] device={device}")

    global_tic = time.perf_counter()
    rows = []

    for num_fires in range(args.fires_min, args.fires_max + 1):
        for i in range(args.experiments_per_fire):
            print(f"[run] fires={num_fires}, experiment={i + 1}/{args.experiments_per_fire}")
            row = run_one_experiment(
                out_root=out_root,
                num_fires=num_fires,
                experiment_idx=i,
                sim_time_seconds=args.sim_time_seconds,
                nt=args.nt,
                save_every=args.save_every,
                device=device,
            )
            rows.append(row)
            print(
                f"[done] {row['experiment_name']} runtime={row['runtime_seconds']:.2f}s "
                f"saved_frames={row['saved_frames']}"
            )

    global_runtime = float(time.perf_counter() - global_tic)
    manifest = {
        "out_root": str(out_root),
        "generated_at": datetime.now().isoformat(),
        "device": str(device),
        "fires_min": int(args.fires_min),
        "fires_max": int(args.fires_max),
        "experiments_per_fire": int(args.experiments_per_fire),
        "sim_time_seconds": float(args.sim_time_seconds),
        "nt": int(args.nt),
        "dt": float(args.sim_time_seconds / args.nt),
        "save_every": int(args.save_every),
        "saved_dt_seconds": float((args.sim_time_seconds / args.nt) * args.save_every),
        "total_experiments": int(len(rows)),
        "total_runtime_seconds": global_runtime,
        "avg_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in rows])) if rows else 0.0,
        "rows": rows,
    }
    write_json(out_root / "simulation_manifest.json", manifest)

    write_csv(
        out_root / "simulation_manifest.csv",
        rows,
        [
            "experiment_name",
            "num_fires",
            "experiment_idx",
            "sim_time_seconds",
            "nt",
            "dt",
            "save_every",
            "saved_frames",
            "saved_dt_seconds",
            "runtime_seconds",
            "sim_steps_per_second",
            "temperature_min",
            "temperature_max",
            "device",
        ],
    )

    if args.normalize:
        print("[info] computing dataset info and normalized files...")
        norm_values(
            str(out_root),
            dt=float(args.sim_time_seconds / args.nt),
            num_timesteps=int(args.nt),
        )
        normalization(str(out_root))

    print(f"[done] generated {len(rows)} experiments in {global_runtime:.2f}s")
    print(f"[done] manifest: {out_root / 'simulation_manifest.csv'}")


if __name__ == "__main__":
    main()
