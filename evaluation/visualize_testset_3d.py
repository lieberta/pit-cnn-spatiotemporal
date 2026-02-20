import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def list_experiment_dirs(base_path: Path):
    return sorted([p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("experiment_")])


def load_field(exp_dir: Path, normalized: bool):
    fname = "normalized_heat_equation_solution.npz" if normalized else "heat_equation_solution.npz"
    npz_path = exp_dir / fname
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    field = data["temperature"]  # [nt, nx, ny, nz]
    times = data["time"] if "time" in data.files else np.arange(field.shape[0], dtype=np.float64)
    return field, times


def render_frame(
    ax,
    temp_3d,
    vmin,
    vmax,
    elev,
    azim,
    downsample,
    threshold_quantile,
    style,
    max_cloud_points,
    cloud_gamma,
    frame_seed,
    source_xy,
    source_marker_mode,
    ambient_temp,
    heat_threshold,
):
    # Downsample to keep rendering practical for long videos.
    t = temp_3d[::downsample, ::downsample, ::downsample]
    if style == "voxel":
        # Only color voxels where temperature is above ambient baseline.
        heat = np.clip(t - ambient_temp, 0.0, None)
        heat_norm = np.clip(heat / (max(vmax - ambient_temp, 1e-12)), 0.0, 1.0)
        active = heat_norm > 0.0
        # Frame-local normalization keeps subtle late-stage diffusion visible.
        if np.any(active):
            frame_ref = float(np.quantile(heat[active], 0.985))
            frame_ref = max(frame_ref, 1e-12)
            heat_vis = np.clip(heat / frame_ref, 0.0, 1.0)
        else:
            heat_vis = np.zeros_like(heat_norm)
        # Keep a low floor so even small temperature increases remain visible.
        low_cut_global = max(0.0015, 0.20 * heat_threshold)
        low_cut_local = 0.02
        filled = (heat_norm >= low_cut_global) | (heat_vis >= low_cut_local)

        facecolors = np.zeros(t.shape + (4,), dtype=np.float64)
        if np.any(filled):
            # Yellow (cooler) -> red (hotter) with partial transparency
            # so inner/occluded structures remain visible.
            h = heat_vis[filled]
            fc = np.zeros((h.shape[0], 4), dtype=np.float64)
            fc[:, 0] = 1.0
            fc[:, 1] = 0.92 * (1.0 - h)
            fc[:, 2] = 0.0
            # Slight changes: pale/translucent red. Hot zones: stronger red.
            fc[:, 3] = 0.08 + 0.47 * np.sqrt(h)
            facecolors[filled] = fc

        # Mark heat sources with three visual layers:
        # 1) source core voxel at z=0, 2) floor halo ring, 3) short vertical source column.
        if source_xy is not None and len(source_xy) > 0:
            sx = np.clip(source_xy[:, 0], 0, t.shape[0] - 1)
            sy = np.clip(source_xy[:, 1], 0, t.shape[1] - 1)
            for x0, y0 in zip(sx, sy):
                # 1) Source core at floor.
                if source_marker_mode in ("all", "core"):
                    filled[x0, y0, 0] = True
                    facecolors[x0, y0, 0] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

                # 2) Floor halo ring around source.
                if source_marker_mode in ("all", "halo"):
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            xx = x0 + dx
                            yy = y0 + dy
                            if 0 <= xx < t.shape[0] and 0 <= yy < t.shape[1]:
                                filled[xx, yy, 0] = True
                                facecolors[xx, yy, 0] = np.array([1.0, 0.80, 0.15, 0.55], dtype=np.float64)

                # 3) Short source column for oblique views.
                if source_marker_mode in ("all", "column"):
                    z_max = min(3, t.shape[2])
                    for zz in range(1, z_max):
                        filled[x0, y0, zz] = True
                        facecolors[x0, y0, zz] = np.array([0.08, 0.08, 0.08, 0.62], dtype=np.float64)

        ax.voxels(
            filled,
            facecolors=facecolors,
            edgecolor=(0.0, 0.0, 0.0, 0.06),
            linewidth=0.12,
        )
    elif style == "cloud":
        rng = np.random.default_rng(frame_seed)
        norm = np.clip((t - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
        active_mask = norm > np.quantile(norm, max(0.0, threshold_quantile - 0.35))
        if not np.any(active_mask):
            active_mask = norm > 0.0

        x, y, z = np.where(active_mask)
        w = np.power(norm[active_mask], max(1e-6, cloud_gamma))
        w_sum = np.sum(w)
        if w_sum <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w_sum

        n = min(max_cloud_points, len(x))
        chosen = rng.choice(len(x), size=n, replace=False, p=w if len(x) > 1 else None)

        xs = x[chosen].astype(np.float64) + rng.normal(0.0, 0.15, size=n)
        ys = y[chosen].astype(np.float64) + rng.normal(0.0, 0.15, size=n)
        zs = z[chosen].astype(np.float64) + rng.normal(0.0, 0.15, size=n)
        vals = norm[active_mask][chosen]

        colors = plt.cm.autumn_r(vals)  # yellow -> red (hotter = red)
        sizes = 10 + 42 * vals
        alphas = 0.12 + 0.72 * vals
        colors[:, 3] = np.clip(alphas, 0.12, 0.98)

        ax.scatter(xs, ys, zs, c=colors, s=sizes, marker="o", linewidths=0.0, depthshade=False)
    else:
        # Two-layer rendering improves 3D shape perception:
        # warm shell + hot core.
        q_shell = np.quantile(t, max(0.0, threshold_quantile - 0.55))
        q_core = np.quantile(t, threshold_quantile)

        mask_shell = t >= q_shell
        mask_core = t >= q_core
        if not np.any(mask_core):
            mask_core = t >= np.max(t)

        if np.any(mask_shell):
            xs, ys, zs = np.where(mask_shell)
            vals_shell = t[mask_shell]
            norm_shell = np.clip((vals_shell - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
            colors_shell = plt.cm.autumn_r(norm_shell)  # yellow -> red
            ax.scatter(xs, ys, zs, c=colors_shell, s=12, marker="s", linewidths=0.0, alpha=0.18)

        xc, yc, zc = np.where(mask_core)
        vals_core = t[mask_core]
        norm_core = np.clip((vals_core - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
        colors_core = plt.cm.autumn_r(norm_core)  # yellow -> red
        sizes_core = 16 + 44 * norm_core
        ax.scatter(xc, yc, zc, c=colors_core, s=sizes_core, marker="s", linewidths=0.0, alpha=0.62)

    if style != "voxel" and source_marker_mode != "none" and source_xy is not None and len(source_xy) > 0:
        sx = source_xy[:, 0]
        sy = source_xy[:, 1]
        # Draw source cells as floor tiles (square fields on z=0).
        sz = np.zeros_like(sx, dtype=np.float64) + 0.02
        ax.scatter(
            sx,
            sy,
            sz,
            c="black",
            s=260,
            marker="s",
            linewidths=1.0,
            edgecolors="black",
            alpha=0.98,
            depthshade=False,
        )

    # Lightweight 3D room boundary lines for orientation.
    x0, x1 = 0, t.shape[0] - 1
    y0, y1 = 0, t.shape[1] - 1
    z0, z1 = 0, t.shape[2] - 1
    edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x0, y1, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x0, y1, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x0, y1, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
    ]
    for a, b in edges:
        ax.plot3D(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=(0.3, 0.3, 0.3, 0.55),
            linewidth=0.8,
        )

    ax.set_xlim(0, t.shape[0] - 1)
    ax.set_ylim(0, t.shape[1] - 1)
    ax.set_zlim(0, t.shape[2] - 1)
    ax.set_box_aspect((t.shape[0], t.shape[1], t.shape[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_facecolor("white")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)


def make_video_for_experiment(
    exp_dir: Path,
    out_root: Path,
    normalized: bool,
    fps: int,
    frame_stride: int,
    max_frames: int,
    downsample: int,
    threshold_quantile: float,
    elev: float,
    azim: float,
    save_frames_every: int,
    style: str,
    max_cloud_points: int,
    cloud_gamma: float,
    source_marker_mode: str,
    ambient_temp: Optional[float],
    heat_threshold: float,
):
    loaded = load_field(exp_dir, normalized=normalized)
    if loaded is None:
        print(f"[skip] missing npz in {exp_dir}")
        return

    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError("imageio is required for video writing (pip install imageio)") from e

    field, times = loaded
    nt = field.shape[0]

    idxs = list(range(0, nt, max(1, frame_stride)))
    if max_frames > 0:
        idxs = idxs[:max_frames]
    if not idxs:
        print(f"[skip] no frames selected for {exp_dir.name}")
        return

    vmin = float(np.min(field))
    vmax = float(np.max(field))

    # Detect source locations on the floor from t=0.
    t0 = field[0, :: max(1, downsample), :: max(1, downsample), :: max(1, downsample)]
    floor = t0[:, :, 0]
    floor_max = float(np.max(floor))
    floor_med = float(np.median(floor))
    source_threshold = floor_med + 0.7 * (floor_max - floor_med)
    source_mask = floor >= source_threshold
    if not np.any(source_mask):
        source_mask = floor == floor_max
    src_x, src_y = np.where(source_mask)
    source_xy = np.stack([src_x, src_y], axis=1) if len(src_x) > 0 else None
    if ambient_temp is None:
        ambient_temp = float(np.quantile(t0, 0.25))

    out_dir = out_root / exp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if save_frames_every > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "temperature_cube.mp4"

    writer = imageio.get_writer(video_path, fps=max(1, fps))
    try:
        for i, t_idx in enumerate(idxs):
            fig = plt.figure(figsize=(8, 6), dpi=120)
            ax = fig.add_subplot(111, projection="3d")
            render_frame(
                ax=ax,
                temp_3d=field[t_idx],
                vmin=vmin,
                vmax=vmax,
                elev=elev,
                azim=azim,
                downsample=max(1, downsample),
                threshold_quantile=threshold_quantile,
                style=style,
                max_cloud_points=max(1, max_cloud_points),
                cloud_gamma=cloud_gamma,
                frame_seed=t_idx,
                source_xy=source_xy,
                source_marker_mode=source_marker_mode,
                ambient_temp=ambient_temp,
                heat_threshold=heat_threshold,
            )
            t_val = times[t_idx] if t_idx < len(times) else float(t_idx)
            unit = "norm" if normalized else "C"
            ax.set_title(f"{exp_dir.name} | t={t_val:.2f} | temp ({unit})")
            fig.tight_layout()

            if save_frames_every > 0 and (i % save_frames_every == 0):
                frame_png = frames_dir / f"frame_{i:05d}_tidx_{t_idx:05d}.png"
                fig.savefig(frame_png)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(img)
            plt.close(fig)

            if (i + 1) % 20 == 0 or (i + 1) == len(idxs):
                print(f"[progress] {exp_dir.name}: {i + 1}/{len(idxs)} frames")
    finally:
        writer.close()

    print(f"[done] wrote {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Render 3D temperature evolution videos for testset experiments.")
    parser.add_argument("--base-path", default="./data/testset_20s")
    parser.add_argument("--out-root", default="./plots/testset_20s_3d")
    parser.add_argument("--normalized", action="store_true", help="use normalized_heat_equation_solution.npz")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=2, help="use every n-th time frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected frames")
    parser.add_argument("--downsample", type=int, default=2, help="spatial downsample for rendering speed")
    parser.add_argument("--threshold-quantile", type=float, default=0.85, help="render hottest voxels above this quantile")
    parser.add_argument("--style", choices=["voxel", "cloud", "points"], default="cloud", help="voxel=filled cubes, cloud=volumetric scatter, points=square scatter")
    parser.add_argument("--max-cloud-points", type=int, default=12000, help="max points per frame in cloud mode")
    parser.add_argument("--cloud-gamma", type=float, default=2.2, help="higher -> stronger focus on hot regions in cloud mode")
    parser.add_argument("--source-marker-mode", choices=["none", "all", "core", "halo", "column"], default="all")
    parser.add_argument("--ambient-temp", type=float, default=None, help="baseline temp; defaults to ambient estimate from t=0")
    parser.add_argument("--heat-threshold", type=float, default=0.04, help="min normalized heat above ambient to render voxels")
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-45.0)
    parser.add_argument(
        "--save-frames-every",
        type=int,
        default=10,
        help="save every n-th selected frame as PNG in a separate frames/ folder; 0 disables",
    )
    parser.add_argument("--experiment", default=None, help="optional single experiment folder name")
    args = parser.parse_args()

    base = Path(args.base_path)
    out_root = Path(args.out_root)

    if args.experiment:
        exps = [base / args.experiment]
    else:
        exps = list_experiment_dirs(base)

    if not exps:
        raise RuntimeError(f"No experiment_* folders found in {base}")

    for exp in exps:
        if not exp.exists():
            print(f"[skip] experiment not found: {exp}")
            continue
        make_video_for_experiment(
            exp_dir=exp,
            out_root=out_root,
            normalized=args.normalized,
            fps=args.fps,
            frame_stride=max(1, args.frame_stride),
            max_frames=max(0, args.max_frames),
            downsample=max(1, args.downsample),
            threshold_quantile=min(max(args.threshold_quantile, 0.0), 1.0),
            elev=args.elev,
            azim=args.azim,
            save_frames_every=max(0, args.save_frames_every),
            style=args.style,
            max_cloud_points=max(1, args.max_cloud_points),
            cloud_gamma=max(1e-6, args.cloud_gamma),
            source_marker_mode=args.source_marker_mode,
            ambient_temp=args.ambient_temp,
            heat_threshold=max(0.0, min(1.0, args.heat_threshold)),
        )


if __name__ == "__main__":
    main()
