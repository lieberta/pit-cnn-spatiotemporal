import argparse
import shutil
from pathlib import Path


def list_experiment_dirs(base_path: Path):
    return sorted(p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("experiment"))


def main():
    parser = argparse.ArgumentParser(description="Copy every nth experiment folder into a target dataset directory.")
    parser.add_argument("--source", required=True, help="Source dataset root.")
    parser.add_argument("--target", required=True, help="Target dataset root.")
    parser.add_argument("--stride", type=int, default=10, help="Copy every nth experiment folder.")
    parser.add_argument("--offset", type=int, default=0, help="Start copying at this source index offset.")
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    source = Path(args.source)
    target = Path(args.target)
    if not source.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    target.mkdir(parents=True, exist_ok=True)

    norm_path = source / "normalization_values.json"
    if norm_path.exists():
        shutil.copy2(norm_path, target / "normalization_values.json")

    experiments = list_experiment_dirs(source)
    copied = 0
    skipped_existing = 0

    for idx, exp_dir in enumerate(experiments):
        if idx < args.offset:
            continue
        if idx % args.stride != 0:
            continue

        dst = target / exp_dir.name
        if dst.exists():
            skipped_existing += 1
            continue

        shutil.copytree(exp_dir, dst)
        copied += 1
        print(f"copied: {exp_dir.name}", flush=True)

    print(f"source={source}")
    print(f"target={target}")
    print(f"stride={args.stride}")
    print(f"offset={args.offset}")
    print(f"copied={copied}")
    print(f"skipped_existing={skipped_existing}")


if __name__ == "__main__":
    main()
