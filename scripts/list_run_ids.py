import argparse
import json
from pathlib import Path

'''takes variable values like a = 1 runs through all config files in ./runs/ and returns a list with IDs'''
def iter_run_configs(runs_root: Path, mode: str):
    modes = ["static", "dynamic"] if mode == "both" else [mode]
    for m in modes:
        mode_dir = runs_root / m
        if not mode_dir.exists():
            continue
        for config_path in sorted(mode_dir.rglob("config.json")):
            run_dir = config_path.parent
            try:
                with config_path.open("r") as f:
                    config = json.load(f)
            except Exception:
                continue
            yield run_dir.name, m, config


def matches_filters(run_id, mode, config, args):
    if args.a is not None and config.get("a") != args.a:
        return False

    if args.model_name is not None:
        # dynamic runs usually store the user model name in "name"
        # fallback to run_id and model_class if name is absent.
        candidate = str(
            config.get("name")
            or config.get("model_name")
            or config.get("model_class")
            or run_id
        )
        if args.model_name not in candidate:
            return False

    if args.model_class is not None and config.get("model_class") != args.model_class:
        return False

    if args.seed is not None and config.get("seed") != args.seed:
        return False

    if args.epochs is not None and config.get("epochs") != args.epochs:
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="List run IDs from ./runs by filtering config.json fields."
    )
    parser.add_argument("--runs-root", default="./runs", help="Path to runs directory.")
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic", "both"],
        default="both",
        help="Which run folders to search.",
    )
    parser.add_argument("--a", type=float, default=None, help="Filter by config value a.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Substring match against config[name]/model_class/run_id.",
    )
    parser.add_argument(
        "--model-class", default=None, help="Exact match for config[model_class]."
    )
    parser.add_argument("--seed", type=int, default=None, help="Filter by seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Filter by epochs.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional file path. If set, writes one run_id per line.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)

    matches = []
    for run_id, mode, config in iter_run_configs(runs_root, args.mode):
        if matches_filters(run_id, mode, config, args):
            matches.append(run_id)

    for run_id in matches:
        print(run_id)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for run_id in matches:
                f.write(f"{run_id}\n")

    print(f"Found {len(matches)} matching run IDs.")


if __name__ == "__main__":
    main()
