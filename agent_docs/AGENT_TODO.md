# Agent TODO

Last updated: 2026-02-20

## Prioritized Backlog

### P0 (High impact, should do first)
- [ ] Make `main.py` use config-driven shared training params (`batch`, `channels`, `lr`) instead of hardcoded values.
- [ ] Unify run artifact schema across static/dynamic/PINN (`config.json`, `metrics.csv`, `proc_time_*.txt`, cumulative duration fields).
- [ ] Add explicit runtime prints for critical params in training logs (`TRAIN_DTYPE`, `batch`, `channels`, model class) to improve reproducibility.

### P1 (Medium impact)
- [ ] Refactor PINN into core training module (`training/train_pinn_core.py`) and keep `training/train_pinn.py` as thin CLI wrapper.
- [ ] Integrate PINN into `main.py` model registry so all training families share one entrypoint.
- [ ] Reuse `training/train_utils.py` helpers for PINN run config/metrics/proc-time handling.

### P2 (Quality and maintainability)
- [ ] Normalize/standardize README + LOCAL_COMMANDS with one canonical command per workflow (train, eval, benchmark, testset generation).
- [ ] Add lightweight smoke tests for new evaluation scripts (`evaluate_horizon.py`, `benchmark_inference.py`, `visualize_3d_predictions.py`).
- [ ] Add optional CSV/plot that compares simulation runtime vs inference runtime directly (`speedup_vs_time`).

## Notes
- Keep `configs/train_config.py` as the single source of truth for `TRAIN_DTYPE`.
- Avoid changing config files while jobs are queued/running unless jobs are held/released to preserve reproducibility.
