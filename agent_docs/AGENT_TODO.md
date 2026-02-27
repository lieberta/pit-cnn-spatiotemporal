# Agent TODO

Last updated: 2026-02-26

## Prioritized Backlog

### P0 (High impact, should do first)
- [ ] Run controlled A/B training check after V0.3 fixes (`a=1` vs `a=0`) and verify whether dynamic physics loss now decreases materially over 10+ epochs.
- [ ] Make the A/B check truly comparable: same seed, same epoch count, same code snapshot, same optimizer settings.
- [ ] Regenerate/normalize a clean dataset with the new interior-only source placement (`z=1:3`, no x/y boundary fireplaces) before judging V0.3 fairly.
- [ ] Decide and enforce one boundary strategy in dynamic models: copy-from-input boundaries vs explicit constant ambient boundary in normalized space.
- [ ] Add a configurable physics weight (`lambda_phys`) and warmup schedule (while keeping `a` as the coarse on/off multiplier).
- [ ] Add optional `use_past_laplacian` switch in physics loss for explicit A/B (`laplacian(output)` vs `laplacian(output_past)`), defaulting to current explicit-Euler-aligned behavior.
- [ ] Add minimal one-batch instrumentation in dynamic training: min/max/mean for `u0`, `target`, `prediction`, and PDE residual (at first batch, plus one random batch).
- [ ] Add a ground-truth sanity check utility: compute simulator FD residual on GT pair `(u_n, u_{n+1})` and compare magnitude against model residual.
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
- [ ] Add lightweight benchmark for `evaluation/visualize_heatvid_3d.py` runtime with zarr input (report frames/sec by style).
- [ ] Add a tiny diagnostic script to compare simulator FD residual on ground truth vs model residual (single batch).

## Notes
- Keep `configs/train_config.py` as the single source of truth for `TRAIN_DTYPE`.
- Avoid changing config files while jobs are queued/running unless jobs are held/released to preserve reproducibility.
- Keep simulation and physics-loss discretization strictly matched (`dt`, Laplacian scaling, source mask, boundary treatment) to prevent optimization conflicts.
