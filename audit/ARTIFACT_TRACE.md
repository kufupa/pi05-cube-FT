# Artifact Trace Map

## Producer -> Artifact -> Consumer

- `scripts/pbs/run_smoke_gpu.pbs`
  - produces: `logs/vlaw_gpu_smoke.log`, `logs/smoke_results.json`, `logs/preflight_login_gpu.json`, `logs/interface_contract_gpu.json`
  - consumed by: audit review / acceptance gate

- `scripts/pbs/run_openpi_real_eval.pbs`
  - produces: `logs/vlaw_openpi_real.log`, `results_base_real.json`
  - consumed by: `src/training/run_vlaw_loop.py` (optional ingestion as `Base-Real`), reporting

- `scripts/pbs/run_vlaw_loop.pbs`
  - produces: `logs/vlaw_loop.log`, `results_vlaw.json`
  - consumed by: `plot_results.py`, `scripts/audit/verify_artifacts.py`

- `scripts/eval_pi05_droid.py`
  - produces: `results_base.json`
  - consumed by: baseline comparison/manual analysis

- `plot_results.py`
  - consumes: `results_vlaw.json`
  - produces: `vlaw_results_plot.png`

- `scripts/audit/verify_artifacts.py`
  - consumes: produced artifacts above
  - produces: `logs/artifact_verification*.json`

## Failure Visibility
- Missing required scripts -> `verify_artifacts.py` status `fail`
- Missing optional result artifacts -> reported in findings, not fatal by default
- Schema mismatch in present results -> status `fail`
