# VLAW Audit Runbook

## Goal
Run deterministic preflight/smoke checks before expensive PBS jobs and enforce a common artifact contract.

## Stage 1: Login-Node Preflight
- Command:
  - `PYTHONPATH=$PWD uv run python scripts/smoke/preflight_login.py --output logs/preflight_login.json`
- Pass criteria:
  - exit code `0`
  - `logs/preflight_login.json` exists
  - JSON status is `pass`
  - no `import_failed:*` failures

## Stage 2: Login-Node Control-Path Smoke
- Command:
  - `PYTHONPATH=$PWD uv run python scripts/smoke/login_smoke.py --config configs/droid_single_task_vlaw.yaml --output logs/login_smoke.json`
- Pass criteria:
  - exit code `0`
  - wrappers instantiate
  - output JSON exists with status `pass`

## Stage 3: Interface Contract Check
- Command:
  - `PYTHONPATH=$PWD uv run python scripts/smoke/interface_contract_check.py --output logs/interface_contract.json`
- Pass criteria:
  - exit code `0`
  - all checks true:
    - `policy.act_shape`
    - `world_model.rollout_contract`
    - `reward_model.score_contract`

## Stage 4: GPU Smoke via PBS (2-5 min)
- Command:
  - `qsub scripts/pbs/run_smoke_gpu.pbs`
- Pass criteria:
  - `logs/vlaw_gpu_smoke.log` contains `GPU smoke done`
  - `logs/smoke_results.json` reports `cuda_available=true`
  - `logs/preflight_login_gpu.json` and `logs/interface_contract_gpu.json` both pass

## Stage 5: Real Baseline Eval (OpenPI)
- Command:
  - `qsub scripts/pbs/run_openpi_real_eval.pbs`
- Pass criteria:
  - `results_base_real.json` exists
  - schema keys include: `result_type`, `task`, `success_rate`, `episodes`, `provenance`

## Stage 6: VLAW Loop
- Command:
  - `qsub scripts/pbs/run_vlaw_loop.pbs`
- Pass criteria:
  - `results_vlaw.json` exists
  - includes `metrics` and `metrics_meta`

## Stage 7: Artifact Verification and Plot
- Commands:
  - `PYTHONPATH=$PWD uv run python scripts/audit/verify_artifacts.py --output logs/artifact_verification.json`
  - `PYTHONPATH=$PWD uv run python plot_results.py`
- Pass criteria:
  - artifact verification status is `pass`
  - `vlaw_results_plot.png` regenerated successfully

## Notes on Environment Separation
- Split environments are allowed for reliability as long as model/data/checkpoint semantics are unchanged.
- Recommended:
  - OpenPI real eval in `~/.venv_openpi`
  - loop/report tooling in project `uv` env
