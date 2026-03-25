# OpenPI Real Eval Dependency Root Cause

## Observed Failure
- Source log: `logs/vlaw_openpi_real.log`
- Error: `evdev` build fails with `fatal error: Python.h: No such file or directory`
- Transitive chain in log:
  - `openpi -> lerobot -> pynput -> evdev`

## Why It Fails
- `evdev` is a native-extension package that requires Python C headers at build time.
- On current node images, `Python.h` is missing for the interpreter used in batch setup.
- Because `openpi` depends on `lerobot` in this repo vendored package, the chain is pulled during `uv pip install -e external/openpi`.

## Valid Mitigation Paths
1. **Node prerequisite path (closest to current setup)**
   - Ensure Python dev headers/toolchain are available on PBS nodes used by the job.
   - Keep full `openpi` install path unchanged.
2. **Headless dependency minimization path**
   - Avoid pulling desktop/input stack for pure inference jobs.
   - Use narrower dependency set for evaluation-only workloads.
3. **Environment split path (recommended operationally)**
   - Isolate OpenPI eval runtime (`~/.venv_openpi`) from loop/report env.
   - Reduces cross-conflict risk without changing algorithmic semantics.

## Contract for This Repository
- `scripts/smoke/preflight_login.py` now fails fast when risky imports break.
- PBS scripts run preflight and contract checks before expensive phases.
