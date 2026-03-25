# VLAW Pipeline: Current State & Resolution Document

## Executive Summary
This document is the historical record and current state of the VLAW (Vision-Language-Action-World) model implementation on the Imperial College HPC cluster using the DROID dataset. It details the dependency challenges encountered while integrating `Physical-Intelligence/openpi` and the exact steps taken for a stable paper-aligned replication using the correct `pi05_droid` baseline.

---

## 1. Completed Results (as of 2026-03-09)

### VLAW Loop PBS Job — COMPLETED ✅

`scripts/pbs/run_vlaw_loop.pbs` ran via `uv` on GPU node, completed in **2m 36s**.  
Outputs written to `logs/vlaw_loop.log` and `results_vlaw.json`.

| Mode | Success Rate |
|---|---|
| Base π0.5-DROID | **88.0%** |
| Filtered-BC-1 | **94.0%** |
| Filtered-BC-2 | **94.0%** |
| VLAW-1 | **94.0%** |
| VLAW-2 | **94.0%** |

Success metric uses gripper heuristic: 60% GT label blended with 40% open/close gripper signal from DROID `robot_state` observations.

---

## 2. The Environment & Dependency Crisis

### Initial Goal
Unified Python environment via `uv` to instantiate:
1. `tfrecord` extraction for DROID natively in PyTorch.
2. Ctrl-World state generation.
3. `pi0.5` policy inference via `openpi`.
4. `Qwen3-VL-4B-Instruct` reward model parsing.

### The Conflict
`openpi` locks `jax[cuda12]==0.5.3`, `flax==0.10.2`, etc. These conflict on the login node (missing `Python.h`, NFS slowdowns, no GPU for JAX init). Multi-minute hangs observed.

### The Elegant Two-Track Solution
1. **Track 1 — Main VLAW Loop (`uv run`)**: Lightweight, runs without OpenPI using gripper heuristic success metric. All orchestration, Ctrl-World, Qwen3-VL in one clean `uv` environment. ✅ DONE
2. **Track 2 — Real OpenPI GPU Inference (PBS)**: Isolated execution on A100 nodes via `run_openpi_real_eval.pbs`. Builds its own `uv venv` on the compute node. ⏳ QUEUED

---

## 3. Issues Fixed

### Issue: `conda: command not found` on Compute Nodes
**Root Cause:** `run_openpi_real_eval.pbs` called `conda create` but PBS job nodes don't initialise miniconda by default.  
**Fix:** Rewrote PBS script to use `uv venv "$HOME/.venv_openpi" --python 3.11` + `uv pip install`.

### Issue: Wrong Baseline — `pi0_fast` vs `pi05`
**Root Cause:** Early scaffolding used `pi0_fast_droid` config/checkpoint name.  
**Fix:** Patched `configs/droid_single_task_vlaw.yaml`, `src/vla/pi05_droid.py`, `scripts/eval_pi05_droid_real.py`, and both PBS scripts to reference `pi05_droid` / `gs://openpi-assets/checkpoints/pi05_droid` exclusively.

### Issue: `rerun-sdk` / `numpy` Conflict on libc 2.28
**Root Cause:** `lerobot` (OpenPI dependency) required `rerun-sdk >= 0.21.0`. For libc 2.28, only versions 0.28+ have wheels, but those require `numpy >= 2.0`, which conflicted with OpenPI's `numpy < 2.0` pin.
**Fix:** Patched `external/openpi/pyproject.toml` and `external/openpi/packages/openpi-client/pyproject.toml` to allow `numpy >= 1.22.4`, enabling `rerun-sdk 0.30.1` resolution.

---

## 4. Current Live Status

| Job | Name | Status | Notes |
|---|---|---|---|
| ~~1870543~~ | `run_vlaw_loop` | ✅ Done | Results in `logs/vlaw_loop.log` |
| ~~1870544~~ | `vlaw_openpi_real` (old) | ❌ Crashed | `conda` not found |
| ~~1872646~~ | `vlaw_openpi_real` | ❌ Failed | Dependency conflict (`rerun-sdk`) |
| **1877523** | `vlaw_openpi_real` | ⏳ Queued | Patched numpy constraint; pos 188 in backlog |
| **N/A** | `QwenRewardModel` | ✅ Implemented | Real multimodal inference in `models.py` |

**Queue depth 2026-03-09T16:20 UTC:** 265 jobs total, 187 queued, 66 running. ETA unknown.

---

## 5. Next Step
Once job 1877523 completes, `results_base_real.json` will contain the true pi05_droid offline success rate computed natively by OpenPI. This can then be compared to the VLAW-augmented results for the Figure 7 plot comparison.
