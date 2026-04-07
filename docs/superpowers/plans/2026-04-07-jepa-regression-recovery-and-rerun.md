# JEPA Regression Recovery And Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the phase07 export and phase08 bridge data path so CEM+WM is the default executed controller, SmolVLA actions are still logged for comparison, real images are captured, then rerun A/B/C training and eval with collision-safe storage and deterministic Slurm orchestration.

**Architecture:** Keep the existing 11-stage Slurm DAG, but harden the two quality-critical handoff points: `jepa_cem_paired_pushv3_export.py` and `bridge_builder.py`. Switch exporter arbitration to CEM-primary (`action_executed <- action_wm_cem_first` when available), retain SmolVLA action proposals as diagnostics, and store side-by-side action streams for auditability. Add exporter/bridge quality gates that fail fast on stride errors, missing images, and heuristic-fallback domination. Scope all run outputs (export, bridge, artifacts, reports, logs) by one run id so parallel jobs cannot overwrite each other.

**Tech Stack:** Bash, Python 3.10, PyTorch, MetaWorld, JEPA-WMS (`VGG JEPA/jepa-wms`), LeRobot, Slurm (`sbatch/squeue/scontrol/sacct`), JSON/Parquet.

---

## Goal Link To Old Plan And Updated DoD

- Keep from old plan (`docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`):
  - Full Slurm autonomy.
  - Branch-parallel scheduling.
  - TrainA/B/C from same base checkpoint.
- Change now (because regression root cause is identified):
  - **Old DoD:** `stage05/06/08` reached `RUNNING`.
  - **New DoD:** pipeline is only done when export+bridge quality gates pass and retrained A/B/C no longer collapse due to heuristic/blank-image data.

### Updated Definition Of Done (Strict)

1. `phase07` exporter quality gate passes:
   - `wm_step_error_rate <= 0.05`
   - `policy_exec_error_rate <= 0.05`
   - `episodes_with_images == total_episodes`
   - `heuristic_fallback_episode_ratio <= 0.10`
2. `phase08` bridge quality gate passes:
   - non-empty train+val
   - `image_nonblank_ratio >= 0.95`
   - `action_std_mean >= 0.02`
3. Retrain from fixed dataset: `stage05`, `stage06`, `stage08` complete.
4. Post-train eval is run against checkpoints from the same run scope.
5. Final report contains baseline vs A/B/C metrics and exact artifact paths.

### Action Arbitration Policy (Updated)

- Default exporter mode: `execution_policy=cem_primary`.
- Per step action choice:
  1. use `action_wm_cem_first` if WM+CEM succeeded,
  2. else use `action_smolvla_raw` if SmolVLA produced a valid action,
  3. else use heuristic fallback.
- Always log all available candidates even when not executed.
- Keep `smolvla_primary` as optional ablation mode only (not default).

### Trajectory Storage Contract (What We Store)

Per episode:
- `images`, `state`, `actions` (executed stream), `action_chunk`, `success`, `done`, `pair_key`, `meta`.

Per step (`cem_plan.per_step[i]`):
- `action_wm_cem_first`: first control from best CEM plan (`env_action_dim` clipped).
- `action_wm_cem_plan_seq`: full CEM planned sequence (`horizon x planner_action_dim`), before env truncation.
- `action_smolvla_raw`: SmolVLA proposal at that step (`env_action_dim` clipped).
- `action_executed`: action sent to `env.step`.
- `policy_source`: one of `cem_mpc_wm`, `smolvla`, `heuristic_fallback`.
- `reward`, `terminated`, `truncated`, `done`.
- `planner_metadata`: CEM params/stats (`cem_cost`, `cem_iterations`, `cem_seed`, dims, errors if any).
- `latent_pred`: WM latent prediction summary for chosen CEM plan.

---

## File/Resource Coverage (Reviewed)

**Core implementation files (modify):**
- `scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py`
- `scripts/smolvla_vggflow/bridge_builder.py`
- `scripts/smolvla_vggflow/run_stage.sh`
- `scripts/smolvla_vggflow/config.sh`
- `scripts/smolvla_vggflow/common.sh`
- `scripts/slurm/stage09_final_eval_and_bundle.slurm`
- `scripts/smolvla_vggflow/README.md`

**Tests to add:**
- `scripts/smolvla_vggflow/tests/test_exporter_contiguous_and_images.py`
- `scripts/smolvla_vggflow/tests/test_bridge_quality_gates.py`
- `scripts/smolvla_vggflow/tests/test_stage_wiring_contract.py`

**Execution and orchestration references (no logic rewrite unless needed):**
- `scripts/smolvla_vggflow/smolvla_workflow_launcher.py`
- `scripts/smolvla_vggflow/watch_workflow.py`
- `scripts/smolvla_vggflow/watch_workflow.sh`
- `scripts/slurm/stage00_preflight.slurm` ... `scripts/slurm/stage11_slurm_orchestration.slurm`

**Upstream JEPA references (read-only alignment):**
- `VGG JEPA/jepa-wms/hubconf.py`
- `VGG JEPA/jepa-wms/evals/simu_env_planning/planning/gc_agent.py`
- `VGG JEPA/jepa-wms/evals/simu_env_planning/planning/planning/planner.py`

**Context docs (read-only):**
- `docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`
- `/homes/aa6622/.cursor/plans/jepa-longterm-context.md`
- `/homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md`
- `/homes/aa6622/.cursor/plans/jepa-rollout-algorithm-integration_c18d9a72.plan.md`

---

## Dependency Check Order (Before Any Full DAG Submit)

1. Slurm + queue health: `sbatch`, `squeue`, `scontrol`, `sacct` availability.
2. Runtime env binaries: `SMOLVLA_LEROBOT_ENV_DIR/bin/python`, `SMOLVLA_JEPA_ENV_DIR/bin/python`.
3. CUDA + render base: `torch.cuda.is_available()`, MetaWorld import, MuJoCo EGL render smoke.
4. JEPA hub load smoke (`jepa_smoke_check.py`) with local `VGG JEPA/jepa-wms`.
5. SmolVLA policy load smoke (checkpoint + pre/post processor).
6. Exporter micro-run (2 episodes) with quality gate ON.
7. Bridge micro-run from that export with quality gate ON.
8. Only then submit full branch-parallel DAG.

---

### Task 1: Exporter Stride/Image/Action-Stream Reliability (CEM Primary)

**Files:**
- Modify: `scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py`
- Test: `scripts/smolvla_vggflow/tests/test_exporter_contiguous_and_images.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib.util
import unittest
from pathlib import Path
import numpy as np

MODULE = Path("scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py").resolve()
spec = importlib.util.spec_from_file_location("jepa_export", MODULE)
jepa_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jepa_export)


class DummyEnv:
    def __init__(self, frame):
        self._frame = frame

    def render(self):
        return self._frame


class ExporterContiguousTests(unittest.TestCase):
    def test_negative_stride_input_becomes_contiguous(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)[:, :, ::-1]
        out = jepa_export._as_contiguous_rgb_uint8(frame)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        self.assertEqual(out.shape, (12, 12, 3))

    def test_collect_step_image_uses_render_fallback(self):
        env = DummyEnv(np.ones((8, 8, 3), dtype=np.uint8) * 255)
        img = jepa_export._collect_step_image(obs={}, env=env)
        self.assertEqual(img.shape, (8, 8, 3))

    def test_cem_primary_action_selection(self):
        out = jepa_export._select_executed_action(
            action_wm_cem_first=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            action_smolvla_raw=np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
            env_action_dim=4,
            wm_available=True,
        )
        self.assertEqual(out["policy_source"], "cem_mpc_wm")
        self.assertEqual(len(out["action_executed"]), 4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_exporter_contiguous_and_images.py" -v`
Expected: FAIL with missing `_as_contiguous_rgb_uint8` / `_collect_step_image` / `_select_executed_action`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to jepa_cem_paired_pushv3_export.py
def _as_contiguous_rgb_uint8(arr: Any) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 3:
        raise RuntimeError("bad image shape")
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating) and float(np.max(x)) <= 1.5:
            x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def _collect_step_image(obs: Any, env: Any) -> np.ndarray:
    img = _find_image(obs)
    if img is None:
        img = env.render()
    return _as_contiguous_rgb_uint8(img)


def _select_executed_action(
    *,
    obs: Any,
    env: Any,
    action_wm_cem_first: np.ndarray | None,
    action_smolvla_raw: np.ndarray | None,
    env_action_dim: int,
    wm_available: bool,
) -> dict[str, Any]:
    if action_wm_cem_first is not None:
        a = np.asarray(action_wm_cem_first, dtype=np.float32).reshape(-1)[:env_action_dim]
        return {"action_executed": a.tolist(), "policy_source": "cem_mpc_wm"}
    if action_smolvla_raw is not None:
        a = np.asarray(action_smolvla_raw, dtype=np.float32).reshape(-1)[:env_action_dim]
        return {"action_executed": a.tolist(), "policy_source": "smolvla"}
    a = heuristic_push_action(obs, env)  # use real obs/env in implementation
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:env_action_dim]
    return {"action_executed": a.tolist(), "policy_source": "heuristic_fallback" if wm_available else "heuristic"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_exporter_contiguous_and_images.py" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py scripts/smolvla_vggflow/tests/test_exporter_contiguous_and_images.py
git commit -m "fix(export): cem-primary arbitration and action-stream logging contract"
```

---

### Task 2: Exporter/Bridge Quality Gates (Fail Fast On Bad Data)

**Files:**
- Modify: `scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py`
- Modify: `scripts/smolvla_vggflow/bridge_builder.py`
- Test: `scripts/smolvla_vggflow/tests/test_bridge_quality_gates.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from scripts.smolvla_vggflow import bridge_builder


class BridgeQualityGateTests(unittest.TestCase):
    def test_blank_images_are_rejected(self):
        metrics = {"image_nonblank_ratio": 0.0, "heuristic_fallback_episode_ratio": 1.0, "action_std_mean": 0.0}
        with self.assertRaises(RuntimeError):
            bridge_builder._enforce_quality_gates(metrics, min_image_coverage=0.95, max_heuristic_ratio=0.1, min_action_std=0.02)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_bridge_quality_gates.py" -v`
Expected: FAIL because `_enforce_quality_gates` is not implemented.

- [ ] **Step 3: Write minimal implementation**

```python
# add to bridge_builder.py
def _enforce_quality_gates(metrics: dict[str, float], *, min_image_coverage: float, max_heuristic_ratio: float, min_action_std: float) -> None:
    if metrics["image_nonblank_ratio"] < min_image_coverage:
        raise RuntimeError("image_nonblank_ratio below threshold")
    if metrics["heuristic_fallback_episode_ratio"] > max_heuristic_ratio:
        raise RuntimeError("heuristic_fallback_episode_ratio above threshold")
    if metrics["action_std_mean"] < min_action_std:
        raise RuntimeError("action_std_mean below threshold")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_bridge_quality_gates.py" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smolvla_vggflow/bridge_builder.py scripts/smolvla_vggflow/tests/test_bridge_quality_gates.py scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py
git commit -m "fix(data): add exporter and bridge quality gates for image/action integrity"
```

---

### Task 3: Wire Strict Thresholds Through Stage Scripts + Scope Paths

**Files:**
- Modify: `scripts/smolvla_vggflow/config.sh`
- Modify: `scripts/smolvla_vggflow/common.sh`
- Modify: `scripts/smolvla_vggflow/run_stage.sh`
- Modify: `scripts/slurm/stage09_final_eval_and_bundle.slurm`
- Test: `scripts/smolvla_vggflow/tests/test_stage_wiring_contract.py`

- [ ] **Step 1: Write the failing wiring test**

```python
import unittest
from pathlib import Path


class StageWiringContractTests(unittest.TestCase):
    def test_phase07_passes_quality_flags(self):
        text = Path("scripts/smolvla_vggflow/run_stage.sh").read_text(encoding="utf-8")
        self.assertIn("--max-wm-error-rate", text)
        self.assertIn("--max-policy-error-rate", text)
        self.assertIn("--require-images", text)
        self.assertIn("--execution-policy", text)
        self.assertIn("--store-cem-plan-seq", text)
        self.assertIn("--store-smolvla-action", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_stage_wiring_contract.py" -v`
Expected: FAIL before flags/path scoping are added.

- [ ] **Step 3: Write minimal implementation**

```bash
# config.sh additions (defaults)
SMOLVLA_JEPA_EXPORT_MAX_WM_ERROR_RATE="${SMOLVLA_JEPA_EXPORT_MAX_WM_ERROR_RATE:-0.05}"
SMOLVLA_JEPA_EXPORT_MAX_POLICY_ERROR_RATE="${SMOLVLA_JEPA_EXPORT_MAX_POLICY_ERROR_RATE:-0.05}"
SMOLVLA_JEPA_EXPORT_REQUIRE_IMAGES="${SMOLVLA_JEPA_EXPORT_REQUIRE_IMAGES:-1}"
SMOLVLA_JEPA_EXPORT_EXECUTION_POLICY="${SMOLVLA_JEPA_EXPORT_EXECUTION_POLICY:-cem_primary}"
SMOLVLA_JEPA_EXPORT_STORE_CEM_PLAN_SEQ="${SMOLVLA_JEPA_EXPORT_STORE_CEM_PLAN_SEQ:-1}"
SMOLVLA_JEPA_EXPORT_STORE_SMOLVLA_ACTION="${SMOLVLA_JEPA_EXPORT_STORE_SMOLVLA_ACTION:-1}"
SMOLVLA_BRIDGE_MAX_HEURISTIC_FALLBACK_RATIO="${SMOLVLA_BRIDGE_MAX_HEURISTIC_FALLBACK_RATIO:-0.10}"
SMOLVLA_BRIDGE_MIN_IMAGE_COVERAGE="${SMOLVLA_BRIDGE_MIN_IMAGE_COVERAGE:-0.95}"
SMOLVLA_BRIDGE_MIN_ACTION_STD="${SMOLVLA_BRIDGE_MIN_ACTION_STD:-0.02}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_stage_wiring_contract.py" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smolvla_vggflow/config.sh scripts/smolvla_vggflow/common.sh scripts/smolvla_vggflow/run_stage.sh scripts/slurm/stage09_final_eval_and_bundle.slurm scripts/smolvla_vggflow/tests/test_stage_wiring_contract.py
git commit -m "fix(wiring): pass quality thresholds end-to-end and scope paths per run"
```

---

### Task 4: Add Deterministic Dependency-Order Preflight Script

**Files:**
- Create: `scripts/smolvla_vggflow/preflight_dependency_order.sh`
- Modify: `scripts/smolvla_vggflow/README.md`
- Test: `scripts/smolvla_vggflow/tests/test_preflight_dependency_order.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path


class PreflightOrderTests(unittest.TestCase):
    def test_ordered_checks_are_declared(self):
        text = Path("scripts/smolvla_vggflow/preflight_dependency_order.sh").read_text(encoding="utf-8")
        for needle in ["CHECK_01_SLURM", "CHECK_02_ENVS", "CHECK_03_CUDA_RENDER", "CHECK_04_JEPA", "CHECK_05_SMOLVLA", "CHECK_06_EXPORT", "CHECK_07_BRIDGE"]:
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_preflight_dependency_order.py" -v`
Expected: FAIL because script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail
echo "CHECK_01_SLURM"
which sbatch squeue scontrol >/dev/null
echo "CHECK_02_ENVS"
test -x "${SMOLVLA_LEROBOT_ENV_DIR}/bin/python"
test -x "${SMOLVLA_JEPA_ENV_DIR}/bin/python"
echo "CHECK_03_CUDA_RENDER"
"${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
import torch, metaworld
print(int(torch.cuda.is_available()))
PY
echo "CHECK_04_JEPA"
"${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" scripts/smolvla_vggflow/jepa_smoke_check.py --repo "${SMOLVLA_VGG_REPO}/jepa-wms" --task push-v3 --smoke-steps 2 --pretrained --device auto
echo "CHECK_05_SMOLVLA"
"${SMOLVLA_LEROBOT_ENV_DIR}/bin/python" - <<'PY'
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print("smolvla_ok")
PY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_preflight_dependency_order.py" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smolvla_vggflow/preflight_dependency_order.sh scripts/smolvla_vggflow/README.md scripts/smolvla_vggflow/tests/test_preflight_dependency_order.py
git commit -m "chore(preflight): enforce deterministic dependency check order before submit"
```

---

### Task 5: Controlled Cleanup + Rerun Sequence (No Overwrite)

**Files:**
- Modify: `scripts/smolvla_vggflow/README.md`
- Modify: `docs/superpowers/plans/2026-04-07-jepa-regression-recovery-and-rerun.md`
- Test: N/A (ops protocol)

- [ ] **Step 1: Archive known-bad data (do not delete first pass)**

Run: `mkdir -p artifacts/regression_archive/2026-04-07 && mv /vol/bitbucket/aa6622/.cache/jepa_workflow_egl_20260407T073139Z artifacts/regression_archive/2026-04-07/`
Expected: old exporter output moved out of active source paths.

- [ ] **Step 2: Set one run scope and non-overwrite roots**

Run: `export RUN_ID="$(date -u +%Y%m%d_%H%M%S)" && export SMOLVLA_RUN_SCOPE_ID="${RUN_ID}" && export SMOLVLA_JEPA_EXPORT_OUT="/vol/bitbucket/aa6622/.cache/jepa_exports/run_${RUN_ID}" && export SMOLVLA_JEPA_SOURCE="${SMOLVLA_JEPA_EXPORT_OUT}" && export SMOLVLA_DATA_ROOT="/vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged_v30_runs/run_${RUN_ID}" && export SMOLVLA_FAIL_ON_PATH_REUSE=1`
Expected: all critical outputs are unique to this run id.

- [ ] **Step 3: Run strict preflight + dry-run**

Run: `bash scripts/smolvla_vggflow/preflight_dependency_order.sh && bash scripts/smolvla_vggflow/submit_workflow.sh --dry-run`
Expected: preflight PASS and `runs/workflow_${RUN_ID}.json` with `<unsubmitted>`.

- [ ] **Step 4: Submit branch-parallel and monitor**

Run: `python3 scripts/smolvla_vggflow/smolvla_workflow_launcher.py --submit --branch-parallel --write-json "runs/workflow_${RUN_ID}.json" --parallel-map-out "artifacts/parallel_submission_map_${RUN_ID}.json" && bash scripts/smolvla_vggflow/watch_workflow.sh "runs/workflow_${RUN_ID}.json" -- --poll 90 --max-retries 2 --auto-resubmit`
Expected: stage02||stage03 parallel; stage05||stage06 parallel; no path collisions.

- [ ] **Step 5: Commit runbook updates**

```bash
git add scripts/smolvla_vggflow/README.md docs/superpowers/plans/2026-04-07-jepa-regression-recovery-and-rerun.md
git commit -m "docs(runbook): define non-overwrite recovery rerun sequence"
```

---

## Rerun Matrix (If Normalization/Actions Still Look Wrong)

- Exporter changed only: rerun `stage03` then `stage04` then `stage05/06/08/09`.
- Bridge gating changed only: rerun `stage04` then `stage05/06/08/09`.
- Gate logic changed only: rerun `stage07` then `stage08` then `stage09`.
- Checkpoint selection/pathing changed only: rerun `stage09`.
- Baseline protocol changed: rerun `stage02` and compare with same seed/episodes.

---

## Parallelism Policy And Storage Guarantees

- Safe parallel jobs:
  - `stage02` and `stage03` after `stage01b`.
  - `stage05` and `stage06` after `stage04 + stage02`.
- Must stay serialized:
  - `stage04` after `stage03`.
  - `stage08` after `stage04 + stage07`.
  - `stage09` after `stage05 + stage06 + stage08`.
- Non-overwrite guarantees:
  - Always set `RUN_ID` + `SMOLVLA_RUN_SCOPE_ID`.
  - Force `SMOLVLA_FAIL_ON_PATH_REUSE=1`.
  - Write workflow and parallel map with `${RUN_ID}` suffix.
  - Write export/dataset roots under run-specific directories.

---

## Optional Work (Only If Regression Persists After This Plan)

- **Worth running:** add per-step action-distribution dashboards to bridge summary (quickly confirms normalization quality).
- **Worth running:** add `watch_workflow.py` fallback when `sacct` is unavailable (host-dependent stability).
- **Not worth before first rerun:** major launcher architecture changes; current branch-parallel logic already matches required DAG.

---

## Self-Review

1. **Spec coverage:** exporter stride/image failures, bridge quality failures, dependency order, rerun ordering, and non-overwrite parallel policy are all covered.
2. **Placeholder scan:** no TODO/TBD placeholders; every task has concrete commands/snippets.
3. **Type consistency:** thresholds and env var names are consistent across `config.sh`, `run_stage.sh`, and bridge/export tasks.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-07-jepa-regression-recovery-and-rerun.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
