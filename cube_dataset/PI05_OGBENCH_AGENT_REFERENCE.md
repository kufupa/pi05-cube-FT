# π0.5-DROID × OGBench cube — agent handoff (swap-env guide)

Use this file as context for another agent working in the same `project/` tree. The **closed-loop VLA rollout** is implemented and runnable; swapping **only the environment** means replacing `gym.make(...)`, **proprio/state packing** (`build_state_14`), and **action mapping** (`map_pi05_to_ogbench_*`), plus episode seeds/options if the new env’s reset API differs.

---

## 1. Entry points (run these)

| Path | Role |
|------|------|
| `cube_dataset/run_pi05_ogbench_rollouts.py` | **Main pipeline**: smoke test, batch rollouts, MP4 + `results.jsonl`. |
| `scripts/pbs/run_pi05_ogbench_rollouts.pbs` | **PBS job**: `cd PBS_O_WORKDIR`, `PYTHONPATH=project`, `MUJOCO_GL=glfw`, `xvfb-run`, `stable-worldmodel/.venv/bin/python`, smoke then `--n 20`. |

**Typical local / interactive:**

```bash
cd /path/to/project
export PYTHONPATH=/path/to/project
export MUJOCO_GL=glfw   # or use xvfb; see script docstring
stable-worldmodel/.venv/bin/python cube_dataset/run_pi05_ogbench_rollouts.py --smoke-test
```

**Checkpoint default:** `gs://openpi-assets/checkpoints/pi05_droid`  
**Overrides:** `--checkpoint` or env `PI05_OGBENCH_CHECKPOINT`.

---

## 2. Core code files (dependency order)

| Path | Role |
|------|------|
| `cube_dataset/run_pi05_ogbench_rollouts.py` | **Orchestration**: `gym.make`, `reset_options_from_meta`, `render_chw01`, `build_state_14`, `map_pi05_to_ogbench_scaled_cartesian`, `rollout_one`, CLI, JSONL/MP4 I/O. **This is the primary file to edit for a new env.** |
| `src/vla/pi05_droid.py` | **`Pi05DroidPolicy`**: loads OpenPI from `gs://` or local dir; `act(observation)` → `[1,8]`; builds OpenPI request via `observation_openpi`. |
| `src/envs/droid/observation_openpi.py` | **`build_openpi_droid_request_from_tensors`**: CHW float `[0,1]` + state → flat DROID keys (`observation/exterior_image_1_left`, wrist, joint 7 + gripper 1, `prompt`). Used internally when observation has `obs` + `state` (no `droid_episode_ref`). |
| `external/openpi/` | **OpenPI** (editable dep in root `pyproject.toml`). JAX inference, `policy_config.create_trained_policy`, DROID transforms. |

**Related evaluation (DROID dataset, not OGBench loop):**

| Path | Role |
|------|------|
| `scripts/eval_pi05_droid_real.py` | Offline / RLDS-style eval with `build_openpi_droid_request` from episodes. |
| `scripts/pbs/run_openpi_real_eval.pbs` | Isolated OpenPI venv path for GPU eval. |

**Not used by this OGBench loop:** `src/vla/pi05_libero.py` (LeRobot LIBERO path).

---

## 3. Data layout (current cube run)

| Path | Role |
|------|------|
| `cube_dataset/goal_images/vlm_start_goal/cube-single-v0/sample_XXXX/meta.json` | Per-episode **reset seed**, **reset_options** (`task_info` / `task_id`, `render_goal`, etc.). Rollout script reads these to match VLM start/goal sampling. |
| `cube_dataset/pi05_rollouts/` | **Outputs**: `sample_XXXX.mp4`, append-only `results.jsonl`. Listed in `.gitignore`. |

---

## 4. Observation contract passed to `Pi05DroidPolicy.act`

Built in `rollout_one()` (see `run_pi05_ogbench_rollouts.py`):

```python
observation = {
    "obs": rgb,       # torch (1,3,H,W) float [0,1]
    "state": st,      # torch (1,14) — see build_state_14
    "instruction": INSTRUCTION,  # fixed English string in script
    "timestep": t,
}
```

For a **new env**, keep this dict shape unless you also extend `Pi05DroidPolicy._build_openpi_request` / add a parallel policy wrapper.

---

## 5. What to change for “swap environment only”

All env-specific logic is concentrated in **`run_pi05_ogbench_rollouts.py`**:

1. **`env_id` and `gym.make(..., ob_type=..., render_mode=...)`** — lines ~208, ~307–311 (and smoke).
2. **`build_state_14(env)`** — assumes **`env.unwrapped`** has **`_arm_joint_ids`**, **`_gripper_opening_joint_id`**, **`_data.qpos`** (OGBench `ManipSpaceEnv`). Replace with your env’s proprio → **14-D vector** compatible with how you stuff DROID joint + gripper in `observation_openpi` (first 7 + last 1 used for OpenPI state after transform).
3. **`map_pi05_to_ogbench_scaled_cartesian(a8, raw_env)`** — maps policy **8-D** → env **`step` action** (here **5-D** normalized EE control). Replace with your env’s action space and scaling.
4. **`max_steps` / `--max-steps`** — default `200` matches `cube-single-v0` horizon; set per env.
5. **Meta / reset** — `reset_options_from_meta()` expects JSON like current `meta.json`. If the new env uses different `reset` options, adjust `reset_options_from_meta` and meta schema.
6. **`INSTRUCTION`** — fixed string for cube task; change for new task/language.

**PBS:** `scripts/pbs/run_pi05_ogbench_rollouts.pbs` — update resource lines, paths, and verification counts if you change batch size or output dir.

---

## 6. Python environment (cluster reality)

- **Rollouts use:** `stable-worldmodel/.venv/bin/python` with **`PYTHONPATH=<project root>`** so `import src.vla...` resolves.
- **Packages needed together:** `ogbench`, `gymnasium`, `mujoco`, `torch`, `openpi` (editable from `external/openpi`), `imageio`, `numpy`, plus OpenPI’s JAX stack. The rollout script docstring notes one-off fixes (e.g. `transformers` pin, HF patches) if OpenPI fails to load.
- **Rendering:** PBS sets **`MUJOCO_GL=glfw`** + **`xvfb-run`**; JAX still uses the requested **GPU** for policy.

---

## 7. Outputs (`results.jsonl` fields)

Each line is JSON including at least: `sample_index`, `sample_dir`, `checkpoint`, `uses_openpi`, `env_id`, `bridge_variant`, `success`, `steps`, `terminated`, `truncated`, `video_path`, `reset_seed`; plus `openpi_failed_reason` if OpenPI did not load.

---

## 8. Config parity elsewhere in repo

`configs/droid_single_task_vlaw.yaml` field `base_policy_ckpt: "gs://openpi-assets/checkpoints/pi05_droid"` matches the rollout default; VLAW loop is a **separate** pipeline from this OGBench rollout script.
