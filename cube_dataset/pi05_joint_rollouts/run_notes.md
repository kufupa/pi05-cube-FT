# pi05 joint rollouts — agent run

## Phase 0 closure — Q1 layout lock (π0.5 8D)

### Q1 evidence — DROID action layout (blocking)

- **Source:** `external/openpi/src/openpi/training/droid_rlds_dataset.py` `restructure()` — `actions = tf.concat((joint_position or joint_velocity, gripper_position), axis=-1)`.
- **Source:** `external/openpi/src/openpi/policies/droid_policy.py` `DroidOutputs` — returns `actions[:, :8]`.
- **`Pi05DroidPolicy` heuristic** (`src/vla/pi05_droid.py`): `torch.cat([zeros(1,7), grip], -1)` → gripper at **index 7**.

### Locked layout (indices 0–7)

| Index | Role |
|------:|------|
| 0–5 | First six joint commands → map to env `out[0:6]` (UR5 six-DOF arm) |
| 6 | Seventh DROID joint command — **not** used on UR5 cube env (no 7th arm joint) |
| 7 | Gripper → `out[6]` after clip to [0,1] |

**`gripper_a8_index` = 7** for `map_pi05_to_joint7` with arm from `a8[0:6]`.

### Imports / venv

- Python: stable-worldmodel `.venv`; `PYTHONPATH=$PROJECT_ROOT`; `import ogbench, gymnasium, mujoco, torch, jax` OK on executor host.

---

## Phase 1 closure — env + gates

### ManipSpaceEnv control path

- `step()` calls `set_control(action)` then `mj_step`. No superclass `step` overwrite of `ctrl` after `set_control` in this chain (see `ogbench/manipspace/envs/manipspace_env.py`).

### Gate: ctrl persistence

- After `step` with `a7 = [0.1,0,0,0,0,0,0.5]`, `ctrl[gripper]` = 127.5 = `255 * 0.5`; arm `ctrl` tracks clipped `q + joint_scale * arm`.

### Gate: state obs vs qpos

- For `ob_type='states'`, **first 6 elements** of `compute_observation()` equal `_data.qpos[_arm_joint_ids]` (proprio/joint_pos). `build_state_14` uses the same raw arm qpos + normalized gripper; consistent for policy state packing.

---

## Phase 3 closure — local joint rollout

- `--control joint --n 1` path exercised during development; `--smoke-test` joint covers gates.

## Phase 4 closure — output dir

- Production batch dir: `cube_dataset/pi05_joint_rollouts/` with `logs/` for PBS.

## Phase 5 closure — PBS

- Added [`scripts/pbs/run_pi05_joint_rollouts.pbs`](../../scripts/pbs/run_pi05_joint_rollouts.pbs): no `gpu_type=A100`; smoke + `--control joint --n 20`. Submit from `PROJECT_ROOT` with `qsub scripts/pbs/run_pi05_joint_rollouts.pbs` (site queues may still need edits).

## Phase 6 closure — success rate (heuristic checkpoint batch)

- Command: `python -c "..."` on `results.jsonl` → **20 episodes, 0 success, rate 0.0** with `--checkpoint heuristic` (expected: no task-directed behavior).
- Re-run with `PI05_OGBENCH_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_droid` for policy-quality evaluation.

## Phase 7 note (0% success)

- With heuristic policy, 0% is not a joint-interface failure. For OpenPI runs: if still 0%, use plan odd-behavior dossier (`joint_scale` sweep, EE `--control ee` Q3 on same metas, re-check Q1 gripper index).

## Phase 8 — synthesis (compact)

| ID | Status | Evidence |
|----|--------|----------|
| A1–A2, Q1 | Validated | DROID RLDS `concat(joint_7, gripper)`; `gripper_a8_index=7`; `run_notes` Phase 0 |
| A10 | Validated | Smoke: `obs[:6]` == arm `qpos`; `build_state_14` uses same raw arm qpos |
| R12 | Not triggered | `ManipSpaceEnv.step` → `set_control` → `mj_step`; ctrl probe ~127.5 for gripper 0.5 |
| DoD items 1–5 | Met | Env module + registration + driver + PBS script + 20 MP4 + JSONL |
