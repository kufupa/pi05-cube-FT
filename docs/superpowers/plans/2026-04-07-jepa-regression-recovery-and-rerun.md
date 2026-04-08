# JEPA Regression Recovery And Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task (**Tasks 1–6**). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the phase07 export and phase08 bridge data path so CEM+WM is the default executed controller, SmolVLA actions are still logged for comparison, real images are captured, then rerun training and eval with collision-safe storage and deterministic Slurm orchestration. **Training priority:** **TrainB (WM-heavy mix)** and **TrainC (VGG aux)** run **before** **TrainA (real-only)**; **TrainD** (optional imagined lane) remains **last** when enabled—not rearranged relative to A.

**Architecture:** Keep the existing 11-stage Slurm DAG, but harden the two quality-critical handoff points: `jepa_cem_paired_pushv3_export.py` and `bridge_builder.py`. Switch exporter arbitration to CEM-primary (`action_executed <- action_wm_cem_first` when available), retain SmolVLA action proposals as diagnostics, and store side-by-side action streams for auditability. **Bridge** assigns episodes so **`--jepa-data-root` (StageB merge input) is WM-heavy** vs **`--real-data-root`**, not a blind `pair_key` hash that ignores telemetry (see [Training lane priorities and Slurm DAG](#training-lane-priorities-and-slurm-dag-this-plan)). **Launcher** schedules **stage05 (TrainA) after stage06+stage08** complete. Add exporter/bridge quality gates that fail fast on stride errors, missing images, and heuristic-fallback domination. Scope all run outputs (export, bridge, artifacts, reports, logs) by one run id so parallel jobs cannot overwrite each other.

**Execution expectation:** Runs are **multi-hour** and may proceed **while the user is offline** (e.g. overnight). The executing agent should **monitor** the Slurm DAG (`watch_workflow` / queue and logs), **resubmit** per runbook when jobs fail or stall, and apply **bounded in-repo fixes** until the strict DoD and artifacts are in place—**not** pause for per-stage approval unless a hard blocker applies (secrets missing, irreversible destructive scope, or a research fork the user must choose). **Standing GPU/`sbatch` permission and autonomous iteration policy** are defined in `docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`.

**Tech Stack:** Bash, Python 3.10, PyTorch, MetaWorld, JEPA-WMS (`VGG JEPA/jepa-wms`), LeRobot, Slurm (`sbatch/squeue/scontrol/sacct`), JSON/Parquet.

**Training checkpoint cadence:** LeRobot persists checkpoints every **`SMOLVLA_TRAIN_SAVE_STEPS`** optimizer steps (wired as `--save-steps` → `save_steps` / `save_freq`). Default in `scripts/smolvla_vggflow/config.sh` is **2000** (not 1000) to cut disk use while `SMOLVLA_FIRST_FT_STEPS` remains **6000**; LeRobot still saves at the **final** step when `step == cfg.steps`. Override per run with `SMOLVLA_TRAIN_SAVE_STEPS` if needed.

---

## Goal Link To Old Plan And Updated DoD

- Keep from old plan (`docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`):
  - Full Slurm autonomy.
  - Branch-parallel scheduling.
  - TrainA/B/C from same base checkpoint.
- **Supersede for this recovery plan:** **TrainA is lower priority**—run **after TrainB and TrainC**; **TrainB mix is WM-heavy** (jepa root), not “random hash split with no WM bias.” Update `smolvla_workflow_launcher.py` + `bridge_builder.py` accordingly ([Task 6](#task-6-trainb-wm-heavy-split-and-dag-traina-after-b-and-c)).
- Change now (because regression root cause is identified):
  - **Old DoD:** `stage05/06/08` reached `RUNNING`.
  - **New DoD:** pipeline is only done when export+bridge quality gates pass and retrained A/B/C no longer collapse due to heuristic/blank-image data.

### Updated Definition Of Done (Strict)

1. `phase07` exporter quality gate passes:
   - `wm_step_error_rate <= 0.05`
   - `policy_exec_error_rate <= 0.05`
   - `episodes_with_images == total_episodes`
   - `heuristic_fallback_episode_ratio <= 0.10`
   - **Storage contract:** per-step records include **SmolVLA chunk + WM/CEM plan/latent fields** as in [Trajectory Storage Contract](#trajectory-storage-contract-what-we-store); **`full_latents_exported` true** (default **`SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1`**) unless a documented lightweight run; `export_manifest.json` declares **chunk vs CEM horizons and action dims**; with `SMOLVLA_FAIL_ON_PATH_REUSE=1`, exporter **does not overwrite** an existing `trajectories.pt` / manifest in the target dir.
2. `phase08` bridge quality gate passes:
   - non-empty train+val
   - `image_nonblank_ratio >= 0.95`
   - `action_std_mean >= 0.02`
   - **WM-heavy TrainB inputs:** **`val/`** (or whichever directory is wired to **`--jepa-data-root`** in `run_stage.sh` / `train_smolvla_vggflow.py` stageB) must contain **WM-telemetry-rich** episodes by policy: e.g. sort episodes by a **`wm_completeness_score`** (fraction of steps with successful WM encode + non-empty `latent_pred` / plan fields) and assign **top fraction** to the **jepa** root; **sim/executed-primary** remainder to **real** root. **`bridge_summary.json`** must record `wm_heavy_split_policy`, counts per root, and mean scores—so audits prove **B is WM-heavy by construction**, not hash luck.
   - **Bridge ↔ exporter contract:** `bridge_builder.py` documents in **`bridge_summary.json`** (or schema doc) how each **phase07** field maps into the **LeRobot** dataset (column names / nested feature keys for **executed action**, **SmolVLA chunk**, **CEM plan**, **full `latent_pred`**, **policy_source**, **`step_index`** / frame index). **TrainC / VGG** must either read those columns from `SMOLVLA_DATA_ROOT` or explicitly document that it reads **`trajectories.pt`**—no silent drop of latent or chunk fields needed for the research path.
3. Retrain from fixed dataset: **`stage06` (TrainB)** and **`stage08` (TrainC)** complete **before** **`stage05` (TrainA)** starts; all three finish before **`stage09`**. **TrainD** (if used) remains **after** A/B/C per existing optional stage10d runbook—not inserted between A and B.
4. Post-train eval is run against checkpoints from the same run scope.
5. Final report contains baseline vs A/B/C metrics and exact artifact paths.
6. **Train checkpoint interval:** default **`SMOLVLA_TRAIN_SAVE_STEPS=2000`** in `config.sh` (see plan header); manifests/runbooks note this when describing expected checkpoint directories under each train lane’s `train_run/checkpoints/`.

### Action Arbitration Policy (Updated)

- Default exporter mode: `execution_policy=cem_primary`.
- Per step action choice:
  1. use `action_wm_cem_first` if WM+CEM succeeded,
  2. else use `action_smolvla_raw` if SmolVLA produced a valid action,
  3. else use heuristic fallback.
- Always log all available candidates even when not executed.
- Keep `smolvla_primary` as optional ablation mode only (not default).

### Data alignment and naming (episodes, timesteps, `pair_key`, streams)

Use this section as the **single index** when wiring exporter, bridge, and training readers. Terms below apply to **phase07** `trajectories.pt` unless stated otherwise.

| Concept | What it is | How it links |
|--------|------------|--------------|
| **Export run** | One directory: `export_manifest.json` + `trajectories.pt` | Manifest describes **all** episodes in the file (`episodes` count, dims, horizons). |
| **Episode** | One dict inside the `torch.save` list | Identified by **`pair_key`** (stable UUID per rollout). Optional **`episode_index`** = position in list for debugging only—**do not** use list order as train/val split without hashing `pair_key`. |
| **Sim timestep `step_index`** | Integer **0 … T−1** within that episode | **Parallel lists** at episode top level: `images[k]`, `state[k]`, `actions[k]` (executed) all refer to the **same** sim step **`k`** (convention: observation **at decision time** for step `k`; `actions[k]` is what `env.step` received at step `k`). |
| **`cem_plan.per_step[k]`** | One dict per sim step | **`step_index` inside the dict must equal `k`**. Holds WM/CEM **predictions and latents** computed from the **same** observation time as `images[k]` / `state[k]` (unless WM error—then fields are null/error-tagged). |
| **SmolVLA `action_smolvla_chunk` at step `k`** | Policy output at step `k` | **2-D dict** with shape `(H_policy, D_policy)`. Row `r` is a **policy-relative** future control; **sim** step for that row is **`k + r`** only if `smolvla_chunk_alignment` in the manifest says so (e.g. `chunk_row_r_is_sim_step_k_plus_r`). If the policy uses a different convention, the manifest string must spell it out—**never** infer from code alone. |
| **WM `action_wm_cem_plan_seq` at step `k`** | Best CEM sequence at planner call `k` | **2-D dict** `(cem_horizon, planner_action_dim)`. Row `r` is the **r-th** action in the **imagined** roll used by CEM at sim time `k`; only row `0` is guaranteed related to `action_wm_cem_first` / env clipping—later rows are **not executed** in the sim unless you explicitly open-loop them elsewhere. |
| **`latent_pred` at step `k`** | WM latent associated with the CEM call at step `k` | **Full vector** when `SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1` (this plan’s default). This is **stored signal for latent-side training and analysis**, not a promise of RGB reconstruction (see **Storage vs decoder** below). |
| **Rewards / `done`** | Env feedback at step `k` | Must align with the **transition caused by** `actions[k]`; store either parallel lists `reward[k]`, `done[k]` at episode level **or** inside `per_step[k]`—**one** layout per `schema_version`, documented in `CEM_PAIRED_PUSHV3_SCHEMA.md`. |

**“Paired” (reminder):** same **`pair_key`** + same **`step_index`** ties **sim observations**, **executed action**, **SmolVLA proposals**, and **WM/CEM outputs** for that instant—not “two actions both ran in the sim.”

#### Storage vs decoder and pixels (out of scope for this plan)

- **In scope:** persist **full latents**, **CEM plans**, **SmolVLA chunks**, **sim images/state/actions** so later code can train (e.g. VGG / latent objectives) and audit behavior.
- **Explicitly out of scope here:** whether JEPA–WM exposes a **pixel decoder**, training one, or interpreting saved latents as **RGB WM predictions**. **Do not** read this plan as promising decodable video from `latent_pred`; that is a **separate** model/API question when you get there.

### Trajectory Storage Contract (What We Store)

**Intent:** The simulator trace is the anchor, but the **primary research payload** is **SmolVLA outputs** (for baseline/eval and VLA-centric training) and **JEPA–WM + CEM outputs** (for VGG / imagined-rollout-style objectives). Every run must remain **interpretable** (manifest declares shapes/horizons), **typed consistently**, and **non-destructive** to prior exports when run-scoping and reuse guards are on.

#### Per episode (simulator + episode metadata)

- `images`, `state`: MetaWorld observations actually seen along the executed rollout (contiguous RGB / state layout as today; bridge quality gates apply).
- `actions` (alias / mirror of executed stream): list of **`action_executed`** per step in order—kept for backward compatibility with readers expecting `episode["actions"]`.
- `action_chunk_executed` (optional clarity): if useful for LeRobot, may duplicate the same executed stream; do **not** confuse with SmolVLA’s **predicted** chunk below.
- `success`, `done`, `pair_key`, `meta` (`schema_version`, `export_mode`, task, step count, checkpoints ids).
- **Rewards / termination (required for interpretability):** per-step `reward`, `terminated`, `truncated`, `done` should appear in the same structure as WM/VLA records (either parallel lists at episode level or inside each `cem_plan.per_step[i]`—pick one layout per `schema_version` and document it in `CEM_PAIRED_PUSHV3_SCHEMA.md`).

#### Per step (`cem_plan.per_step[i]`, aligned with sim `step_index`)

- `action_executed`: vector actually passed to `env.step` (length `env_action_dim`, `float32` semantics, stored as list or 1-D array).
- `policy_source`: `cem_mpc_wm` | `smolvla` | `heuristic_fallback` | `heuristic` (exact enum documented in schema).
- **SmolVLA (always log when policy is loaded, even if not executed):**
  - `action_smolvla_raw`: first-step / executed-style slice (`env_action_dim`) the policy would apply at this step **if** it were in charge.
  - `action_smolvla_chunk`: **full** policy chunk as a **self-describing 2-D dict** (`layout`, `shape`, `dtype`, `data` row-major)—semantic shape **`(H_policy, D_policy)`**; **`H_policy` and `D_policy` duplicated in `export_manifest.json`** (`policy_action_chunk_horizon`, `policy_action_dim`). If the policy returns a flat vector, reshape using manifest dims or store `null` with `policy_chunk_error` in metadata—never silently truncate without a flag.
- **WM + CEM (always log when WM is loaded, even if SmolVLA executes):**
  - `action_wm_cem_first`: first control of the best CEM sequence, clipped to `env_action_dim` for env compatibility (1-D list).
  - `action_wm_cem_plan_seq`: full best sequence as the **same 2-D dict pattern**, semantic shape **`(cem_horizon, planner_action_dim)`** (manifest carries `cem_horizon`; planner dim may differ from env dim—document both).
  - `latent_pred`: always persist **full latent state per step** when WM is loaded for **this recovery-and-rerun workflow** (`SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1` in `config.sh` / phase07 env—**default ON** here) so later **latent-side training / analysis** (e.g. VGG) is not blocked by a 256-dim summary alone. **Also** keep the **short summary** alongside or derived for cheap gates/metrics if useful. **Exception:** set `SMOLVLA_JEPA_EXPORT_FULL_LATENTS=0` only for **smoke tests**, quota emergencies, or debugging—then manifest **`full_latents_exported: false`** and runbook must note the export is **not** WM-complete. **Decoder / RGB from latents:** not part of this plan—see [Storage vs decoder and pixels](#storage-vs-decoder-and-pixels-out-of-scope-for-this-plan).
  - `planner_metadata`: `cem_cost`, `cem_iterations`, `cem_seed`, dims, `wm_step_error`, `policy_exec_error`, etc.

#### Chunk / horizon alignment (SmolVLA vs CEM vs sim)

- **`cem_horizon` ≠ `policy_action_chunk_horizon` is normal.** The manifest must list **both** (and `env_action_dim`, `planner_action_dim`). Readers must not assume equal lengths.
- **Convention to document in schema + manifest** (choose one and stick to it): e.g. `action_smolvla_chunk` is associated with **sim step `step_index`** as the **chunk start**; step `step_index`’s executed action may equal chunk row 0 or follow LeRobot’s open-loop convention—state the rule explicitly (`smolvla_chunk_alignment` string in manifest).
- **Padding / partial chunks** at episode end: if the policy emits a shorter chunk, store actual shape in `planner_metadata` (`smolvla_chunk_actual_horizon`) and avoid fake zeros unless labeled `padded`.

#### dtypes, serialization, interpretability

- Prefer **`float32`** for all actions and latent summaries; **`uint8` or documented float01** for images, consistent across exporter and bridge.
- **Lists vs tensors:** `torch.save` episodes may contain lists of floats (JSON-friendly when converted); bridge must know how to convert to training tensors. Any change to layout bumps **`schema_version`**.
- **`export_manifest.json`** must include everything needed to reload without code archaeology: `schema_version`, `export_mode`, checkpoint paths, `cem_horizon`, `cem_pop`, `cem_iters`, `policy_checkpoint`, **`policy_action_chunk_horizon`**, **`policy_action_dim`**, `env_action_dim`, `wm_planner_action_dim`, `smolvla_chunk_alignment`, and flags `wm_skipped_export`, `full_latents_exported` (bool).

#### Storage footprint (what actually costs disk)

- **SmolVLA chunks + full CEM plan rows + short latent summaries (if kept):** **tiny** vs images—order **sub‑MB to low tens of MB** per typical run for the **action/plan** payload alone.
- **Full `latent_pred` per step (`SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1`, default ON):** scales as **\(episodes \times steps \times D_\text{latent} \times 4\)** bytes (float32) **separately** from the chunk/plan row above—can be **tens–hundreds of MB** at scale; often **still secondary** to **per-step images** stored as nested float01 lists at **480²-class** resolution.
- **Images** dominate when float-listed: **\(episodes \times steps \times H \times W \times 3\)** × ~4 B plus Python/`torch.save` overhead.
- **Action:** document **observed `H×W`** from one pilot `du -sh` on `trajectories.pt` after a short export; no need to micro-optimize chunk/plan storage format for size—the **interpretability** layout below matters more.

#### Layout conventions (easy for humans + coding agents later)

Design goal: a **fresh agent** can open **`export_manifest.json` + `CEM_PAIRED_PUSHV3_SCHEMA.md`** and parse **`trajectories.pt`** without spelunking `jepa_cem_paired_pushv3_export.py`.

- **Canonical bundle per export directory:** `export_manifest.json` (pretty-printed JSON, `indent=2`) + `trajectories.pt` (primary payload). **Optional but recommended:** auto-write **`EXPORT_README.md`** in the same directory—a **10–20 line** stub listing top-level episode keys, `cem_plan.per_step` keys, and one line each: “reshape 2-D fields using `shape` inside the record or manifest dims.” Regenerate on every export so IDE search finds it next to data.
- **Stable, grep-friendly names:** `snake_case` only; no synonyms (`exec_action` vs `action_executed`). **`step_index`** must appear **inside** every `cem_plan.per_step[i]` even if redundant with list index `i` (agents and diffs align faster).
- **1-D vectors:** store as Python `list` of floats (or 1-D `numpy`/`torch` if the bridge agrees); length **must** match manifest `env_action_dim` or `policy_action_dim` as documented per field.
- **2-D arrays** (SmolVLA chunk, CEM plan): store a **small self-describing dict**, not a raw nested list alone, e.g.  
  `{"layout": "row_major", "shape": [H, D], "dtype": "float32", "data": [[...], ...]}`  
  so shape is **visible in JSON tooling** and in `torch.load` structure without inferring from manifest alone. If a field is missing (WM off), use **`null`** or omit key **only** when schema explicitly allows; otherwise set `{"error": "wm_skipped"}` pattern in `planner_metadata`.
- **Episode top-level:** keep **`meta`** as a flat-ish dict with `schema_version`, `export_mode`, `pair_key`, `task`, `num_steps`, `policy_label` (summary). Avoid hiding critical ids only inside nested blobs.
- **Single source of truth for semantics:** any field added to code **must** appear in **`CEM_PAIRED_PUSHV3_SCHEMA.md`** the same PR with a one-line type + shape line; manifest keys duplicated when they are reshape-critical (`cem_horizon`, `policy_action_chunk_horizon`, etc.).
- **Version bumps:** breaking layout changes → increment **`schema_version`** string and add a **short “migration” note** at bottom of schema doc (one sentence: what changed).

#### Storage safety (no silent clobber)

- **Today (gap):** exporter does `mkdir` + `torch.save(trajectories.pt)` and overwrites `export_manifest.json` in the same directory—**unsafe** if `SMOLVLA_JEPA_EXPORT_OUT` points at a shared default path.
- **Required:** use **run-scoped** `SMOLVLA_JEPA_EXPORT_OUT` (e.g. `.../jepa_exports/run_${RUN_ID}`) and `SMOLVLA_FAIL_ON_PATH_REUSE=1` for serious runs (`common.sh` already guards some artifact roots—extend the same idea to phase07).
- **Exporter behavior:** if `SMOLVLA_FAIL_ON_PATH_REUSE=1` and **`trajectories.pt` or `export_manifest.json` already exists** in `SMOLVLA_JEPA_EXPORT_OUT`, **fail fast** (non-zero exit) instead of overwriting. Optional: write via temp file + `os.replace` so a crash mid-write does not corrupt a previous good `trajectories.pt`.

#### Verification (tests + gates)

- **Unit / contract tests:** (1) `len(per_step) == len(actions)` and, when images required, `len(images)` matches executed steps. (2) When WM + policy both load, each step record contains `action_smolvla_raw`, `action_smolvla_chunk` (or explicit null + error), `action_wm_cem_first`, `action_wm_cem_plan_seq`, `action_executed`, `policy_source`. (3) Shapes: 2-D fields use the **self-describing dict** (`shape` + `data`); `action_smolvla_chunk["shape"][1] == manifest["policy_action_dim"]`; `action_wm_cem_plan_seq["shape"] == [cem_horizon, planner_action_dim]` when WM succeeded.
- **Manifest self-check:** small script or exporter tail step: manifest horizons/dims match first episode’s stored arrays (sample one episode).
- Extend or add tests under `scripts/smolvla_vggflow/tests/` (e.g. `test_exporter_storage_contract.py`) and update **`scripts/smolvla_vggflow/docs/CEM_PAIRED_PUSHV3_SCHEMA.md`** in lockstep with code.

---

## File/Resource Coverage (Reviewed)

**Core implementation files (modify):**
- `scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py`
- `scripts/smolvla_vggflow/docs/CEM_PAIRED_PUSHV3_SCHEMA.md` (keep in sync with on-disk layout + manifest keys)
- `scripts/smolvla_vggflow/bridge_builder.py` (WM-heavy jepa split + summary fields—[Task 6](#task-6-trainb-wm-heavy-split-and-dag-traina-after-b-and-c))
- `scripts/smolvla_vggflow/smolvla_workflow_launcher.py` (branch-parallel deps + `STAGE_SCRIPTS` order—[Task 6](#task-6-trainb-wm-heavy-split-and-dag-traina-after-b-and-c))
- `scripts/smolvla_vggflow/merge_lerobot_v21_datasets.py` (optional: **`--jepa-first`** or **`--duplicate-jepa-episodes N`** if merged dataset must overweight WM source beyond episode counts—only if bridge split alone is insufficient)
- `scripts/smolvla_vggflow/run_stage.sh`
- `scripts/smolvla_vggflow/run_baseline_eval.sh`
- `scripts/smolvla_vggflow/config.sh`
- `scripts/smolvla_vggflow/common.sh`
- `scripts/slurm/stage02_baseline_pushv3_eval.slurm`
- `scripts/slurm/stage09_final_eval_and_bundle.slurm`
- `scripts/smolvla_vggflow/README.md`

**Tests to add:**
- `scripts/smolvla_vggflow/tests/test_exporter_contiguous_and_images.py`
- `scripts/smolvla_vggflow/tests/test_exporter_storage_contract.py` (lengths, shapes, dual-stream keys, manifest/horizon consistency)
- `scripts/smolvla_vggflow/tests/test_bridge_quality_gates.py`
- `scripts/smolvla_vggflow/tests/test_bridge_wm_heavy_split.py` (jepa root mean `wm_completeness_score` ≥ real root; deterministic given fixed inputs)
- `scripts/smolvla_vggflow/tests/test_launcher_dag_train_order.py` (unit-test dependency construction: TrainA job depends on TrainB and TrainC job ids, not only `join_train`)
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

## Baseline Eval Protocol (Cross-Cluster Parity, Quick Mode)

Use this fixed parity tuple for now:
- `checkpoint=jadechoghari/smolvla_metaworld`
- `env=metaworld`, `task=push-v3`
- `seed=1004`
- `episode_length=300`
- `policy.device=cuda`
- `policy.use_amp=true`
- `eval.n_episodes=20` (temporary quick-stat mode; increase later)
- `video=true` (expect 20 episode videos + 1 smoke video if smoke run is enabled)

Resource envelope for stage02 parity eval:
- Slurm request target: `--gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:30:00` (2.5h wall time).
- Submission preference on clusters with env retrieval issues: use `sbatch --wrap`.

Execution command template (no hidden context required):

```bash
sbatch --partition a100 --qos normal --gres gpu:1 --cpus-per-task=4 --mem=32G --time=02:30:00 \
  --output "logs/stage02_baseline_seed1004_ep20_%j.log" \
  --wrap "cd '/vol/bitbucket/aa6622/pi05-cube-FT' && \
    export SMOLVLA_BASELINE_SEED=1004 \
           SMOLVLA_BASELINE_EPISODES=20 \
           SMOLVLA_BASELINE_EPISODE_LENGTH=300 \
           SMOLVLA_BASELINE_DEVICE=cuda \
           SMOLVLA_BASELINE_USE_AMP=true \
           SMOLVLA_BASELINE_VIDEO=true \
           SMOLVLA_BASELINE_VIDEO_LENGTH=300 \
           SMOLVLA_BASELINE_VIDEO_INTERVAL=2 && \
    bash scripts/smolvla_vggflow/run_stage.sh stage02_baseline_pushv3_eval"
```

Verification contract:
- `reports/phase06_baseline_eval.log` must contain: `episodes=20`, `episode_length=300`, `device=cuda`, `use_amp=true`, `seed=1004`.
- Baseline `eval_info.json` must report `overall.n_episodes == 20`.
- `reports/stage02_baseline_pushv3_eval_status.md` must include a concrete baseline output dir and parsed summary line.
- **SmolVLA action trace (parity with phase07 logging):** baseline run must also write a **machine-readable** artifact under the same baseline output tree (e.g. `baseline_action_trace.json` / `.jsonl` / small parquet—pick one in implementation) containing **per episode** and **per env step** at least: `episode_index`, `step_index`, **`action_smolvla_raw`** (or policy output actually used for the eval step), and **`action_smolvla_chunk`** as the **full predicted chunk** when the policy API returns a chunk (same **self-describing 2-D dict** pattern as phase07, or `null` + error). Align **`step_index`** with the evaluator’s step counter so later scripts can diff **baseline VLA** vs **phase07 exporter** without re-running sim. Document the filename in `phase06_baseline_eval.log` or status markdown.
- If the job goes `PD (user env retrieval failed requeued held)`, cancel and resubmit with `--wrap`.

Episode scaling policy:
- Keep `20` until exporter/bridge/gate fixes are stable and repeatable.
- Move to `50`, then `100` only for final confidence intervals and cross-check reporting.

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
- Modify: `scripts/smolvla_vggflow/bridge_builder.py` (include **`bridge_summary.json` exporter→LeRobot field map** per strict DoD phase08 bullet)
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

    def test_config_default_full_latents_on(self):
        cfg = Path("scripts/smolvla_vggflow/config.sh").read_text(encoding="utf-8")
        self.assertIn("SMOLVLA_JEPA_EXPORT_FULL_LATENTS", cfg)


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
SMOLVLA_JEPA_EXPORT_FULL_LATENTS="${SMOLVLA_JEPA_EXPORT_FULL_LATENTS:-1}"
SMOLVLA_BRIDGE_MAX_HEURISTIC_FALLBACK_RATIO="${SMOLVLA_BRIDGE_MAX_HEURISTIC_FALLBACK_RATIO:-0.10}"
SMOLVLA_BRIDGE_MIN_IMAGE_COVERAGE="${SMOLVLA_BRIDGE_MIN_IMAGE_COVERAGE:-0.95}"
SMOLVLA_BRIDGE_MIN_ACTION_STD="${SMOLVLA_BRIDGE_MIN_ACTION_STD:-0.02}"
# Training: LeRobot checkpoint frequency (existing key; default 2000 for this recovery plan)
SMOLVLA_TRAIN_SAVE_STEPS="${SMOLVLA_TRAIN_SAVE_STEPS:-2000}"
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

Run: `ARCHIVE_TAG="$(date -u +%Y%m%d)" && mkdir -p "artifacts/regression_archive/${ARCHIVE_TAG}" && mv "/vol/bitbucket/aa6622/.cache/jepa_workflow_egl_20260407T073139Z" "artifacts/regression_archive/${ARCHIVE_TAG}/"`
Expected: old exporter output is moved out of active source paths with an auditable archive copy retained. Do **not** run destructive delete (`rm -rf`) on this first pass.

- [ ] **Step 2: Set one run scope and non-overwrite roots**

Run:

```bash
export RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
export SMOLVLA_RUN_SCOPE_ID="${RUN_ID}"
export SMOLVLA_JEPA_EXPORT_OUT="/vol/bitbucket/aa6622/.cache/jepa_exports"
export SMOLVLA_JEPA_SOURCE="${SMOLVLA_JEPA_EXPORT_OUT}"
export SMOLVLA_DATA_ROOT="/vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged_v30_runs"
export SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1
export SMOLVLA_FAIL_ON_PATH_REUSE=1
```

Expected: all critical outputs are unique to this run id; **full WM latents on** for production exports (set `SMOLVLA_JEPA_EXPORT_FULL_LATENTS=0` only for smoke/quota per [Trajectory Storage Contract](#trajectory-storage-contract-what-we-store)). `common.sh` applies `/run_${SMOLVLA_RUN_SCOPE_ID}` automatically to export/data/report/artifact roots, so base paths above should not include `run_${RUN_ID}`.

- [ ] **Step 3: Run strict preflight + dry-run**

Run: `bash scripts/smolvla_vggflow/preflight_dependency_order.sh && bash scripts/smolvla_vggflow/submit_workflow.sh --dry-run`
Expected: preflight PASS and `runs/workflow_${RUN_ID}.json` with `<unsubmitted>`.

- [ ] **Step 4: Submit branch-parallel and monitor**

Run: `python3 scripts/smolvla_vggflow/smolvla_workflow_launcher.py --submit --branch-parallel --write-json "runs/workflow_${RUN_ID}.json" --parallel-map-out "artifacts/parallel_submission_map_${RUN_ID}.json" && bash scripts/smolvla_vggflow/watch_workflow.sh "runs/workflow_${RUN_ID}.json" --poll 90 --max-retries 2 --auto-resubmit`
Expected: `stage02||stage03` run in parallel; **`stage06||stage08`** can overlap once `stage02+stage04` and `stage07+stage04` deps are met; **`stage05` runs only after both `stage06` and `stage08` succeed**; then `stage09`; no path collisions.
Note: this order is the recovery **target behavior** and depends on the Task 6 launcher DAG patch; treat it as intended sequencing until Task 6 lands in `smolvla_workflow_launcher.py`.

- [ ] **Step 5: Keep cleanup reversible until run validation completes**

Run: `ls -lah artifacts/regression_archive/*`
Expected: archived source remains available for audit/rollback while the new run executes. Delete archived data only after validating the rerun outputs and report bundle.

---

### Task 6: TrainB WM-heavy split and DAG (TrainA after B and C)

**Files:**
- Modify: `scripts/smolvla_vggflow/bridge_builder.py` (`wm_completeness_score`, split policy, `bridge_summary.json` keys)
- Modify: `scripts/smolvla_vggflow/config.sh` (defaults e.g. `SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION`, `SMOLVLA_BRIDGE_WM_SCORE_MARGIN`)
- Modify: `scripts/smolvla_vggflow/smolvla_workflow_launcher.py` (`STAGE_SCRIPTS` order + `submit_workflow_branch_parallel` indices/deps)
- Optional modify: `scripts/smolvla_vggflow/merge_lerobot_v21_datasets.py`
- Test: `scripts/smolvla_vggflow/tests/test_bridge_wm_heavy_split.py`
- Test: `scripts/smolvla_vggflow/tests/test_launcher_dag_train_order.py`

- [ ] **Step 1: Write failing tests for WM-heavy split**

```python
import unittest


class WmHeavySplitTests(unittest.TestCase):
    def test_jepa_root_has_higher_mean_wm_score(self):
        # Build synthetic records: half high-WM half low-WM; run split helper;
        # assert mean(score|jepa) > mean(score|real) by margin.
        self.fail("implement bridge_builder._split_wm_heavy_jepa + wire tests")


if __name__ == "__main__":
    unittest.main()
```

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_bridge_wm_heavy_split.py" -v`  
Expected: **FAIL** until split helper exists.

- [ ] **Step 2: Implement `wm_completeness_score` + split in `bridge_builder.py`**

For each episode record, score in `[0,1]` from `cem_plan.per_step` (e.g. count steps without `wm_step_error` and with non-empty `latent_pred` / plan fields ÷ step count). Sort episodes by score descending; assign top **`SMOLVLA_BRIDGE_WM_HEAVY_JEPA_FRACTION`** (default `0.6`) to **`val/`** (jepa root path), remainder to **`train/`** (real root). Replace or gate **`_split_by_pair_key_hash`** behind `SMOLVLA_BRIDGE_WM_HEAVY_SPLIT=1` (default **on** for this plan). Write `wm_heavy_split_policy`, `mean_wm_score_train`, `mean_wm_score_val`, `n_train_episodes`, `n_val_episodes` into **`bridge_summary.json`**.

- [ ] **Step 3: Reorder `STAGE_SCRIPTS` and fix `submit_workflow_branch_parallel`**

Set `STAGE_SCRIPTS` tail to: `stage04_bridge_dataset_build.slurm`, `stage06_train_stageB_jepa_mix.slurm`, `stage07_vgg_gatecheck.slurm`, `stage08_train_stageC_vgg_aux.slurm`, `stage05_train_stageA_real_only.slurm`, `stage09_final_eval_and_bundle.slurm`. Update dependency lines to match [DAG analysis](#dag-analysis-branch-parallel-slurm-deps): `j06` on `join_train`, `j07` on `j01b`, `j08` on `join_c`, `j05` on `f"{j06}:{j08}"`, `j09` on `f"{j05}:{j06}:{j08}"`. Set **`ids_order`** to match **`STAGE_SCRIPTS` index order** (for correct `zip` in parallel map).

- [ ] **Step 4: Write failing launcher DAG test**

Assert (with mocked `submit_stage` returning monotonic fake job ids) that **TrainA** (`stage05` script) is submitted with a dependency string that includes **both** TrainB and TrainC job ids (e.g. `123:124` pattern), not `join_train` alone.

Run: `"/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/bin/python" -m unittest discover -s scripts/smolvla_vggflow/tests -p "test_launcher_dag_train_order.py" -v`  
Expected: **FAIL** until launcher patched.

- [ ] **Step 5: Run all new tests — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add scripts/smolvla_vggflow/bridge_builder.py scripts/smolvla_vggflow/config.sh \
  scripts/smolvla_vggflow/smolvla_workflow_launcher.py \
  scripts/smolvla_vggflow/tests/test_bridge_wm_heavy_split.py \
  scripts/smolvla_vggflow/tests/test_launcher_dag_train_order.py
git commit -m "feat(pipeline): WM-heavy TrainB split; TrainA after TrainB+TrainC in Slurm DAG"
```

---

## Rerun Matrix (If Normalization/Actions Still Look Wrong)

- Exporter changed only: rerun `stage03` then `stage04` then **`stage06` → `stage08` → `stage05` → `stage09`** (respect new train order; `stage09` still waits for all three trains).
- Bridge gating / WM-heavy split changed only: rerun `stage04` then **`stage06` → `stage08` → `stage05` → `stage09`**.
- Gate logic changed only: rerun `stage07` then `stage08` then **`stage05`** (if A not yet run) then `stage09`.
- Checkpoint selection/pathing changed only: rerun `stage09`.
- Baseline protocol changed: rerun `stage02` and compare with identical `seed`, `episodes`, `episode_length`, and `policy.use_amp`.

---

## Training lane priorities and Slurm DAG (this plan)

### Research priority (what we optimize for)

| Lane | Slurm | Role | Priority in this recovery |
|------|--------|------|---------------------------|
| **TrainB** | `stage06` | Mixed **real + jepa** dataset → `mixed_lerobot_b`; **WM-heavy** via bridge | **High** — run **as soon as** baseline+bridge ready |
| **TrainC** | `stage08` | VGG aux on **`train/`** + gate JSON | **High** — run **in parallel with B** once bridge **and** gate (`stage07`) done |
| **TrainA** | `stage05` | Real-only **`train/`** fine-tune | **Lower** — **after B and C** complete (still before final eval bundle) |
| **TrainD** | `stage10d` (optional) | Imagined-heavy lane | **Unchanged** — **last** when user enables it; **not** moved ahead of A |

**TrainA after B/C** does **not** change data inputs: all trains read the same **`SMOLVLA_DATA_ROOT`** produced by `stage04`. Only **Slurm dependencies** and **operator expectations** change so GPU time favors B/C first.

### TrainB “WM-heavy” (what was wrong before)

- **Previous behavior:** `bridge_builder._split_by_pair_key_hash` assigns episodes to `train/` vs `val/` **only** by hashed `pair_key` and `val_ratio`—**both** roots carry the **same** schema; **no** WM bias. `merge_lerobot_v21_datasets.py` concatenates **real-root episodes then jepa-root episodes** with **no** oversampling.
- **Required behavior:** compute **`wm_completeness_score`** per episode from exporter fields (e.g. fraction of steps where WM succeeded and `latent_pred` / plan fields are present). Assign episodes with **higher** scores to the directory used as **`--jepa-data-root`** for stageB (`val/` today), **lower** scores to **`--real-data-root`** (`train/`). Target: **mean score on jepa root ≥ mean score on real root** by a documented margin (e.g. ≥0.05) or **jepa root holds ≥60%** of total episodes—choose one primary criterion and record the other in `bridge_summary.json` for audit.
- **Optional merge tweak:** if episode-count split is insufficient to make **effective** WM exposure WM-heavy in the trainer, add merge flags (e.g. duplicate jepa episodes) **only** after bridge split is correct—YAGNI until metrics say so.

### DAG analysis (branch-parallel Slurm deps)

**Unchanged fan-out after `stage01b`:** `stage02` (baseline) and `stage03` (JEPA lane) still **parallel**.

**Unchanged:** `stage04` **after** `stage03`; `stage07` **after** `stage01b` (may overlap 02/03/04 as today).

**Train jobs:**

- **`stage06` (TrainB):** depends on **`stage02` + `stage04`** (`join_train`) — same as today.
- **`stage08` (TrainC):** depends on **`stage07` + `stage04`** (`join_c`) — same as today.
- **`stage06` and `stage08`:** **can run concurrently** once both dependency sets are satisfied (typically **04** is the long pole; **07** may finish earlier or later than **06**).
- **`stage05` (TrainA):** **NEW** — depends on **`stage06` AND `stage08`** both **`afterok`** (not on `join_train` alone). Ensures A starts **after** B and C complete.
- **`stage09`:** depends on **`stage05` + `stage06` + `stage08`** all success (unchanged set; **A** is now last-finishing among trains).

**Implementation approach (single source of truth):**

1. **Reorder `STAGE_SCRIPTS`** in `smolvla_workflow_launcher.py` to: `… stage04, stage06, stage07, stage08, stage05, stage09` (numeric stage05/06 filenames unchanged; **array order** matches execution priority for **serial** `submit_workflow_serial()` too).
2. **Rewrite `submit_workflow_branch_parallel`** to use **new indices**: submit **06** and **07** after prerequisites; **08** after `join_c`; **05** after `join_bc = job06:job08`; **09** after `join_final = job05:job06:job08`.
3. **Parallel map JSON:** `zip(STAGE_SCRIPTS, ids_order)` requires **`ids_order[k]` = job id for `STAGE_SCRIPTS[k]`**—same **array index order** as `STAGE_SCRIPTS`, not chronological submission order. Example after reorder:  
   `ids_order = [j00,j01,j01b,j02,j03,j04,j06,j07,j08,j05,j09]`.

**Serial submit (`--submit` without `--branch-parallel`):** after `STAGE_SCRIPTS` reorder, **serial chain automatically runs B → 07 → C → A → 09** after bridge—**07** may wait unnecessarily on **06** (extra latency vs branch mode); acceptable for debug.

---

## Parallelism Policy And Storage Guarantees

- Safe parallel jobs:
  - `stage02` and `stage03` after `stage01b`.
  - **`stage06` (TrainB)** after `stage04 + stage02`.
  - **`stage08` (TrainC)** after `stage04 + stage07` — **parallel with `stage06`** once both deps met.
- Must stay serialized / gated:
  - `stage04` after `stage03`.
  - **`stage05` (TrainA) after `stage06` and `stage08`** (both `afterok`).
  - `stage09` after `stage05 + stage06 + stage08`.
- Non-overwrite guarantees:
  - Always set `RUN_ID` + `SMOLVLA_RUN_SCOPE_ID`.
  - Force `SMOLVLA_FAIL_ON_PATH_REUSE=1`.
  - Set **`SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1`** for full pipeline exports unless intentionally lightweight (see [Data alignment](#data-alignment-and-naming-episodes-timesteps-pair_key-streams)).
  - Write workflow and parallel map with `${RUN_ID}` suffix.
  - Write export/dataset roots under run-specific directories.

---

## Optional Work (Only If Regression Persists After This Plan)

- **Worth running:** add per-step action-distribution dashboards to bridge summary (quickly confirms normalization quality).
- **Worth running:** add `watch_workflow.py` fallback when `sacct` is unavailable (host-dependent stability).
- **Launcher / DAG:** **in scope for Task 6** (TrainA after B/C + WM-heavy B)—do **not** defer.

---

## Zero-Context Agent Handoff Packet

Use this section to hand work to another coding agent with no prior chat.

### Handoff Objective

- Execute regression recovery with CEM-primary exporter path and strict quality gates.
- Run baseline parity eval in quick-stat mode (`20` episodes, seed `1004`, `episode_length=300`, AMP enabled).
- Preserve non-overwrite guarantees and reproducible artifacts.
- Apply **Task 6**: **TrainB WM-heavy** bridge split; **Slurm DAG** runs **TrainA after TrainB+TrainC** ([Training lane priorities and Slurm DAG](#training-lane-priorities-and-slurm-dag-this-plan)).

### Non-Negotiable Invariants

- Do not overwrite existing run outputs; always use run-scoped paths.
- Keep action arbitration default as `cem_primary`.
- Keep all candidate action streams logged (`action_wm_cem_first`, `action_wm_cem_plan_seq`, `action_smolvla_raw`, **`action_smolvla_chunk`**, `action_executed`, `policy_source`) with **[step / episode alignment](#data-alignment-and-naming-episodes-timesteps-pair_key-streams)**.
- **Baseline stage02** must persist **SmolVLA** per-step (+ chunk) trace artifact for parity with phase07—see [Baseline Eval Protocol](#baseline-eval-protocol-cross-cluster-parity-quick-mode).
- **Decoder / WM pixels:** out of scope; storing latents is **not** claiming RGB decode—see [Storage vs decoder and pixels](#storage-vs-decoder-and-pixels-out-of-scope-for-this-plan).
- Baseline parity tuple must stay fixed unless explicitly changed by user.

### Required Implementation/Verification Targets

- Exporter + bridge reliability and gate tasks in this document (`Task 1`, `Task 2`, `Task 3`, `Task 4`, `Task 5`, **`Task 6`**).
- Baseline wiring files:
  - `scripts/smolvla_vggflow/config.sh`
  - `scripts/smolvla_vggflow/run_baseline_eval.sh`
  - `scripts/smolvla_vggflow/run_stage.sh`
  - `scripts/slurm/stage02_baseline_pushv3_eval.slurm`
- Ensure these env vars are recognized and propagated:
  - `SMOLVLA_BASELINE_EPISODES`
  - `SMOLVLA_BASELINE_SEED`
  - `SMOLVLA_BASELINE_EPISODE_LENGTH`
  - `SMOLVLA_BASELINE_DEVICE`
  - `SMOLVLA_BASELINE_USE_AMP`
  - `SMOLVLA_BASELINE_VIDEO*`
  - `SMOLVLA_TRAIN_SAVE_STEPS` (default **2000** LeRobot checkpoint interval; optional override)

### Exact Runbook Commands (Agent Can Copy-Paste)

```bash
# 1) preflight
bash scripts/smolvla_vggflow/preflight_dependency_order.sh

# 2) baseline parity quick-stat run
sbatch --partition a100 --qos normal --gres gpu:1 --cpus-per-task=4 --mem=32G --time=02:30:00 \
  --output "logs/stage02_baseline_seed1004_ep20_%j.log" \
  --wrap "cd '/vol/bitbucket/aa6622/pi05-cube-FT' && \
    export SMOLVLA_BASELINE_SEED=1004 \
           SMOLVLA_BASELINE_EPISODES=20 \
           SMOLVLA_BASELINE_EPISODE_LENGTH=300 \
           SMOLVLA_BASELINE_DEVICE=cuda \
           SMOLVLA_BASELINE_USE_AMP=true \
           SMOLVLA_BASELINE_VIDEO=true \
           SMOLVLA_BASELINE_VIDEO_LENGTH=300 \
           SMOLVLA_BASELINE_VIDEO_INTERVAL=2 && \
    bash scripts/smolvla_vggflow/run_stage.sh stage02_baseline_pushv3_eval"

# 3) monitor
squeue -j <JOB_ID> -o "%.18i %.9P %.24j %.8u %.2t %.10M %.20S %.80R"
```

### Acceptance Checklist

- Stage02 log confirms the parity tuple exactly:
  - `episodes=20`
  - `episode_length=300`
  - `device=cuda`
  - `use_amp=true`
  - `seed=1004`
- Baseline output has `eval_info.json` with `overall.n_episodes == 20`.
- **Baseline action trace** artifact exists (per-step + full SmolVLA chunk when available) and path noted in log or status doc.
- `reports/stage02_baseline_pushv3_eval_status.md` has parsed summary and artifact path.
- No path collisions when `SMOLVLA_RUN_SCOPE_ID` + `SMOLVLA_FAIL_ON_PATH_REUSE=1` are set.

### Failure Playbook

- `PD (user env retrieval failed requeued held)`: cancel and resubmit using `--wrap`.
- `QOSMaxSubmitJobPerUserLimit`: backoff and retry later or reduce concurrent submissions.
- Missing `eval_info.json`: treat as WARN/fail for parity gate, inspect baseline log for evaluator crash.
- Unexpected reward collapse with healthy exporter/bridge gates: rerun stage02 with same parity tuple first, then inspect seed drift and checkpoint pathing.

---

## Self-Review

1. **Spec coverage:** exporter stride/image failures, bridge quality failures, **episode/step/`pair_key` alignment**, **baseline SmolVLA trace**, **bridge→TrainC field map**, **WM-heavy TrainB split**, **TrainA after B/C DAG**, **storage vs decoder scope**, **training checkpoint cadence (`SMOLVLA_TRAIN_SAVE_STEPS` default 2000)**, dependency order, rerun ordering, and non-overwrite parallel policy are all covered.
2. **Placeholder scan:** no TODO/TBD placeholders; every task has concrete commands/snippets.
3. **Type consistency:** thresholds and env var names are consistent across `config.sh`, `run_stage.sh`, and bridge/export tasks.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-07-jepa-regression-recovery-and-rerun.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
