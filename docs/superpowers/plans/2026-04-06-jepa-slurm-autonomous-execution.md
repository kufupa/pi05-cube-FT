# JEPA + SmolVLA Autonomous Slurm Execution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (recommended) or `superpowers:subagent-driven-development` to run this document **task-by-task** with review checkpoints. Prefer the **same** Cursor session that already holds context from the companion plans below. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) **Implement and land first** a **push-v3** data path where **real executed rollouts** (simulator or physical—same task **push-v3**) are **paired** with **CEM-planned / JEPA–WM latent rollouts** under one schema; (2) wire [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh) phase07 + [`bridge_builder.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/bridge_builder.py) so **only** that exporter feeds the bridge; (3) **then** run the Slurm DAG (branch-parallel preferred), watch with auto-resubmit, fix bounded breakages, and report to [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md) / [`jepa-longterm-context.md`](file:///homes/aa6622/.cursor/plans/jepa-longterm-context.md).

**Architecture:** **No fallback** trajectory mode: the legacy [`jepa_metaworld_rollout_export.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/jepa_metaworld_rollout_export.py) path (**random** `action_space.sample()`) is **not part of this plan**—remove its use from phase07 in code during **Task 0**, and do not document it as an option. The **only** supported export is the **paired real + CEM latent** pipeline (Task 0). Slurm orchestration: single **primary** agent owns `RUN_ID` and submit; [`smolvla_workflow_launcher.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/smolvla_workflow_launcher.py) branch-parallel DAG; [`watch_workflow.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.sh) / [`watch_workflow.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.py) with `--auto-resubmit`.

**Tech Stack:** Bash, Slurm (`sbatch`, `scontrol`, `squeue`, `sacct` when accounting DB is reachable), Python 3, PyTorch, JEPA-WM, Meta-World **push-v3** (for real executed traces), repo root [`pi05-cube-FT`](file:///vol/bitbucket/aa6622/pi05-cube-FT), env from [`config.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh).

**File map (writing-plans):** Launcher, watcher (`*.py` + `.sh`), [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh), and [`scripts/slurm/*.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/) — expanded table in [File and responsibility map](#file-and-responsibility-map) below.

---

## Companion plans and sources of truth

- **Milestone “training started”** (definition of done, transcript audit, blocker list, mermaid flow): [`pipeline_to_training_start_b778a44c.plan.md`](file:///homes/aa6622/.cursor/plans/pipeline_to_training_start_b778a44c.plan.md).
- **Research intent + TrainA/B/C/D lanes** (YAML `status:` may be stale): [`jepa-rollout-algorithm-integration_c18d9a72.plan.md`](file:///homes/aa6622/.cursor/plans/jepa-rollout-algorithm-integration_c18d9a72.plan.md) — top section *How to actually run Slurm and monitor* links back here.
- **First-person checklist / handoff:** [`slurm_execution_master_prompt_e4ece88c.plan.md`](file:///homes/aa6622/.cursor/plans/slurm_execution_master_prompt_e4ece88c.plan.md).
- **Status template:** [`jepa-autonomous-run-report-template.md`](file:///homes/aa6622/.cursor/plans/jepa-autonomous-run-report-template.md).
- **What actually landed in code (vs stale plan YAML):** [`jepa-longterm-context.md`](file:///homes/aa6622/.cursor/plans/jepa-longterm-context.md).
- **Transcript (same thread, tool history):** [SmolVLA pipeline chat](f5c2a444-f5e5-424d-b17b-f3d6038068db) → [`f5c2a444-f5e5-424d-b17b-f3d6038068db.jsonl`](file:///homes/aa6622/.cursor/projects/vol-bitbucket-aa6622/agent-transcripts/f5c2a444-f5e5-424d-b17b-f3d6038068db/f5c2a444-f5e5-424d-b17b-f3d6038068db.jsonl).

---

## Executive summary (what / order / data / GPU / auto vs you)

**What the plan is:** Build **one** trajectory system: **push-v3** episodes where each record links **executed real-world (or real-sim) transitions** to **CEM + JEPA–WM latent predictions** (`pair_key`, step alignment). Extend **bridge** and **phase07** so Slurm never ingests unpaired or random-policy junk. Then run the **11-job** workflow to baseline, export→bridge, gate, TrainA/B/C, final eval.

**Order (mandatory):**

1. **Task 0** (implementation, mostly **local or interactive GPU** until you choose to test on Slurm): schema → paired exporter → `bridge_builder` → **`run_stage.sh` rewiring** (drop old exporter call) → smoke **export → bridge → non-empty `bridge_summary.json`**.
2. **Task 0.5** (**Slurm wrapper reliability**, small diff — do **before Task 5**): unify **`REPO_ROOT`** resolution across **all** DAG `.slurm` files that still use legacy `BASH_SOURCE`-only logic; fix **`stage11`** partition if used. See **Task 0.5** below (verified 2026-04-06 against repo).
3. **Tasks 1–4** (Slurm prep): authority, preflight, static gates, `RUN_ID`, dry-run.
4. **Task 5** `sbatch` **branch-parallel** full DAG.
5. **Tasks 6–9** watch, fix loop, artifacts, docs.

**What data goes where (after Task 0):**

| Step | Data in | Data out |
|------|---------|----------|
| **Real push-v3 rollouts** | Meta-World **push-v3** env (executed under **SmolVLA policy** or other fixed policy—**not** random actions) + optional future real-robot LeRobot dataset | Episode tensors / parquet-friendly records with **obs, action, success**, `pair_key` |
| **CEM + WM** | Same episode init / goals; JEPA-WM weights `SMOLVLA_JEPA_CKPT` | **Predicted latent / imagined** rollouts **paired** to the real trace |
| **phase07 export** | WM + real-rollout driver | Writes under **`SMOLVLA_JEPA_EXPORT_OUT`** (default aligns with **`SMOLVLA_JEPA_SOURCE`**) — **`export_mode: cem_paired_push_v3`**, manifest + `trajectories.*` |
| **phase08 bridge** | **`SMOLVLA_JEPA_SOURCE`** (now **only** paired-schema files) | **`SMOLVLA_DATA_ROOT`**: LeRobot **`train/`** + **`val/`** + `bridge_summary.json` (layout defined in Task 0) |
| **TrainA** | `SMOLVLA_INIT_CHECKPOINT` + **`$SMOLVLA_DATA_ROOT/train`** | `artifacts/.../stageA` |
| **TrainB** | Merge **`train`** (real-heavy or primary split) + **`val`** (latent-heavy or secondary split) per **Task 0** bridge contract—not two arbitrary unpaired roots | `mixed_lerobot_b` + train |
| **TrainC** | Same **`train`** root + **`SMOLVLA_VGG_GATE_JSON`** | `vgg_aux` |

**When a GPU Slurm job is queued:** The **first GPU allocation** happens when **`sbatch`** returns job ids from **Task 5** (stage00 is first in the DAG and requests `--gres=gpu:1` in current `.slurm` scripts). **stage01b** is **CPU-only**. **No** full DAG submit should happen until **Task 0** is merged (otherwise phase07/bridge violate this plan). Even after submit, jobs **stage04–09** can **exit immediately** with “`run_stage.sh` not found” if **`REPO_ROOT`** resolves to the wrong tree — Slurm often runs the batch script from **`/var/spool/slurm/...`**, so **`BASH_SOURCE`-only** `../..` is unsafe; **Task 0.5** removes that class of failure.

**Done automatically by the agent (in an execution session):** preflight commands, `scancel` hygiene when clearly broken deps, dry-run, launcher submit, `watch_workflow` polling, **infra-class** auto-resubmit hints, bounded in-repo fixes, autofix + longterm log entries, status messages in chat.

**Standing authorization (user policy):** The user grants **full permission** to spend GPU hours and run **`sbatch`** for this pipeline **without pausing for per-step approval**. The agent must **not** wait on the user for a separate “go-ahead” before submit, resubmit, or bounded `scancel` of broken dependency chains. **Operational discipline instead:** before and during runs, check **`squeue` / `scontrol`** (and **`sacct`** when the accounting DB works); right-size **`#SBATCH` time, mem, CPU, partition, and `--gres`** so requests are neither wasteful nor set to fail (`TIMEOUT`, OOM). **Use prior job telemetry:** elapsed / queued / running / failed times from logs, `sacct` (if available), and [`logs/`](file:///vol/bitbucket/aa6622/pi05-cube-FT/logs/) to tune future requests; record notable timings in [`jepa-longterm-context.md`](file:///homes/aa6622/.cursor/plans/jepa-longterm-context.md) or the autofix log. **Cluster / policy blockers:** resolve in-session where possible (docs, partition lists, QOS errors, `sacct` host issues); use **web search** (e.g. Exa MCP or equivalent) for site Slurm docs or error strings rather than idling for human CSG. Only escalate to the user for **secrets missing from the environment**, **irreversible destructive scope** beyond normal pipeline hygiene, or **research forks** (schema/thesis) they must choose.

**When we may still need you:** **HF / WandB / other tokens** not present on the runner; **optional** confirmation on **large irreversible** actions outside normal `scancel`/resubmit; **schema / thesis** choices for Task 0; **physical robot** data paths if push-v3 moves off pure sim. **Product note:** A Cursor session still starts from a user message—this policy applies **inside** that execution session, not unattended 24/7 without any chat turn.

---

## How the user triggers real execution (“Build equivalent”)

Markdown files do **not** queue GPU jobs. To leave **plan-only** mode, the user sends one **execute** / **submit and monitor** message to open an execution session; **after that**, the agent proceeds under **Standing authorization** in the **Executive summary** above (no waiting for repeated GPU/`sbatch` approval). **Copy-paste:**

> Execute [`docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`](file:///vol/bitbucket/aa6622/pi05-cube-FT/docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md) plus [`pipeline_to_training_start_b778a44c.plan.md`](file:///homes/aa6622/.cursor/plans/pipeline_to_training_start_b778a44c.plan.md): **Task 0 + Task 0.5 already merged** (paired push-v3 + CEM + bridge + phase07; **Slurm `REPO_ROOT` hardening** on stage04–09/10d/11; **no** random MetaWorld export). Then hygiene, branch-parallel submit, watch `--auto-resubmit`, fix failures, log to [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md), until **stage05/06/08** `RUNNING`. TrainD only if I ask. **Standing GPU/sbatch permission—do not ask me between stages.**

**Naming:** **TrainA / TrainB / TrainC / TrainD** (research plan) map to Slurm **stage05 / stage06 / stage08 / optional stage10d** and `SMOLVLA_TRAIN_VARIANT=a|b|c|d`. Those lanes describe *what* is trained; they are not a substitute for the opening execute message above.

## Definition of done

| Gate | Done when |
|------|-----------|
| **Task 0 (blocking)** | Paired **push-v3** + **CEM + JEPA–WM** exporter implemented; **`jepa_metaworld_rollout_export.py` unused** in [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh) phase07; [`bridge_builder.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/bridge_builder.py) ingests **paired** records; local smoke: export → bridge → **non-empty** `bridge_summary.json`; manifest **`export_mode`** documents **cem_paired_push_v3** (or final name). |
| **Training started** | After Task 0: **`smolvla-s05` / `smolvla-s06` / `smolvla-s08`** **`RUNNING`** at least once for a fresh `RUN_ID`. |
| **Full runbook** | Tasks **1–9** complete; artifact checks; reporting; **Task 10** only if user requests TrainD. |
| **Task 0.5 (blocking for reliable Slurm starts)** | Every **default-DAG** `.slurm` wrapper uses the same **`REPO_ROOT`** pattern as [`stage00_preflight.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage00_preflight.slurm) (`SLURM_SUBMIT_DIR` when set, else `SCRIPT_DIR/../..`). **Verified gap (repo, 2026-04-06):** [`stage04_bridge_dataset_build.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage04_bridge_dataset_build.slurm) through [`stage09_final_eval_and_bundle.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage09_final_eval_and_bundle.slurm), [`stage10_train_stageD_imagined.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage10_train_stageD_imagined.slurm), and [`stage11_slurm_orchestration.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage11_slurm_orchestration.slurm) still used **BASH_SOURCE-only** `../..`; **stage00–stage03** already fixed. [`stage11_slurm_orchestration.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage11_slurm_orchestration.slurm) also uses **`#SBATCH --partition=gpucluster3`** (not in current Imperial `sinfo` list — use **`a16` / `a100` / `t4`** or CSG default). |

**Optional engineering:** [`watch_workflow.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.py) `sacct` fallback.

---

## Order of operations (what runs when)

**Before any Slurm submit in this plan:** complete **Task 0** (paired exporter + bridge + `run_stage.sh` phase07 — **no** `jepa_metaworld_rollout_export.py`) and **Task 0.5** (Slurm **`REPO_ROOT`** hardening + **stage11** partition if applicable). Tasks **1–9** assume that code is already merged.

**Slurm script order in** [`smolvla_workflow_launcher.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/smolvla_workflow_launcher.py) `STAGE_SCRIPTS` **(serial chain index):**

| # | Slurm script | `run_stage.sh` alias → **internal phase** | Dependency note (branch-parallel) |
|---|--------------|---------------------------------------------|-----------------------------------|
| 1 | `stage00_preflight.slurm` | `stage00` → **phase00** inventory | Chain start |
| 2 | `stage01_install_lerobot_mw.slurm` | `stage01` → **phase01** env topology | After 00 |
| 3 | `stage01b_install_metaworld.slurm` | `stage01b` → **phase04** Meta-World install | After 01; CPU-only job |
| 4 | `stage02_baseline_pushv3_eval.slurm` | `stage02` → **phase06** baseline eval | **Parallel with 5 after 3:** baseline **lane** |
| 5 | `stage03_install_jepa_wms.slurm` | `stage03` → **phase07** JEPA setup (+ optional export) | **Parallel with 4 after 3:** data **lane** |
| 6 | `stage04_bridge_dataset_build.slurm` | `stage04` → **phase08** bridge | After **5** only (needs JEPA trajectories ready under `SMOLVLA_JEPA_SOURCE`) |
| 7 | `stage05_train_stageA_real_only.slurm` | `stage05` → **phase10** TrainA | After **4 and 6** (`join_train`) |
| 8 | `stage06_train_stageB_jepa_mix.slurm` | `stage06` → **phase10** TrainB | After **4 and 6** (parallel with 7) |
| 9 | `stage07_vgg_gatecheck.slurm` | `stage07` → **phase09** VGG gate | After **3** (parallel with 4–6 until gate needs prior artifacts); in practice runs in DAG after 01b |
| 10 | `stage08_train_stageC_vgg_aux.slurm` | `stage08` → **phase10** TrainC | After **6 and 9** |
| 11 | `stage09_final_eval_and_bundle.slurm` | `stage09` → **phase13** post-train eval | After **7, 8, 10** |

**Not separate Slurm jobs in this DAG:** **phase02** (GPU compat smoke), **phase03 SmolVLA install** (numbered phase03 in `run_stage.sh`, not the Slurm `stage03` file), and **phase05 model pull** — they are **not** in `STAGE_SCRIPTS`. The **HF init checkpoint** is consumed directly as **`SMOLVLA_INIT_CHECKPOINT`** in baseline (if wired), gate, and training commands (LeRobot download at use time unless you run those phases manually elsewhere).

**Branch-parallel recap:** After **`stage01b`**, **baseline (`stage02`)** and **JEPA lane (`stage03`)** run in parallel; **TrainA/TrainB** wait on **baseline + bridge**; **TrainC** waits on **bridge + gate**.

---

## Data used at every stage (precise defaults)

Environment defaults from [`config.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh). Paths are **defaults**; override with `export` / `#SBATCH` env.

| Stage (Slurm) | Primary **inputs** (read) | Primary **outputs** (write) |
|---------------|---------------------------|-----------------------------|
| **stage00** | Host: CUDA/Slurm/fs | [`SMOLVLA_REPORT_ROOT`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh) preflight / inventory markdown |
| **stage01** | None (network for `pip`/`uv`) | Virtualenvs under `SMOLVLA_ENV_ROOT` (`lerobot_mw_py310`, `jepa_wms_py310`, `vggflow_py311`, …), env snapshots under `SMOLVLA_LOCK_ROOT` |
| **stage01b** | PyPI / caches | Meta-World inside **`SMOLVLA_LEROBOT_ENV_DIR`** |
| **stage02** | **Policy:** [`SMOLVLA_INIT_CHECKPOINT`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh) (HF id or path) via `run_baseline_eval.sh` / `SMOLVLA_BASELINE_CMD`; Meta-World env from stage01b | Logs under `SMOLVLA_REPORT_ROOT` (`phase06_baseline_eval.log`); optional **`eval_info.json`** under directory printed as `Baseline eval output directory:` in log |
| **stage03** | **JEPA-WM:** repo [`SMOLVLA_VGG_REPO`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh)`/jepa-wms`, checkpoint **`SMOLVLA_JEPA_CKPT`**; **push-v3** in lerobot env; **Export** when `SMOLVLA_JEPA_EXPORT_ENABLED=1` | Smoke logs; if export enabled: **paired** **`trajectories.*`** + **`export_manifest.json`** under **`SMOLVLA_JEPA_EXPORT_OUT`** (default **`SMOLVLA_JEPA_SOURCE`**). **Exporter:** **executed push-v3 rollouts** (policy-driven, **not** random actions) **paired** with **CEM + WM latent** predictions (`pair_key`, aligned steps). Legacy random export **must not** run (Task 0 removes it from phase07). |
| **stage04** | **`SMOLVLA_JEPA_SOURCE`:** files written by Task 0 exporter (paired schema; **not** a bare WM checkpoint) | **`SMOLVLA_DATA_ROOT`**: LeRobot **`train/`** + **`val/`** + **`bridge_summary.json`**. [`bridge_builder.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/bridge_builder.py) reads paired records and **splits** into train vs val per Task 0 contract (e.g. real-heavy vs latent-heavy columns or episode tags). |
| **stage05 TrainA** | **Checkpoint:** `SMOLVLA_INIT_CHECKPOINT`. **Dataset:** **`$SMOLVLA_DATA_ROOT/train`** (LeRobot v2.1 layout from bridge). | Checkpoints/metrics under **`SMOLVLA_ARTIFACT_ROOT/stage10_a_<run_id>[_job<id>]/stageA`** (see `phase10` in [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh)). |
| **stage06 TrainB** | **Checkpoint:** `SMOLVLA_INIT_CHECKPOINT`. **Two LeRobot roots:** `--real-data-root` = **`$SMOLVLA_DATA_ROOT/train`**, `--jepa-data-root` = **`$SMOLVLA_DATA_ROOT/val`** (defaults in `run_stage.sh`). [`merge_lerobot_v21_datasets.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/merge_lerobot_v21_datasets.py) builds **`mixed_lerobot_b`**. | **After Task 0:** `train/` and `val/` are **both** derived from the **same paired** export (split strategy defined in bridge — e.g. executed vs latent emphasis), **not** from unrelated roots unless you override env for ablations. |
| **stage07 gate** | **`SMOLVLA_INIT_CHECKPOINT`**; writes/reads **`SMOLVLA_VGG_GATE_JSON`** (default **`$SMOLVLA_REPORT_ROOT/phase09_gatecheck.json`**) | Gate JSON + report |
| **stage08 TrainC** | **`SMOLVLA_INIT_CHECKPOINT`**; **`$SMOLVLA_DATA_ROOT/train`**; **`SMOLVLA_VGG_GATE_JSON`** (must match checkpoint when strict) | **`…/stageC/vgg_aux`** (and related) under `SMOLVLA_ARTIFACT_ROOT/stage10_c_…` |
| **stage09** | **`SMOLVLA_FINAL_EVAL_CHECKPOINT`** if set, else **`SMOLVLA_INIT_CHECKPOINT`** | Post-train eval bundle per **phase13** |

**TrainD (optional, not in default 11 jobs):** **`SMOLVLA_STAGE_D_DATA_ROOT`** (default **`$SMOLVLA_DATA_ROOT/val`**) + same gate/checkpoint pattern — see `run_stage.sh` **phase10** / `stage10_train_stageD_imagined.slurm`.

---

## Cluster environment: Imperial DoC GPU cluster (`gpucluster3.doc.ic.ac.uk`)

**Context:** Submit from **`gpucluster2.doc.ic.ac.uk` or `gpucluster3.doc.ic.ac.uk`** (DoC policy; see [`reports/10_status.md`](file:///vol/bitbucket/aa6622/pi05-cube-FT/reports/10_status.md)). Shared project data under **`/vol/bitbucket/`**; avoid heavy `pip`/`git` on submit hosts (use lab PC / shell — [GitBook step 3](https://systems.pages.doc.ic.ac.uk/gpucluster/step3.html)).

### Official documentation (web)

- **GitBook (primary):** [What is Slurm and the GPU cluster?](https://systems.pages.doc.ic.ac.uk/gpucluster/) — Ubuntu 24.04 note (Sept 2024), Slurm overview, DoC-only service.
- **FAQ / hardware:** [Frequently Asked Questions](https://systems.pages.doc.ic.ac.uk/gpucluster/step8.html) — GPU types: **NVIDIA T4 (16GB), A16 (16GB), A30 (24GB), A40 (48GB), A100 (80GB)**.
- **Faculty mirror:** [Department of Computing GPU Cluster Guide](https://www.imperial.ac.uk/computing/people/csg/guides/hpcomputing/gpucluster/).

### Live inventory snapshot (research on this node, 2026-04-06)

Commands run: `sinfo -o '%P %a %l %D %t %N %G'`, `squeue -u $USER`, `scontrol show job <id>`, `sinfo -a -h -o '%P'`.

| Partition | Role (from `sinfo` + scripts) | Notes |
|-----------|------------------------------|--------|
| **`a16*`** | Default `*` partition; matches most `scripts/slurm/stage*.slurm` (`#SBATCH --partition=a16`) | Mixed/idle A16-class nodes (`gpuvm35-36`, `parrot`). |
| **`a100`** | High-memory training; launcher fallback list includes `a100` | User had pending SmolVLA jobs targeting **a100** (see below). |
| **`t4`** | **stage03** JEPA install script uses `t4` | Many idle T4 nodes in snapshot. |
| **`a40`**, **`a30`** | Launcher fallback | `a40` had one node `down*` in snapshot. |
| **`docker`**, **`training`**, **`interactive`** | Special pools | `interactive` shares T4-like nodes; confirm policy before using. |

**Partition name `gpucluster3`:** **`sinfo` does not list `gpucluster3` as a partition** on this cluster (only the hostnames above). [`stage11_slurm_orchestration.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage11_slurm_orchestration.slurm) sets `#SBATCH --partition=gpucluster3` — treat as **legacy or wrong for current Slurm config** unless CSG confirms; prefer **`a16` / `a100` / `t4`** aligned with other stages.

### `sacct` vs `squeue` / `scontrol` (important)

On the session host used for research, **`sacct` failed**:

```text
sacct: error: _open_persist_conn: failed to open persistent connection to host:localhost:6819: Connection refused
```

So **history queries may be unavailable** from some VMs; rely on **`squeue`**, **`scontrol show job <id>`**, and **log files** under [`logs/`](file:///vol/bitbucket/aa6622/pi05-cube-FT/logs/). If [`watch_workflow.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.py) misbehaves on a host without accounting DB, run the watcher from a login node where `sacct` works or ask CSG for `slurmdbd` access.

### Example: stuck dependency chain (same session)

| JobID | Name | State | Reason |
|-------|------|-------|--------|
| 228430 | `smolvla-s06` (TrainB) | PENDING | **`DependencyNeverSatisfied`** — `afterok:228429` **(failed)** |
| 228431 | `smolvla-s07` | PENDING | Dependency on 228430 unfulfilled |
| 228432 | `smolvla-s08` | PENDING | Dependency unfulfilled |

**Action for autonomous runs:** After a failed upstream job, **cancel** dependent placeholders (`scancel 228430 228431 228432`) and **resubmit from the failed stage** with a fresh `workflow_*.json`; do not expect Slurm to recover `DependencyNeverSatisfied` automatically.

### Subagents + parallelism (recap)

- **Parallel Slurm waves** are defined by [`submit_workflow_branch_parallel`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/smolvla_workflow_launcher.py) (e.g. **stage02 ∥ stage03** after `stage01b`; **TrainA ∥ TrainB** after baseline+bridge join).
- **Subagents:** use for **read-only** parallel log/`scontrol` inspection; **one** owner submits and updates `RUN_ID` / workflow JSON.

---

## File and responsibility map

*(Writing-plans: primary files an agent edits or invokes during Task 0 and Tasks 1–10.)*

| Path | Role |
|------|------|
| [`scripts/smolvla_vggflow/submit_workflow.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/submit_workflow.sh) | Serial submit + dry-run; writes `runs/workflow_${RUN_ID}.json` |
| [`scripts/smolvla_vggflow/smolvla_workflow_launcher.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/smolvla_workflow_launcher.py) | `--submit --branch-parallel`, partition fallback, `parallel_submission_map.json` |
| [`scripts/smolvla_vggflow/watch_workflow.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.sh) | Extract job IDs from workflow JSON → `watch_workflow.py` |
| [`scripts/smolvla_vggflow/watch_workflow.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.py) | Poll `sacct`, optional `--auto-resubmit` with new job id tracking |
| [`scripts/smolvla_vggflow/run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh) | Phase logic invoked by each `scripts/slurm/stage*.slurm` |
| [`scripts/slurm/stage05_train_stageA_real_only.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage05_train_stageA_real_only.slurm) (and 06, 08, 10d) | Pin `SMOLVLA_TRAIN_VARIANT` |
| [`runs/workflow_*.json`](file:///vol/bitbucket/aa6622/pi05-cube-FT/runs/) | Job id registry for watcher |
| [`artifacts/`](file:///vol/bitbucket/aa6622/pi05-cube-FT/artifacts/), [`datasets/bridged/`](file:///vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged/), [`reports/`](file:///vol/bitbucket/aa6622/pi05-cube-FT/reports/) | Runtime outputs per `config.sh` |

---

### Task 0: Paired push-v3 + CEM / JEPA–WM exporter and bridge (**before** Tasks 1–5)

**Prerequisite for this entire runbook:** Without Task 0, phase07/phase08 do not match the **no-fallback** architecture above.

**Status today:** **Not implemented** in repo as specified — [`jepa_metaworld_rollout_export.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/jepa_metaworld_rollout_export.py) must be **removed from** [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh) phase07; [`bridge_builder.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/bridge_builder.py) must gain paired-field ingestion.

**Files (expected touch set):**

- **Create:** e.g. `scripts/smolvla_vggflow/jepa_cem_paired_pushv3_export.py` — roll out **push-v3** with a **fixed policy** (SmolVLA or agreed checkpoint), log **executed** transitions; run **CEM** in JEPA latent space against the same goals; emit **paired** episode records + manifest.
- **Modify:** [`run_stage.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/run_stage.sh) phase07 — invoke **only** the new exporter when `SMOLVLA_JEPA_EXPORT_ENABLED=1` (**no** `SMOLVLA_JEPA_EXPORT_MODE` fork to a random-policy script).
- **Modify:** [`bridge_builder.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/bridge_builder.py) — `pair_key`, latent vs executed fields, `schema_version`, fail-fast if pairing missing.
- **Modify:** [`config.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh) — defaults for episode counts, paths, policy checkpoint for real rollouts if needed.
- **Remove or archive:** [`jepa_metaworld_rollout_export.py`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/jepa_metaworld_rollout_export.py) — **delete** or keep in repo **unused**; must not be called from `run_stage.sh`.
- **Docs:** [`jepa-rollout-algorithm-integration_c18d9a72.plan.md`](file:///homes/aa6622/.cursor/plans/jepa-rollout-algorithm-integration_c18d9a72.plan.md) — **algorithm_contract** as spec for schema/CEM (YAML may stay stale; body is truth).

- [ ] **Step 1: Lock minimal schema (one page)**

Per-episode `pair_key`; per timestep (or chunk): **executed** obs/action (push-v3), **WM / CEM** latent prediction and planner metadata. Align with `paired_mapping_schema_lock` intent in the rollout plan.

- [ ] **Step 2: Implement CEM loop + real push-v3 rollout driver**

JEPA-WM load, CEM action/plan selection, logging paired to the executed trace.

- [ ] **Step 3: Write exporter output + manifest**

`trajectories.pt` (or agreed format) + `export_manifest.json` with `schema_version`, `export_mode: cem_paired_push_v3` (or final name).

- [ ] **Step 4: Extend bridge**

Split/merge into LeRobot v2.1 **`train/`** and **`val/`** so TrainB’s two roots are **meaningful** for mixed training (per Executive summary table).

- [ ] **Step 5: Wire `run_stage.sh` and adjust Slurm time if needed**

Phase07 calls **only** the new exporter. Increase `#SBATCH` time/mem for stage03 if CEM is slower than the old random rollouts.

- [ ] **Step 6: Verification**

Local smoke: export → `bridge_builder` → non-empty **`bridge_summary.json`**. Log in [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md).

**Commit:** After Step 6 — e.g. `feat: paired push-v3 CEM JEPA export + bridge`.

---

### Task 0.5: Slurm batch wrappers — `REPO_ROOT` + stage11 partition (**before Task 5**)

**Why:** Slurm may execute the submitted script from a **spool path** (`/var/spool/slurm/...`). Then **`dirname "${BASH_SOURCE[0]}"`** is **not** under the repo; **`REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"`** points at garbage and **`run_stage.sh`** is not found — the same failure mode as earlier attempts.

**Verified in repo (2026-04-06):** [`stage00_preflight.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage00_preflight.slurm), [`stage01_install_lerobot_mw.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage01_install_lerobot_mw.slurm), [`stage01b_install_metaworld.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage01b_install_metaworld.slurm), [`stage02_baseline_pushv3_eval.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage02_baseline_pushv3_eval.slurm), and [`stage03_install_jepa_wms.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage03_install_jepa_wms.slurm) already use:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
```

**Still legacy (must match the above):** [`stage04_bridge_dataset_build.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage04_bridge_dataset_build.slurm), [`stage05_train_stageA_real_only.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage05_train_stageA_real_only.slurm), [`stage06_train_stageB_jepa_mix.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage06_train_stageB_jepa_mix.slurm), [`stage07_vgg_gatecheck.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage07_vgg_gatecheck.slurm), [`stage08_train_stageC_vgg_aux.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage08_train_stageC_vgg_aux.slurm), [`stage09_final_eval_and_bundle.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage09_final_eval_and_bundle.slurm), [`stage10_train_stageD_imagined.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage10_train_stageD_imagined.slurm), [`stage11_slurm_orchestration.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage11_slurm_orchestration.slurm).

**Precondition:** Submit with `sbatch` from the **repo root** (so `SLURM_SUBMIT_DIR` is the checkout), as the launcher already assumes.

- [ ] **Step 1:** Apply the **same** `SLURM_SUBMIT_DIR` / `else` block to every file in the list above (keep each file’s existing `export` / `bash run_stage.sh …` lines after `cd`).

- [ ] **Step 2:** In [`stage11_slurm_orchestration.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage11_slurm_orchestration.slurm), change **`#SBATCH --partition=gpucluster3`** to **`a16`** (or your site default; see **Cluster environment** section below). **Only if** you use stage11 (not in default 11-job DAG).

**Commit:** e.g. `fix(slurm): unify REPO_ROOT via SLURM_SUBMIT_DIR for stage04–11`.

**Related audit notes (other model — confirmed):**

- **[`runs/workflow_*.json`](file:///vol/bitbucket/aa6622/pi05-cube-FT/runs/)** in the workspace may show **`"job_id": "<unsubmitted>"`** — that is **dry-run** / not-yet-submitted; a **live** DAG is only proven after Task 5 with numeric ids.
- **[`reports/stage08_bridge_design_status.md`](file:///vol/bitbucket/aa6622/pi05-cube-FT/reports/stage08_bridge_design_status.md)** can report **`[FAIL] SMOLVLA_JEPA_SOURCE produced no usable trajectory files`** until **Task 0** export (or a valid pre-populated `SMOLVLA_JEPA_SOURCE`) exists — logical downstream blocker for TrainA/B/C, separate from the spool-path bug.
- **QOS / partition fit** (`QOSMaxSubmitJobPerUserLimit`, etc.): cluster policy; catch in **Task 2** preflight and launcher retries — not a single repo file.
- **CUDA missing in batch step:** phase07 / phase09 may log WARN/skip when CUDA is not visible; can block **successful** JEPA work without blocking **submission** — treat as env/`#SBATCH --gres` / partition issue during watch.

---

### Task 1: Confirm execution session and working directory

**Files:** None (operator/agent confirmation).

- [ ] **Step 1: Confirm not in plan-only mode**

If the user has **never** opened an execution turn (`execute` / `run the plan` / `submit and monitor`), **stop** — Cursor agents do not run tools without a user-started execution session. If the user **has** opened such a session, treat **Standing authorization** (Executive summary) as in effect: **do not** pause for extra “may I `sbatch`?” prompts; proceed with Tasks 2–9 while checking queue load and right-sizing resources.

- [ ] **Step 2: `cd` to repo root**

```bash
cd /vol/bitbucket/aa6622/pi05-cube-FT && pwd
```

**Expected:** `stdout` ends with `pi05-cube-FT`.

---

### Task 2: Slurm preflight

**Files:** None.

- [ ] **Step 1: Verify Slurm CLI**

```bash
which sbatch squeue sacct scontrol
```

**Expected:** Four absolute paths (no empty which).

- [ ] **Step 2: Optional queue snapshot**

```bash
squeue -u "$USER" -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R' | head -20
```

**Expected:** Command succeeds (may be empty if no jobs).

- [ ] **Step 2b: Ops hygiene — broken Slurm dependency chains**

```bash
squeue -u "$USER" -o '%.18i %.12j %.2t %R' | head -40
```

If any `smolvla-s*` (or this pipeline’s) jobs show **`DependencyNeverSatisfied`** or are clearly stuck on `afterok` of a **failed** upstream job, **`scancel`** those dependent job IDs before a fresh submit. Matches companion plan todo `ops-hygiene`. Append a line to [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md) if the cancel is non-obvious.

**Expected:** No dead pending chain blocking a new `RUN_ID`. Under **Standing authorization**, do not wait for the user to approve **`scancel`** on clearly broken dependents.

- [ ] **Step 2c: Queue load + historical job times**

Use **`squeue`** (partition pressure, pending reasons) before **`sbatch`**. If **`sacct`** works, sample **recent** `smolvla-s*` (or same-account) jobs: **Elapsed**, **State**, **MaxRSS**, **`TIMEOUT` / `OUT_OF_MEMORY`**. Adjust **`#SBATCH --time` / `--mem` / partition** only when evidence supports it; record baselines in [`jepa-longterm-context.md`](file:///homes/aa6622/.cursor/plans/jepa-longterm-context.md) for future runs. If **`sacct`** fails on this host, rely on **`logs/stage*_*.log`** timestamps and **`scontrol show job <id>`** instead.

- [ ] **Step 3: Probe accounting DB (`sacct`)**

```bash
sacct -n -j 1 2>&1 | head -3
```

**Expected:** Either a header/data row (DB reachable) or `Connection refused` to `localhost:6819` / `slurmdbd` — if the latter, rely on `scontrol`/`squeue` and log files for this host; see **Cluster environment** above.

- [ ] **Step 4: Skim prior workflows**

```bash
ls -la runs/workflow_*.json | tail -5
```

**Expected:** At least zero or more JSON files; use latest as informal reference for naming. Prefer **`logs/stage*_*.log`** for real job ids when `workflow_*.json` is dry-run.

---

### Task 3: Static in-band blocker gate (code + config, no GPU)

**Files:** Read-only inspection of Slurm + `run_stage.sh` + trainer.

- [ ] **Step 1: Variant pins present**

```bash
grep -n "SMOLVLA_TRAIN_VARIANT" scripts/slurm/stage05_train_stageA_real_only.slurm \
  scripts/slurm/stage06_train_stageB_jepa_mix.slurm \
  scripts/slurm/stage08_train_stageC_vgg_aux.slurm
```

**Expected:** `=a`, `=b`, `=c` respectively in those files.

- [ ] **Step 2: TrainC command includes `--max-steps` in default builder**

```bash
grep -n "STAGE_C_CMD" scripts/smolvla_vggflow/run_stage.sh | head -5
grep "max-steps" scripts/smolvla_vggflow/run_stage.sh | grep STAGE_C
```

**Expected:** Default `STAGE_C_CMD` line includes `--max-steps '${SMOLVLA_FIRST_FT_STEPS}'` (see ~1314 in current tree).

- [ ] **Step 3: Launcher branch-parallel entry exists**

```bash
python3 -c "import ast, pathlib; p=pathlib.Path('scripts/smolvla_vggflow/smolvla_workflow_launcher.py'); ast.parse(p.read_text()); print('launcher_parse_ok')"
```

**Expected:** `launcher_parse_ok`.

- [ ] **Step 4: Record gate result**

Append a short note (in chat or in autofix log): `blocker_gate_static: PASS` or document first failing step with file:line.

- [ ] **Step 5: Data path for JEPA / bridge (companion plan `data-bridge-path`)**

Before real submit (Task 5), confirm **one** intentional path to trajectories for phase08:

```bash
echo "SMOLVLA_JEPA_EXPORT_ENABLED=${SMOLVLA_JEPA_EXPORT_ENABLED:-0}"
echo "SMOLVLA_JEPA_SOURCE=${SMOLVLA_JEPA_SOURCE:-<unset>}"
```

**Expected:** Either `SMOLVLA_JEPA_EXPORT_ENABLED=1` (Task 0 exporter runs in the DAG) **or** `SMOLVLA_JEPA_SOURCE` points at existing **paired-schema** trajectory artifacts on shared storage. If both are empty/wrong, phase08 will hard-fail. Artifacts must match **Task 0** manifest (`export_mode` / `schema_version`); do not submit the full DAG until Task 0 is merged unless you are explicitly testing with pre-exported paired data only.

**Commit (optional):** Only if you changed files to fix a blocker; otherwise skip.

```bash
# git add … && git commit -m "fix: slurm blocker gate …"  # only if edits made
```

---

### Task 4: Assign `RUN_ID` and dry-run workflow JSON

**Files:** Creates [`runs/workflow_${RUN_ID}.json`](file:///vol/bitbucket/aa6622/pi05-cube-FT/runs/) (dry-run = `<unsubmitted>` job ids).

- [ ] **Step 1: Export `RUN_ID`**

```bash
cd /vol/bitbucket/aa6622/pi05-cube-FT
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$RUN_ID"
```

**Expected:** A UTC-like string, e.g. `20260406_143022`.

- [ ] **Step 2: Dry-run**

```bash
bash scripts/smolvla_vggflow/submit_workflow.sh --dry-run
```

**Expected:** Exit code 0; file `runs/workflow_${RUN_ID}.json` exists.

- [ ] **Step 3: Validate JSON**

```bash
python3 -c "import json, os; p=f\"runs/workflow_{os.environ['RUN_ID']}.json\"; json.load(open(p)); print('json_ok', p)"
```

**Expected:** `json_ok runs/workflow_<RUN_ID>.json`.

---

### Task 5: Submit full DAG (branch-parallel, recommended)

**Files:** Writes [`runs/workflow_${RUN_ID}.json`](file:///vol/bitbucket/aa6622/pi05-cube-FT/runs/) with real job ids; writes [`artifacts/parallel_submission_map_${RUN_ID}.json`](file:///vol/bitbucket/aa6622/pi05-cube-FT/artifacts/) (path per command below).

- [ ] **Step 1: Ensure `RUN_ID` still exported**

```bash
echo "${RUN_ID:?RUN_ID must be set}"
```

- [ ] **Step 2: Submit branch-parallel DAG**

```bash
cd /vol/bitbucket/aa6622/pi05-cube-FT
export RUN_ID
python3 scripts/smolvla_vggflow/smolvla_workflow_launcher.py \
  --submit \
  --branch-parallel \
  --write-json "runs/workflow_${RUN_ID}.json" \
  --parallel-map-out "artifacts/parallel_submission_map_${RUN_ID}.json"
```

**Expected:** Exit code 0; stdout includes Slurm job ids; `runs/workflow_${RUN_ID}.json` contains numeric `job_id` fields (not `<unsubmitted>`).

- [ ] **Step 3: Serial fallback (only if branch-parallel fails)**

```bash
bash scripts/smolvla_vggflow/submit_workflow.sh
```

**Expected:** Same as above for serial chain; longer wall-clock.

**Commit:** None unless you changed launcher/config to fix submit.

---

### Task 6: Monitor until terminal state

**Files:** Uses [`watch_workflow.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/watch_workflow.sh).

**Training-started milestone:** When **stage05 / stage06 / stage08** jobs (see `workflow_${RUN_ID}.json` → `stages` / Slurm job names `smolvla-s05`, `smolvla-s06`, `smolvla-s08`) show **`RUNNING`** in `squeue` or `scontrol show job <id>`, companion plan todo `verify-train-running` is satisfied; report with job IDs. Continue through watcher completion for the **full runbook** (Tasks 8–9).

- [ ] **Step 1: Start watcher (foreground)**

```bash
cd /vol/bitbucket/aa6622/pi05-cube-FT
bash scripts/smolvla_vggflow/watch_workflow.sh "runs/workflow_${RUN_ID}.json" -- \
  --poll 90 \
  --max-retries 2 \
  --auto-resubmit
```

**Expected (success):** Exit code 0; message like `all jobs completed successfully`.

**Expected (failure):** Exit code 1; note failing `job_id` and reason class from stdout.

- [ ] **Step 2: Background alternative (if user cannot block terminal)**

Run Step 1 in background job or `tmux`/`screen`; poll manually:

```bash
sacct -j JOBID --format=JobID,State,ExitCode -n
```

Repeat for each active id from `runs/workflow_${RUN_ID}.json`.

---

### Task 7: Failure classification and bounded fix loop

**Files:** May modify [`scripts/smolvla_vggflow/*`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/), [`config.sh`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/smolvla_vggflow/config.sh), or env; must log to [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md).

- [ ] **Step 1: Collect evidence**

```bash
# Replace JOBID
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P
ls slurm-JOBID.* 2>/dev/null || ls logs/slurm*JOBID* 2>/dev/null || true
```

**Expected:** `State` and `ExitCode` explain pass/fail.

- [ ] **Step 2: If class is `node_resources` / `no-gpu` / transient infra**

Rely on watcher `--auto-resubmit` first; if exhausted, adjust `SMOLVLA_PARTITION_LIST` or Slurm `#SBATCH` partition lines **only** with cluster-appropriate values, then append **numbered** entry to `jepa-autofix-attempt-log.md` and resubmit **failed stage only** via `sbatch` with correct `--dependency=afterok:...` if needed.

- [ ] **Step 3: If class is schema / empty bridge / missing JEPA trajectories / gate**

Do **not** blindly resubmit full DAG. Inspect:

```bash
test -f "${SMOLVLA_DATA_ROOT:-datasets/bridged}/bridge_summary.json" && head -c 400 "${SMOLVLA_DATA_ROOT}/bridge_summary.json"
```

Fix data path, export flags (`SMOLVLA_JEPA_EXPORT_ENABLED`, etc.), or code; log autofix entry; resubmit from the **first failed stage** forward.

- [ ] **Step 4: Re-run watcher**

Point `watch_workflow.sh` at an **updated** `workflow_*.json` if job ids changed after partial resubmit (you may need to regenerate JSON or hand-edit job list—prefer regenerate from launcher if possible).

---

### Task 8: Post-run artifact contract checks

**Files:** Read outputs under `artifacts/`, `datasets/bridged/`, `reports/`.

- [ ] **Step 1: Bridge summary**

```bash
python3 - <<'PY'
import json, os
root = os.environ.get("SMOLVLA_DATA_ROOT", "datasets/bridged")
p = os.path.join(root, "bridge_summary.json")
d = json.load(open(p))
tr = d.get("train_records") or d.get("train", 0)
va = d.get("val_records") or d.get("val", 0)
assert (tr or 0) + (va or 0) > 0, (p, d)
print("bridge_ok", p, "train", tr, "val", va)
PY
```

**Expected:** `bridge_ok` line printed.

- [ ] **Step 2: Gate JSON (if TrainC ran)**

```bash
python3 -c "import json,sys; g=json.load(open(sys.argv[1])); print('gate_ok',g.get('gate_ok'),'contract',g.get('contract_ok'))" "${SMOLVLA_VGG_GATE_JSON:-reports/vgg_gate.json}" 2>/dev/null || echo "gate_file_missing_or_unset"
```

**Expected:** Interpret per run: if StageC executed, `gate_ok` should be truthy for success path.

- [ ] **Step 3: Training manifests / metrics**

```bash
find artifacts -maxdepth 4 -name 'run_manifest.json' -mmin -1440 2>/dev/null | head
```

**Expected:** At least one manifest for recent train stages if phase10 completed.

---

### Task 9: Reporting and long-term documentation

**Files:** Edit [`jepa-autofix-attempt-log.md`](file:///homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md), [`jepa-longterm-context.md`](file:///homes/aa6622/.cursor/plans/jepa-longterm-context.md).

- [ ] **Step 1: Fill [`jepa-autonomous-run-report-template.md`](file:///homes/aa6622/.cursor/plans/jepa-autonomous-run-report-template.md) in chat**

Every ~15 minutes while jobs run, and once at closure.

- [ ] **Step 2: Append autofix log entry for each fix attempt**

Fields: UTC time, lane/stage/job id, failure signature, command, outcome (`resolved` / `not_resolved`), next action.

- [ ] **Step 3: Update long-term context**

Bump “Last updated” and add a short bullet for: `RUN_ID`, submit mode (serial vs branch-parallel), final sacct summary, and any new env vars.

**Commit:**

```bash
git add docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md \
  /homes/aa6622/.cursor/plans/jepa-autofix-attempt-log.md \
  /homes/aa6622/.cursor/plans/jepa-longterm-context.md 2>/dev/null
git status
# git commit -m "docs: autonomous slurm run RUN_ID=… outcomes"  # if under git and user wants
```

---

### Task 10: Optional TrainD (explicit scope only)

**Out of default execution.** Only if the user explicitly requests in the same session:

- TrainD: enable `SMOLVLA_ENABLE_TRAIN_D` / submit [`stage10_train_stageD_imagined.slurm`](file:///vol/bitbucket/aa6622/pi05-cube-FT/scripts/slurm/stage10_train_stageD_imagined.slurm) per README.

- [ ] **Step 1: Default — skip Task 10**

Mark N/A in final report unless user expanded scope.

---

## Self-review (writing-plans checklist)

1. **Spec coverage:** **Task 0** and **Task 0.5** are mandatory before Slurm submit (data path + **spool-safe `REPO_ROOT`**). Tasks map to [`jepa-rollout-algorithm-integration_c18d9a72.plan.md`](file:///homes/aa6622/.cursor/plans/jepa-rollout-algorithm-integration_c18d9a72.plan.md) (paired schema / CEM intent) and to companion [`pipeline_to_training_start_b778a44c.plan.md`](file:///homes/aa6622/.cursor/plans/pipeline_to_training_start_b778a44c.plan.md) todos (`ops-hygiene` → Task 2 Step 2b; `data-bridge-path` → Task 3 Step 5; `submit-branch-parallel` / `watch-loop` / `verify-train-running` → Tasks 5–6 + training-started milestone). Tasks 8–9 close reporting. **DAG order + per-stage data I/O:** *Executive summary*, *Order of operations*, *Data used at every stage*. Remaining gaps: `run_scoped_path_isolation`, `append_only_concurrency_guard`, full `gate_freshness_binding` — separate engineering unless user tightens scope.
2. **Placeholder scan:** No `TBD` / unfilled commands; `JOBID` is the only replaceable token and is labeled.
3. **Type consistency:** `RUN_ID`, stage05/06/08, and TrainA/B/C naming are aligned with companion plans; paths match repo layout.

---

## Execution choice

**Plan complete and saved to** [`pi05-cube-FT/docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md`](file:///vol/bitbucket/aa6622/pi05-cube-FT/docs/superpowers/plans/2026-04-06-jepa-slurm-autonomous-execution.md).

**1. Subagent-driven (writing-plans default)** — Dispatch a **fresh subagent per task** (**Task 0** + **Task 0.5** for code/Slurm wrappers; Tasks **1–9** for ops) with `superpowers:subagent-driven-development`, **two-stage review** between tasks. **Constraint:** only **one** subagent (or the primary) may call `sbatch` / own `RUN_ID` and `runs/workflow_*.json`; other subagents **readonly** (logs, `scontrol`, `squeue`).

**2. Inline execution (recommended for this pipeline)** — The **primary** agent completes **Task 0** and **Task 0.5**, then runs Tasks **1–9** in order; use `superpowers:executing-plans` checkpoints after Task 0 Step 6, Task 0.5 Step 2, Task 3, Task 5, and Task 8. Optional **readonly** parallel helpers for log tailing only.

**Which approach?** Default to **2** for this pipeline (single owner of submit/watch); use **1** only if subagents are strictly readonly and one owner is explicit.
