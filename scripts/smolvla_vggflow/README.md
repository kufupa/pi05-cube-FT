# SmolVLA + Meta-World + JEPA-WMS + VGG Workflow

This directory stores the execution artifacts for the full plan implementation.

## Quick entry points

- `run_stage.sh <stage>`: run a single numbered/plan phase locally.
- `preflight_dependency_order.sh`: run deterministic dependency checks in strict order (`CHECK_01_SLURM` through `CHECK_07_BRIDGE`) before submitting workflow stages.
- `submit_workflow.sh`: submit the full **11-stage** serial Slurm DAG and write a `workflow_*.json` (includes **`generated_at_utc`**, **`run_id`** from exported `RUN_ID`; override path with `SMOLVLA_WORKFLOW_JSON`).
- `smolvla_workflow_launcher.py --submit --branch-parallel`: optional fan-out/fan-in submit (same 11 scripts; **do not** use with `SMOLVLA_STAGE11_ENABLED=1`).
- `watch_workflow.sh <workflow.json> [watch_workflow.py args]`: forwards to `watch_workflow.py` (e.g. `--auto-resubmit --max-retries 2`). Auto-resubmit adds `--gres=gpu:1` only when `scontrol` shows GPU in **Gres** or **AllocTRES/ReqTRES**, or the failure class was `no-gpu` (CPU stages such as `stage01b` stay CPU-only on `node_resources` retry). Job state uses **`sacct`** with **JobID-aware row selection** when Slurm returns multiple lines per id.

## Dependency-order preflight

Run this from the repo root before full workflow submission when validating a host or env setup:

```bash
bash scripts/smolvla_vggflow/preflight_dependency_order.sh
```

Expected behavior:
- The script executes checks in deterministic dependency order and prints stage markers in sequence:
  `CHECK_01_SLURM`, `CHECK_02_ENVS`, `CHECK_03_CUDA_RENDER`, `CHECK_04_JEPA`, `CHECK_05_SMOLVLA`, `CHECK_06_EXPORT`, `CHECK_07_BRIDGE`.
- Each stage emits practical PASS/FAIL lines for required commands, env python executables, CUDA/render readiness, JEPA/SmolVLA script presence, export config, and bridge dataset roots.
- The script exits `0` only when all checks pass; any failed dependency returns non-zero.

### Execution policy

Run heavy GPU stages only through Slurm, never directly on the login VM.
- `stage06_baseline_eval` (`stage02_baseline_pushv3_eval`)
- `stage07_jepa_setup` (`stage03_install_jepa_wms`)
- `stage09_vgg_gates` (`stage07_vgg_gatecheck`)
- `stage10_train_loop` (train stage scripts `stage05` / `stage06` / `stage08` / optional `stage10_train_stageD_imagined` with `SMOLVLA_TRAIN_VARIANT` / `SMOLVLA_ENABLE_TRAIN_D`)
- `phase13_post_train_eval` then `phase12_reporting` (`stage09_final_eval_and_bundle` runs both in one job)
- `stage01b_install_metaworld` is **CPU-only** (Meta-World pip install); other Slurm stages still request GPU in their `#SBATCH` lines unless noted.

`run_stage.sh` now includes a hard guard that blocks direct local execution of those stages unless
`SMOLVLA_ALLOW_CPU_RUN=1` is set explicitly.
For normal operation, from the **repo root** (`pi05-cube-FT`):
- `bash scripts/smolvla_vggflow/submit_workflow.sh` (or `--dry-run`).

Preferred GPU partition order can be controlled with:
- `SMOLVLA_PARTITION_LIST="a100,a40,a30,t4,a16"`
- `smolvla_workflow_launcher.py --submit` marks **stage00**, **stage01**, and **stage04** as GPU stages for partition retry (alongside train/eval), consistent with their `#SBATCH --gres=gpu:1` lines.

## Controlled cleanup + rerun (no overwrite)

Run this sequence from the repo root when recovering from a bad export/bridge run.

1) Archive first (no destructive delete on first pass):

```bash
ARCHIVE_TAG="$(date -u +%Y%m%d)"
mkdir -p "artifacts/regression_archive/${ARCHIVE_TAG}"
mv "/vol/bitbucket/aa6622/.cache/jepa_workflow_egl_20260407T073139Z" \
  "artifacts/regression_archive/${ARCHIVE_TAG}/"
```

2) Set one run scope and strict non-overwrite guards:

```bash
export RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
export SMOLVLA_RUN_SCOPE_ID="${RUN_ID}"
export SMOLVLA_JEPA_EXPORT_OUT="/vol/bitbucket/aa6622/.cache/jepa_exports"
export SMOLVLA_JEPA_SOURCE="${SMOLVLA_JEPA_EXPORT_OUT}"
export SMOLVLA_DATA_ROOT="/vol/bitbucket/aa6622/pi05-cube-FT/datasets/bridged_v30_runs"
export SMOLVLA_JEPA_EXPORT_FULL_LATENTS=1
export SMOLVLA_FAIL_ON_PATH_REUSE=1
```

`common.sh` applies the run scope automatically (`.../run_${SMOLVLA_RUN_SCOPE_ID}`) to export/data/report/artifact roots, so do not pre-append `run_...` in these base paths.

3) Run strict preflight and a dry-run DAG render:

```bash
bash scripts/smolvla_vggflow/preflight_dependency_order.sh && \
bash scripts/smolvla_vggflow/submit_workflow.sh --dry-run
```

4) Submit branch-parallel and monitor:

```bash
python3 scripts/smolvla_vggflow/smolvla_workflow_launcher.py \
  --submit \
  --branch-parallel \
  --write-json "runs/workflow_${RUN_ID}.json" \
  --parallel-map-out "artifacts/parallel_submission_map_${RUN_ID}.json" && \
bash scripts/smolvla_vggflow/watch_workflow.sh \
  "runs/workflow_${RUN_ID}.json" \
  --poll 90 --max-retries 2 --auto-resubmit
```

Expected sequencing in branch-parallel mode: `stage02 || stage03`, then bridge/gate prerequisites, then `stage06 || stage08`, then `stage05`, then `stage09`.
This sequence is the recovery **target behavior**; it depends on the Task 6 launcher DAG patch and should be treated as intended ordering until that patch lands.

## Stage map

```text
scripts/slurm/stage00_preflight.slurm -> phase00_inventory
scripts/slurm/stage01_install_lerobot_mw.slurm -> phase01_env_topology
scripts/slurm/stage01b_install_metaworld.slurm -> phase04_metaworld_install (CPU sbatch; no --gres)
scripts/slurm/stage02_baseline_pushv3_eval.slurm -> phase06_baseline_eval
scripts/slurm/stage03_install_jepa_wms.slurm -> phase07_jepa_setup (optional rollout export if SMOLVLA_JEPA_EXPORT_ENABLED=1)
scripts/slurm/stage04_bridge_dataset_build.slurm -> phase08_bridge_design
scripts/slurm/stage05_train_stageA_real_only.slurm -> phase10_train_loop (SMOLVLA_TRAIN_VARIANT=a)
scripts/slurm/stage06_train_stageB_jepa_mix.slurm -> phase10_train_loop (SMOLVLA_TRAIN_VARIANT=b)
scripts/slurm/stage07_vgg_gatecheck.slurm -> phase09_vgg_gates
scripts/slurm/stage08_train_stageC_vgg_aux.slurm -> phase10_train_loop (SMOLVLA_TRAIN_VARIANT=c)
scripts/slurm/stage10_train_stageD_imagined.slurm -> phase10_train_loop (optional; SMOLVLA_TRAIN_VARIANT=d; not in default 11-job DAG)
scripts/slurm/stage09_final_eval_and_bundle.slurm -> phase13_post_train_eval, then phase12_reporting
scripts/slurm/stage11_slurm_orchestration.slurm -> phase11_slurm_orchestration (optional; append via SMOLVLA_STAGE11_ENABLED=1 — not mixed with --branch-parallel)
```

## Environment overrides

- `SMOLVLA_WORKSPACE_ROOT`, `SMOLVLA_REPO_ROOT`, `SMOLVLA_VGG_REPO`
- `SMOLVLA_BASELINE_CMD`: custom baseline evaluation command
- `phase01_env_topology` now seeds core runtime packages (`torch`, `typing_extensions`, `transformers`, plus optional stack extras) in all stacks (`lerobot_mw_py310`, `jepa_wms_py310`, `vggflow_py311`) to avoid missing-import failures before later phases.
- `SMOLVLA_TRAIN_STAGE_A_CMD`, `SMOLVLA_TRAIN_STAGE_B_CMD`, `SMOLVLA_TRAIN_STAGE_C_CMD`, `SMOLVLA_TRAIN_STAGE_D_CMD`
- `run_manifest.json`: phase10 sets `SMOLVLA_MANIFEST_STAGE_*` env vars per launched stage (then writes `stage_*_dataset_root` fields). It **clears** those env vars at the start of each phase10 run so a repeated local `run_stage.sh stage10_train_loop` in the same shell does not leak skipped-stage paths.
- `SMOLVLA_ENABLE_TRAIN_D=1`: run **TrainD** after earlier stages in the same `phase10` job (same VGG gate JSON as StageC; uses `--mode stageD` and `SMOLVLA_STAGE_D_DATA_ROOT`, default `${SMOLVLA_DATA_ROOT}/val`). Or submit `stage10_train_stageD_imagined.slurm` (`SMOLVLA_TRAIN_VARIANT=d`, `SMOLVLA_ENABLE_VGG=0`).
- `SMOLVLA_POLICY_REPORT_TO` (or `SMOLVLA_TRAIN_REPORT_TO`): telemetry targets passed through
  `train_smolvla_vggflow.py`. Current `lerobot` build supports `wandb` (`--wandb.enable/--wandb.mode`);
  `tensorboard` is ignored.
- `SMOLVLA_JEPA_SOURCE`: path to JEPA trajectory artifacts (directory or file). Model-only checkpoints
  like `*.pth.tar` are not trajectory datasets and are intentionally skipped.
- `SMOLVLA_JEPA_EXPORT_ENABLED`, `SMOLVLA_JEPA_EXPORT_OUT`, `SMOLVLA_JEPA_EXPORT_EPISODES`, `SMOLVLA_JEPA_EXPORT_MAX_STEPS`, `SMOLVLA_JEPA_EXPORT_SEED`, `SMOLVLA_JEPA_CEM_*`: phase07 runs `jepa_cem_paired_pushv3_export.py` (paired CEM + WM + push-v3 → `trajectories.pt`) when enabled. Legacy `jepa_metaworld_rollout_export.py` is unused.
- `SMOLVLA_CHECKPOINT_POLICY` (default `base_init_compare`), `SMOLVLA_RUN_SCOPE_ID`, `SMOLVLA_FAIL_ON_PATH_REUSE`, `SMOLVLA_FINAL_EVAL_CHECKPOINT`
- `phase08_bridge_design` now hard-fails if `SMOLVLA_JEPA_SOURCE` yields zero trajectory-like inputs
  (instead of writing empty placeholder datasets), which keeps stage10 from starting on invalid bridge data.
- `phase08_bridge_design` **requires** an executable lerobot env python, `scripts/smolvla_vggflow/bridge_builder.py`, a **zero exit** from `bridge_builder`, and a resulting `bridge_summary.json`; missing script/env or non-zero bridge build fails the stage with an explicit report line (no silent skip + vague “summary missing”).
- `SMOLVLA_ENABLE_VGG=1` to enable StageC wiring when `SMOLVLA_TRAIN_VARIANT` is `auto`, `a`, or `b` (combined jobs). **`SMOLVLA_TRAIN_VARIANT=c` always runs the StageC VGG path** in `run_stage.sh` (and `stage08_train_stageC_vgg_aux.slurm` exports `SMOLVLA_ENABLE_VGG=1` for clarity).
- `SMOLVLA_STAGE11_ENABLED=1` to include `stage11_slurm_orchestration.slurm` in generated workflows
- `SMOLVLA_VGG_GATE_JSON`: destination for gate report JSON
- `SMOLVLA_VGG_GATE_MIN_VALUE_GRAD`: minimum stable value-gradient norm for gate pass
- `SMOLVLA_VGG_GATE_MAX_BASE_FLOW_DIFF`: maximum allowed base-flow delta for gate pass
- `SMOLVLA_VGG_GATE_SKIP_FLOW_CHECK=1` to skip denoise rollout validation and emit a CPU-safe placeholder gate record
- `SMOLVLA_VGG_GATE_FORCE_VALIDATE=1` to force full flow validation even on CPU-limited hosts
- `SMOLVLA_JEPA_SMOKE_FORCE_VALIDATE=1` to force JEPA smoke rollout validation even when CUDA is unavailable
- `SMOLVLA_VGG_MATCH_WEIGHT`, `SMOLVLA_VGG_MATCH_WARMUP_STEPS`, `SMOLVLA_VGG_VALUE_HEAD_DIM`
- `SMOLVLA_VGG_TRACE_MAX_STEPS`, `SMOLVLA_VGG_TRACE_MAX_BATCH` for velocity trace emission
- `SMOLVLA_VGG_VALUE_HEAD_STEPS`, `SMOLVLA_VGG_VALUE_HEAD_SEED`, `SMOLVLA_VGG_VALUE_HEAD_MIN_GRAD`
- `SMOLVLA_VGG_MATCH_TRACE_CAP`
- `SMOLVLA_REQUIRE_GPU_STAGES=1` (default) to enforce queued GPU execution for required stages
- `SMOLVLA_ALLOW_CPU_RUN=1` to bypass guard for local CPU execution when debugging non-production paths
- `SMOLVLA_PARTITION_LIST` (default `a100,a40,a30,t4,a16`) for launcher GPU partition fallback

Reports, artifacts, locks, and logs are written under:
- `reports/`
- `artifacts/`
- `datasets/bridged/`

