# SmolVLA + Meta-World + JEPA-WMS + VGG Workflow

This directory stores the execution artifacts for the full plan implementation.

## Quick entry points

- `run_stage.sh <stage>`: run a single numbered/plan phase locally.
- `submit_workflow.sh`: submit the full 10-stage Slurm DAG and write a `workflow_*.json`.
- `watch_workflow.sh <workflow.json>`: observe running job IDs from a workflow JSON.

## Stage map

```text
scripts/slurm/stage00_preflight.slurm -> phase00_inventory
scripts/slurm/stage01_install_lerobot_mw.slurm -> phase01_env_topology
scripts/slurm/stage02_baseline_pushv3_eval.slurm -> phase06_baseline_eval
scripts/slurm/stage03_install_jepa_wms.slurm -> phase07_jepa_setup
scripts/slurm/stage04_bridge_dataset_build.slurm -> phase08_bridge_design
scripts/slurm/stage05_train_stageA_real_only.slurm -> phase10_train_loop
scripts/slurm/stage06_train_stageB_jepa_mix.slurm -> phase10_train_loop
scripts/slurm/stage07_vgg_gatecheck.slurm -> phase09_vgg_gates
scripts/slurm/stage08_train_stageC_vgg_aux.slurm -> phase10_train_loop
scripts/slurm/stage09_final_eval_and_bundle.slurm -> phase12_reporting
scripts/slurm/stage11_slurm_orchestration.slurm -> phase11_slurm_orchestration
```

## Environment overrides

- `SMOLVLA_WORKSPACE_ROOT`, `SMOLVLA_REPO_ROOT`, `SMOLVLA_VGG_REPO`
- `SMOLVLA_BASELINE_CMD`: custom baseline evaluation command
- `SMOLVLA_TRAIN_STAGE_A_CMD`, `SMOLVLA_TRAIN_STAGE_B_CMD`, `SMOLVLA_TRAIN_STAGE_C_CMD`
- `SMOLVLA_ENABLE_VGG=1` to enable StageC wiring
- `SMOLVLA_STAGE11_ENABLED=1` to include `stage11_slurm_orchestration.slurm` in generated workflows
- `SMOLVLA_VGG_GATE_JSON`: destination for gate report JSON
- `SMOLVLA_VGG_GATE_MIN_VALUE_GRAD`: minimum stable value-gradient norm for gate pass
- `SMOLVLA_VGG_GATE_MAX_BASE_FLOW_DIFF`: maximum allowed base-flow delta for gate pass
- `SMOLVLA_VGG_MATCH_WEIGHT`, `SMOLVLA_VGG_MATCH_WARMUP_STEPS`, `SMOLVLA_VGG_VALUE_HEAD_DIM`
- `SMOLVLA_VGG_TRACE_MAX_STEPS`, `SMOLVLA_VGG_TRACE_MAX_BATCH` for velocity trace emission
- `SMOLVLA_VGG_VALUE_HEAD_STEPS`, `SMOLVLA_VGG_VALUE_HEAD_SEED`, `SMOLVLA_VGG_VALUE_HEAD_MIN_GRAD`
- `SMOLVLA_VGG_MATCH_TRACE_CAP`

Reports, artifacts, locks, and logs are written under:
- `reports/`
- `artifacts/`
- `datasets/bridged/`

