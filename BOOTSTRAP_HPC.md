# Pi0.5 Cube Pipeline Bootstrap (New HPC)

This is the fastest path to recreate the current pi0.5 + OGBench cube setup after cloning this repo on another cluster.

## What Git Contains vs Not

Code in git (portable):
- `src/vla/pi05_ur5e.py`
- `src/envs/droid/observation_openpi_ur5e.py`
- `cube_dataset/pi05_joint_space/cube_env_joint_target.py`
- `cube_dataset/run_pi05_base_ur5e_rollouts.py`
- `cube_dataset/finetune_pi05_cube_single_v1/render_cube_single_dualcam.py`
- `scripts/pbs/run_pi05_base_ur5e_eval_gpu.pbs`
- `scripts/pbs/run_cube_single_dualcam_data_gen_gpu.pbs`
- `external/openpi/` (vendored OpenPI source)

Large artifacts not in git (must be regenerated/redownloaded):
- OGBench datasets (`cube_dataset/datasets/*.npz`)
- checkpoint blobs (`gs://openpi-assets/checkpoints/...`)
- rollout/data-gen videos (`*.mp4`)
- local caches/logs/checkpoints

## Prerequisites on Target HPC

- Linux + CUDA GPU node access
- `python3.11` available
- `git`
- `xvfb-run` available on compute nodes (or install package providing Xvfb)
- outbound internet access to:
  - PyPI/GitHub
  - Google Cloud Storage (`openpi-assets`)

## Quick Start

From the cloned repo root:

```bash
bash scripts/bootstrap_hpc_env.sh
bash scripts/verify_hpc_bootstrap.sh --mode quick
```

Full end-to-end check including OpenPI checkpoint load:

```bash
bash scripts/verify_hpc_bootstrap.sh --mode full
```

## Environment Contract (Important)

Current pipeline scripts assume Python lives at:

`stable-worldmodel/.venv/bin/python`

The bootstrap script creates this path to keep compatibility with existing PBS and rollout scripts.

## Recreate Data + Eval

### 1) Generate 100 dual-camera episodes (preflight + generation + QA) on GPU PBS

```bash
qsub scripts/pbs/run_cube_single_dualcam_data_gen_gpu.pbs
```

Optional overrides:
- `PI05_CUBE_RUN_ID`
- `PI05_CUBE_N_EPISODES`
- `PI05_CUBE_MAX_STEPS`

### 2) Run pi0.5-base UR5e evaluation on cube (20 episodes default)

```bash
qsub scripts/pbs/run_pi05_base_ur5e_eval_gpu.pbs
```

Optional overrides:
- `PI05_RUN_TAG`
- `PI05_NUM_EPISODES`
- `PI05_START_INDEX`
- `PI05_OGBENCH_CHECKPOINT`

## Manual Local Commands (No PBS)

Activate environment:

```bash
source stable-worldmodel/.venv/bin/activate
export PYTHONPATH="$PWD"
```

Preflight data generation:

```bash
MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1280x1024x24" \
  stable-worldmodel/.venv/bin/python \
  cube_dataset/finetune_pi05_cube_single_v1/render_cube_single_dualcam.py \
  --preflight
```

Smoke test rollout + OpenPI:

```bash
MUJOCO_GL=glfw xvfb-run -a -s "-screen 0 1024x768x24" \
  stable-worldmodel/.venv/bin/python \
  cube_dataset/run_pi05_base_ur5e_rollouts.py \
  --smoke-test --require-openpi \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base
```

## Troubleshooting

- `FATAL: xvfb-run not found`:
  - Install/load Xvfb package on the cluster.
- TLS/certificate errors downloading checkpoint:
  - Verify cert bundle is available; scripts already set `SSL_CERT_FILE` using `certifi` when possible.
- `OpenPI did not load`:
  - Re-run full verify mode and inspect the exact exception.
- MuJoCo headless render crash:
  - Use `xvfb-run` + `MUJOCO_GL=glfw` (current default in PBS scripts).
