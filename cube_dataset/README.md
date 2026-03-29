# OGBench cube datasets: download, top-33, videos

Run [`download_and_replay_ogbench.py`](download_and_replay_ogbench.py) with **stable-worldmodel’s virtualenv** so `ogbench` and MuJoCo match your project:

```bash
cd /path/to/project/stable-worldmodel
.venv/bin/python ../cube_dataset/download_and_replay_ogbench.py --preflight
export MUJOCO_GL=egl   # typical on GPU compute nodes; omit on desktop with DISPLAY
.venv/bin/python ../cube_dataset/download_and_replay_ogbench.py
```

Outputs (under this directory):

- `datasets/` — `.npz` from RAIL + merged triple shards
- `top_episodes/*-top33.h5` — stacked top episodes + `goals`, optional `qpos`/`qvel`
- `videos_frames/<family>/` — `ep_*.mp4` and `ep_*_frames.npz`

PBS example: [`../scripts/pbs/run_ogbench_replay.pbs`](../scripts/pbs/run_ogbench_replay.pbs).

**Headless:** without `DISPLAY`, set `MUJOCO_GL` before the full run (not only preflight). Login-node preflight may skip render to avoid MuJoCo/GL abort; imports and a tiny MP4 write are still checked.

**HTTPS:** if RAIL downloads fail TLS verification, the script retries with an unverified context (see `OGBENCH_INSECURE_SSL` in the script). Prefer fixing `SSL_CERT_FILE` on your cluster.

---

## GT export + Qwen3-VL instructions (`gt_export/`)

This path keeps **all** pipeline outputs under `gt_export/run_<UTC_stamp>_<PBS_JOBID>/` (HDF5, full-episode MP4s, trimmed clip MP4s, `manifest.jsonl`, `rejects.jsonl`, later `instructions.jsonl`, `vlm_meta.json`, `run_meta.json`). It does **not** write `videos_frames/` or `top_episodes/`. The legacy script’s `main()` is never called; [`gt_export.py`](gt_export.py) only **imports** helpers from [`download_and_replay_ogbench.py`](download_and_replay_ogbench.py).

**Two interpreters (do not merge venvs without an audit):**

| Step | Command |
|------|---------|
| Export (ogbench, MuJoCo, H5, MP4) | `stable-worldmodel/.venv/bin/python` |
| VLM instructions | `cd <project> && uv run python ...` |

**Headless login / broken EGL:** MuJoCo needs either a real `DISPLAY` or working `MUJOCO_GL` (often `egl` on GPU nodes). If EGL/OSMesa fails, use **Xvfb** and **`MUJOCO_GL=glfw`** (GLFW uses the virtual X server):

```bash
cd /path/to/project
# Optional when RAIL HTTPS fails behind TLS interception:
# export OGBENCH_INSECURE_SSL=1

xvfb-run -a -s "-screen 0 1024x768x24" env MUJOCO_GL=glfw \
  stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py --smoke

# Or let the script re-exec under xvfb-run (it sets MUJOCO_GL=glfw for the child):
stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py --headless-xvfb --smoke
```

**Smoke checks (login / interactive — not inside the PBS job):**

```bash
cd /path/to/project
# With GPU / working EGL:
export MUJOCO_GL=egl
stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py --smoke
# VLM smoke: always use uv from project root (not system python3 — avoids missing libffi / wrong torch):
uv run python scripts/generate_cube_gt_instructions.py --smoke
```

**Full export (writes a new `run_*` directory):**

```bash
cd /path/to/project
export MUJOCO_GL=egl   # on GPU nodes where EGL works
stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py
# headless without EGL (pick one):
# xvfb-run -a -s "-screen 0 1024x768x24" stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py
# stable-worldmodel/.venv/bin/python cube_dataset/gt_export.py --headless-xvfb
# optional: --top-k 33 --npz cube_dataset/datasets/cube-single-play-v0.npz
```

The last line of export stdout is `GT_EXPORT_RUN_DIR=...`; the same path is stored in `cube_dataset/gt_export/_last_run_dir.txt`.

**Instructions (after export, same machine as GPU):**

```bash
cd /path/to/project
uv run python scripts/generate_cube_gt_instructions.py --run-dir /path/to/run_20260329_120000_12345.hpc
```

Production: require CUDA; do **not** set `VLAW_MOCK_REWARD`. For smoke only, you may use `VLAW_MOCK_REWARD=1` so the instruction script emits placeholder text without loading the full model.

If `uv run python -c "import torch"` fails with `libffi.so.8` (or similar), load your site’s **libffi** module first (e.g. on this cluster: `module load libffi/3.4.4-GCCcore-13.2.0`), then `uv run python …`. The PBS script attempts that load non-fatally. Do not use bare `python3` for the VLM step.

**PBS (export + VLM, no mocks, no `--smoke`):**

```bash
cd /path/to/project
qsub scripts/pbs/run_cube_gt_export.pbs
```

Logs default under `cube_dataset/gt_export/_pbs_logs/` (job stdout/stderr); the job copies that log to `run_*/pbs_stdout.log` and sets `pbs_stdout_log` in `run_meta.json`.

The PBS script runs the **export** step under `xvfb-run` when `DISPLAY` is unset and `xvfb-run` exists (compute nodes often have no X). Install or `module load` Xvfb on your cluster if needed.
