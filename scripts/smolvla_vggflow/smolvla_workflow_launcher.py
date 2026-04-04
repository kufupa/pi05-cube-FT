#!/usr/bin/env python3
"""Utilities to render and optionally submit the SmolVLA Slurm DAG."""

from __future__ import annotations

import argparse
import json
import subprocess
import os
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
SLURM_DIR = ROOT / "scripts" / "slurm"
STAGE_SCRIPTS = [
    "stage00_preflight.slurm",
    "stage01_install_lerobot_mw.slurm",
    "stage02_baseline_pushv3_eval.slurm",
    "stage03_install_jepa_wms.slurm",
    "stage04_bridge_dataset_build.slurm",
    "stage05_train_stageA_real_only.slurm",
    "stage06_train_stageB_jepa_mix.slurm",
    "stage07_vgg_gatecheck.slurm",
    "stage08_train_stageC_vgg_aux.slurm",
    "stage09_final_eval_and_bundle.slurm",
]


def get_stage_scripts() -> List[str]:
    stages = STAGE_SCRIPTS.copy()
    if os.environ.get("SMOLVLA_STAGE11_ENABLED", "0") == "1":
        stages.append(os.environ.get("SMOLVLA_STAGE11_SLURM", "stage11_slurm_orchestration.slurm"))
    return stages


def submit_stage(sbatch_path: Path, dependency: str | None = None) -> str:
    cmd: List[str] = ["sbatch"]
    if dependency:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd.append(str(sbatch_path))
    out = subprocess.check_output(cmd, text=True).strip()
    # Expected format: Submitted batch job <id>
    for token in reversed(out.split()):
        if token.isdigit():
            return token
    raise RuntimeError(f"Unexpected sbatch output: {out}")


def submit_workflow() -> List[str]:
    prev = None
    job_ids: List[str] = []
    for stage in get_stage_scripts():
        path = SLURM_DIR / stage
        if not path.exists():
            raise FileNotFoundError(f"missing stage script: {path}")
        job_id = submit_stage(path, prev)
        job_ids.append(job_id)
        prev = job_id
    return job_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--write-json", default="")
    args = parser.parse_args()

    if args.submit:
        job_ids = submit_workflow()
    else:
        job_ids = ["dry-run"]
        for stage in get_stage_scripts():
            job_ids.append(str(SLURM_DIR / stage))

    if args.write_json:
        payload = {"stages": []}
        if args.submit:
            for stage, job_id in zip(get_stage_scripts(), job_ids):
                payload["stages"].append({"stage": stage, "job_id": job_id})
        else:
            for stage in get_stage_scripts():
                payload["stages"].append({"stage": stage, "job_id": "<unsubmitted>"})
        Path(args.write_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"workflow.json written: {args.write_json}")
    else:
        print("Stages:")
        for stage in get_stage_scripts():
            print(f"- {stage}")
        if args.submit:
            print("Job ids:")
            for item in job_ids:
                print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

