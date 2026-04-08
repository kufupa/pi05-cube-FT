#!/usr/bin/env python3
"""Utilities to render and optionally submit the SmolVLA Slurm DAG (serial or branch-parallel)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
SLURM_DIR = ROOT / "scripts" / "slurm"
STAGE_SCRIPTS = [
    "stage00_preflight.slurm",
    "stage01_install_lerobot_mw.slurm",
    "stage01b_install_metaworld.slurm",
    "stage02_baseline_pushv3_eval.slurm",
    "stage03_install_jepa_wms.slurm",
    "stage04_bridge_dataset_build.slurm",
    "stage06_train_stageB_jepa_mix.slurm",
    "stage07_vgg_gatecheck.slurm",
    "stage08_train_stageC_vgg_aux.slurm",
    "stage05_train_stageA_real_only.slurm",
    "stage09_final_eval_and_bundle.slurm",
]

GPU_REQUIRED_STAGES: Dict[str, bool] = {
    "stage00_preflight.slurm": True,
    "stage01_install_lerobot_mw.slurm": True,
    "stage01b_install_metaworld.slurm": False,
    "stage02_baseline_pushv3_eval.slurm": True,
    "stage03_install_jepa_wms.slurm": True,
    "stage04_bridge_dataset_build.slurm": True,
    "stage05_train_stageA_real_only.slurm": True,
    "stage06_train_stageB_jepa_mix.slurm": True,
    "stage07_vgg_gatecheck.slurm": True,
    "stage08_train_stageC_vgg_aux.slurm": True,
    "stage09_final_eval_and_bundle.slurm": True,
    # Optional manual submit (not in STAGE_SCRIPTS); keeps partition retry consistent if wired into custom DAGs.
    "stage10_train_stageD_imagined.slurm": True,
}


def _stage_requires_gpu(stage: str) -> bool:
    return GPU_REQUIRED_STAGES.get(stage, False)


def _parse_job_id(output: str) -> str:
    for token in reversed(output.strip().split()):
        if token.isdigit():
            return token
    raise RuntimeError(f"Unexpected sbatch output: {output}")


def _preferred_partitions() -> List[str]:
    raw = os.environ.get("SMOLVLA_PARTITION_LIST", "a100,a40,a30,t4,a16")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or ["a100", "a40", "a30", "t4", "a16"]


def _retryable_submit_error(stderr_text: str) -> bool:
    markers = (
        "Requested node configuration is not available",
        "Unable to allocate resources",
        "Node configuration",
        "Requested node count is not available",
        "Could not start job",
        "Unable to find suitable resources",
        "QOSMaxJobsPerUserLimit",
        "QOSMaxSubmitJobPerUserLimit",
        "Invalid partition specified",
    )
    lower = stderr_text.lower()
    return any(m.lower() in lower for m in markers)


def submit_stage(
    sbatch_path: Path,
    dependency: str | None = None,
    *,
    requires_gpu: bool = False,
) -> str:
    dep_arg: List[str] = []
    if dependency:
        dep_arg = [f"--dependency=afterok:{dependency}"]

    if not requires_gpu:
        cmd: List[str] = ["sbatch", *dep_arg, str(sbatch_path)]
        out = subprocess.check_output(cmd, text=True).strip()
        return _parse_job_id(out)

    partitions = _preferred_partitions()
    for partition in partitions:
        cmd = [
            "sbatch",
            *dep_arg,
            "--partition",
            partition,
            "--qos",
            "normal",
            "--gres",
            "gpu:1",
            str(sbatch_path),
        ]
        for attempt in range(1, 4):
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.returncode == 0:
                return _parse_job_id(proc.stdout)

            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            lower_output = output.lower()
            if "qosmaxsubmitjobperuserlimit" in lower_output:
                print(
                    f"[launcher] user QOS submit limit reached for {sbatch_path.name} "
                    f"on {partition}; attempt {attempt}/3, sleeping 30s before retry"
                )
                time.sleep(30)
                continue

            if partition == partitions[-1] or not _retryable_submit_error(output):
                raise RuntimeError(f"sbatch failed: {output.strip() or 'Unknown error'}")

            print(
                f"[launcher] partition {partition} unavailable for {sbatch_path.name}; trying next partition"
            )
            break
    raise RuntimeError(f"no partitions available for GPU stage {sbatch_path.name}")


def get_stage_scripts() -> List[str]:
    stages = STAGE_SCRIPTS.copy()
    if os.environ.get("SMOLVLA_STAGE11_ENABLED", "0") == "1":
        stages.append(os.environ.get("SMOLVLA_STAGE11_SLURM", "stage11_slurm_orchestration.slurm"))
    return stages


def submit_workflow_serial() -> List[str]:
    prev = None
    job_ids: List[str] = []
    for stage in get_stage_scripts():
        path = SLURM_DIR / stage
        if not path.exists():
            raise FileNotFoundError(f"missing stage script: {path}")
        job_id = submit_stage(path, prev, requires_gpu=_stage_requires_gpu(stage))
        job_ids.append(job_id)
        prev = job_id
    return job_ids


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def submit_workflow_branch_parallel(map_out: Path | None) -> List[str]:
    """Fan-out after stage01+metaworld; join before training; matches plan J1/J2-style deps (best-effort on one cluster)."""
    # Fixed 11-stage mainline (includes CPU stage01b metaworld). No SMOLVLA_STAGE11_ENABLED.
    stages = list(STAGE_SCRIPTS)
    if os.environ.get("SMOLVLA_STAGE11_ENABLED", "0") == "1":
        raise RuntimeError(
            "branch_parallel does not support SMOLVLA_STAGE11_ENABLED; use serial --submit or disable stage11."
        )
    if len(stages) != 11:
        raise RuntimeError(f"branch_parallel expects 11 Slurm stages, got {len(stages)}")
    for stage in stages:
        p = SLURM_DIR / stage
        if not p.exists():
            raise FileNotFoundError(f"missing stage script: {p}")

    s = {name: SLURM_DIR / name for name in stages}
    j00 = submit_stage(s[stages[0]], None, requires_gpu=_stage_requires_gpu(stages[0]))
    j01 = submit_stage(s[stages[1]], j00, requires_gpu=_stage_requires_gpu(stages[1]))
    j01b = submit_stage(s[stages[2]], j01, requires_gpu=_stage_requires_gpu(stages[2]))
    j02 = submit_stage(s[stages[3]], j01b, requires_gpu=_stage_requires_gpu(stages[3]))
    j03 = submit_stage(s[stages[4]], j01b, requires_gpu=_stage_requires_gpu(stages[4]))
    j04 = submit_stage(s[stages[5]], j03, requires_gpu=_stage_requires_gpu(stages[5]))
    join_train = f"{j02}:{j04}"
    j06 = submit_stage(s[stages[6]], join_train, requires_gpu=_stage_requires_gpu(stages[6]))
    j07 = submit_stage(s[stages[7]], j01b, requires_gpu=_stage_requires_gpu(stages[7]))
    join_c = f"{j07}:{j04}"
    j08 = submit_stage(s[stages[8]], join_c, requires_gpu=_stage_requires_gpu(stages[8]))
    join_a = f"{j06}:{j08}"
    j05 = submit_stage(s[stages[9]], join_a, requires_gpu=_stage_requires_gpu(stages[9]))
    join_final = f"{j05}:{j06}:{j08}"
    j09 = submit_stage(s[stages[10]], join_final, requires_gpu=_stage_requires_gpu(stages[10]))

    ids_order = [j00, j01, j01b, j02, j03, j04, j06, j07, j08, j05, j09]
    if map_out:
        rows = []
        for name, jid in zip(stages, ids_order):
            rows.append(
                {
                    "lane": "branch_parallel",
                    "stage": name,
                    "job_id": jid,
                    "variant": "",
                    "depends_on": [],
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        payload = {
            "schema_version": "parallel_submission_map_v1",
            "mode": "branch_parallel",
            "stages": rows,
        }
        _atomic_write_json(map_out, payload)
        print(f"[launcher] wrote {map_out}")
    return ids_order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--branch-parallel", action="store_true", help="Fan-out/fan-in deps after phase01 (see plan).")
    parser.add_argument("--write-json", default="")
    parser.add_argument(
        "--parallel-map-out",
        default="",
        help="With --submit --branch-parallel, write parallel_submission_map JSON here.",
    )
    args = parser.parse_args()

    if args.submit:
        if args.branch_parallel:
            pmap = Path(args.parallel_map_out) if args.parallel_map_out else ROOT / "artifacts" / "parallel_submission_map.json"
            job_ids = submit_workflow_branch_parallel(pmap)
        else:
            job_ids = submit_workflow_serial()
    else:
        job_ids = ["dry-run"]
        for stage in get_stage_scripts():
            job_ids.append(str(SLURM_DIR / stage))

    if args.write_json:
        payload: dict = {
            "stages": [],
            "schema_version": "workflow_outline_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": os.environ.get("RUN_ID") or os.environ.get("SMOLVLA_RUN_ID", ""),
        }
        if args.submit:
            for stage, job_id in zip(get_stage_scripts(), job_ids):
                payload["stages"].append(
                    {
                        "stage": stage,
                        "job_id": job_id,
                        "requires_gpu": _stage_requires_gpu(stage),
                    }
                )
        else:
            for stage in get_stage_scripts():
                payload["stages"].append(
                    {
                        "stage": stage,
                        "job_id": "<unsubmitted>",
                        "requires_gpu": _stage_requires_gpu(stage),
                    }
                )
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
