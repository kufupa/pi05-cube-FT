#!/usr/bin/env python3
"""Monitor Slurm jobs; optional auto-resubmit on retryable node/GPU allocation failures."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
import os
from typing import Dict, List, Tuple


RETRY_HINTS = [
    ("OutOfMemoryError", "OOM", "reduce batch size / rollout length"),
    ("CUDA out of memory", "OOM", "reduce batch size / rollout length"),
    ("RuntimeError: torch.cuda", "GPU runtime", "check CUDA availability"),
    ("torch.cuda.is_available", "no-gpu", "submit GPU stage through Slurm with --gres=gpu:1"),
    ("Requested node configuration is not available", "node_resources", "retry with next partition in SMOLVLA_PARTITION_LIST"),
    ("Unable to allocate resources", "node_resources", "retry with next partition in SMOLVLA_PARTITION_LIST"),
    ("Could not start job", "node_resources", "retry with relaxed partition preferences"),
    ("ModuleNotFoundError", "dependency", "pin missing package in lock file"),
    ("No module named", "dependency", "pin missing package in lock file"),
    ("Timeout", "infrastructure", "resubmit with longer time or smaller job"),
    ("Failed to fetch", "network", "retry download"),
]

FALLBACK_PARTITIONS = [part.strip() for part in os.environ.get("SMOLVLA_PARTITION_LIST", "a100,a40,a30,t4,a16").split(",") if part.strip()]


def _parse_job_id(output: str) -> str:
    for token in reversed(output.strip().split()):
        if token.isdigit():
            return token
    raise RuntimeError(f"Unexpected sbatch output: {output}")


def _scontrol_job_attrs(job_id: str) -> Dict[str, str]:
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", "-d", "-o", job_id],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return {}
    attrs: Dict[str, str] = {}
    for token in out.split():
        if "=" in token:
            key, value = token.split("=", 1)
            attrs[key] = value
    return attrs


def _scontrol_state(job_id: str) -> Tuple[str, str]:
    attrs = _scontrol_job_attrs(job_id)
    state = attrs.get("JobState", "UNKNOWN")
    exit_code = attrs.get("ExitCode", "")
    return state, exit_code


def _job_log_path(job_id: str) -> Path:
    attrs = _scontrol_job_attrs(job_id)
    std_out = attrs.get("StdOut")
    if std_out and std_out not in {"(null)", "N/A"}:
        return Path(std_out.replace("%j", job_id))
    return Path(f"logs/{job_id}.log")


def _job_had_gpu_gres(attrs: Dict[str, str]) -> bool:
    """True if Slurm recorded GPU use (Gres and/or TRES fields; sites vary)."""
    chunks: List[str] = []
    for key in ("Gres", "AllocTRES", "ReqTRES"):
        raw = (attrs.get(key) or "").strip()
        if raw and raw not in {"(null)", "N/A"}:
            chunks.append(raw)
    if not chunks:
        return False
    blob = " ".join(chunks).lower()
    return "gpu" in blob or "gres/gpu" in blob


def _retry_command(job_id: str, reason: str) -> str:
    if reason not in {"node_resources", "no-gpu"}:
        return ""
    attrs = _scontrol_job_attrs(job_id)
    command = attrs.get("Command")
    if not command or command in {"(null)", "N/A"}:
        return ""
    command = command.replace('\\"', "")
    part_list = ",".join(FALLBACK_PARTITIONS) if FALLBACK_PARTITIONS else "a100,a40,a30,t4,a16"
    prefix = f'SMOLVLA_PARTITION_LIST="{part_list}" sbatch'
    # Always retry no-gpu class with a GPU request. For node_resources, only add --gres when the
    # original job was GPU (avoids breaking CPU stages like stage01b_install_metaworld).
    add_gres = reason == "no-gpu" or _job_had_gpu_gres(attrs)
    if add_gres:
        return f"{prefix} --gres=gpu:1 {command}"
    return f"{prefix} {command}"


def _sacct_state(job_id: str) -> Tuple[str, str]:
    """Resolve State/ExitCode from sacct, preferring the aggregate row for this job id.

    Slurm often emits multiple lines (e.g. 12345 and 12345.batch); the first line is not always the
    job shell row we care about for exit codes.
    """
    target = job_id.strip()
    try:
        out = subprocess.check_output(
            ["sacct", "-j", target, "-n", "-P", "-X", "-o", "JobID,State,ExitCode"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return _scontrol_state(job_id)
    if not out.strip():
        return "PENDING", ""
    lines = [line for line in out.splitlines() if line.strip()]
    rows: List[List[str]] = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            rows.append(parts)
    if not rows:
        return "PENDING", ""

    def to_pair(parts: List[str]) -> Tuple[str, str]:
        st = parts[1]
        ec = parts[2] if len(parts) > 2 else ""
        return st, ec

    for parts in rows:
        if parts[0] == target:
            return to_pair(parts)
    for parts in rows:
        jid = parts[0]
        if jid.startswith(f"{target}."):
            return to_pair(parts)
    return to_pair(rows[0])


def _state_is_final(state: str) -> bool:
    return any(state.startswith(x) for x in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"])


def _diagnose(log_path: Path) -> str:
    if not log_path.exists():
        return "no-log"
    data = log_path.read_text(encoding="utf-8", errors="ignore")
    for needle, tag, _ in RETRY_HINTS:
        if needle in data:
            return tag
    return "unknown"


def _retry_message(tag: str) -> str:
    if tag == "OOM":
        return "reduce batch size / rollout length before retry"
    if tag == "dependency":
        return "pin missing package versions and rerun with env lock refresh"
    if tag == "infrastructure":
        return "increase walltime / avoid node saturation"
    if tag == "network":
        return "retry after network or mirror stabilization"
    if tag == "GPU runtime":
        return "verify CUDA runtime compatibility and restart with fresh env"
    if tag == "no-gpu":
        return "resubmit on GPU partition with --gres=gpu:1"
    if tag == "node_resources":
        return "retry with next partition preference in SMOLVLA_PARTITION_LIST"
    if tag == "no-log":
        return "check scheduler state and rerun missing-output stage"
    return "manual inspection required before retry"


def monitor(
    job_ids: List[str],
    polling_seconds: int,
    max_retries: int,
    *,
    auto_resubmit: bool,
) -> int:
    """Return 0 if all jobs eventually succeed; 1 on hard failure."""
    active = list(job_ids)
    retries_left: Dict[str, int] = {jid: max_retries for jid in job_ids}

    while active:
        time.sleep(polling_seconds)
        next_round: List[str] = []
        for jid in active:
            state, code = _sacct_state(jid)
            if state in {"PENDING", "SUSPENDED"} or state.startswith("RUNNING"):
                next_round.append(jid)
                continue
            if state.startswith("COMPLETED"):
                ec0 = (code or "0:0").split(":")[0].strip()
                try:
                    ok = int(ec0) == 0
                except ValueError:
                    ok = True
                if ok:
                    print(f"[watcher] {jid}: COMPLETED code={code}")
                    continue
                print(f"[watcher] {jid}: COMPLETED with non-zero exit code={code} (treating as failure)")
                state = "FAILED"

            if _state_is_final(state):
                print(f"[watcher] {jid}: {state} code={code}")
                log_path = _job_log_path(jid)
                reason = _diagnose(log_path)
                print(f"[watcher] failure reason class for {jid}: {reason} ({_retry_message(reason)})")
                requeue_cmd = _retry_command(jid, reason)
                if requeue_cmd:
                    print(f"[watcher] requeue hint: {requeue_cmd}")
                if retries_left.get(jid, 0) > 0 and auto_resubmit and requeue_cmd:
                    retries_left[jid] -= 1
                    proc = subprocess.run(requeue_cmd, shell=True, capture_output=True, text=True)
                    if proc.returncode == 0 and proc.stdout:
                        try:
                            new_jid = _parse_job_id(proc.stdout)
                            print(f"[watcher] auto-resubmitted {jid} -> {new_jid} (retries_left for {jid}={retries_left[jid]})")
                            retries_left[new_jid] = max_retries
                            next_round.append(new_jid)
                            continue
                        except RuntimeError as exc:
                            print(f"[watcher] could not parse new job id: {exc}")
                    else:
                        print(f"[watcher] auto-resubmit command failed: {(proc.stderr or proc.stdout or '').strip()}")
                print(
                    f"[watcher] job {jid} failed with no successful recovery "
                    f"(auto_resubmit={auto_resubmit}, had_requeue_cmd={bool(requeue_cmd)})"
                )
                return 1
            next_round.append(jid)
        active = next_round
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-ids", nargs="+", required=True)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--auto-resubmit",
        action="store_true",
        help="On node_resources/no-gpu class failures, run sbatch requeue hint and track the new job id. "
        "CPU-only jobs omit --gres on node_resources retry; GPU inferred from Gres/AllocTRES/ReqTRES.",
    )
    args = parser.parse_args()

    code = monitor(args.job_ids, args.poll, args.max_retries, auto_resubmit=args.auto_resubmit)
    if code != 0:
        return code
    print("[watcher] all jobs completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
