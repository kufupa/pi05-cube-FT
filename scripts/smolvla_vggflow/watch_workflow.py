#!/usr/bin/env python3
"""Monitor a stage job chain and emit actionable retry notes."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


RETRY_HINTS = [
    ("OutOfMemoryError", "OOM", "reduce batch size / rollout length"),
    ("CUDA out of memory", "OOM", "reduce batch size / rollout length"),
    ("RuntimeError: torch.cuda", "GPU runtime", "check CUDA availability"),
    ("ModuleNotFoundError", "dependency", "pin missing package in lock file"),
    ("No module named", "dependency", "pin missing package in lock file"),
    ("Timeout", "infrastructure", "resubmit with longer time or smaller job"),
    ("Failed to fetch", "network", "retry download"),
]


def _sacct_state(job_id: str) -> Tuple[str, str]:
    out = subprocess.check_output(
        ["sacct", "-j", job_id, "-n", "-P", "-X", "-o", "State,ExitCode"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    if not out.strip():
        # fallback for jobs not yet in accounting
        return "PENDING", ""
    lines = [line for line in out.splitlines() if line.strip()]
    first = lines[0].split("|")
    state = first[0]
    exit_code = first[1] if len(first) > 1 else ""
    return state, exit_code


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
    if tag == "no-log":
        return "check scheduler state and rerun missing-output stage"
    return "manual inspection required before retry"


def monitor(job_ids: List[str], polling_seconds: int, max_retries: int) -> None:
    retries_left: Dict[str, int] = {jid: max_retries for jid in job_ids}
    completed = set()
    while len(completed) < len(job_ids):
        progress = True
        for jid in job_ids:
            if jid in completed:
                continue
            state, code = _sacct_state(jid)
            if state == "PENDING" or state.startswith("RUNNING"):
                progress = False
                continue
            if state.startswith("COMPLETED"):
                print(f"[watcher] {jid}: COMPLETED code={code}")
                completed.add(jid)
                continue
            if _state_is_final(state):
                print(f"[watcher] {jid}: {state} code={code}")
                if retries_left[jid] > 0:
                    reason = _diagnose(Path(f"logs/{jid}.log"))
                    print(
                        f"[watcher] retry allowed for {jid} reason={reason} "
                        f"remaining={retries_left[jid]}"
                    )
                    print(f"[watcher] retry action for {jid}: {_retry_message(reason)}")
                    retries_left[jid] -= 1
                    completed.add(jid)
                else:
                    print(f"[watcher] no retries remaining for {jid}; stopping chain")
                    completed.add(jid)
                continue
        if progress:
            break
        time.sleep(polling_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-ids", nargs="+", required=True)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    monitor(args.job_ids, args.poll, args.max_retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

