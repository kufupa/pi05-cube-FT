#!/usr/bin/env python3
"""Monitor OpenPI training stdout and emit step/timing logs."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


STEP_RE = re.compile(r"Step\s+(\d+):\s*(.*)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--idle-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = args.stdout_log.resolve()

    metrics_log = out_dir / "metrics_step.log"
    timing_log = out_dir / "timing_100step.log"
    eta_log = out_dir / "eta.log"
    ckpt_log = out_dir / "checkpoint_events.log"

    last_offset = 0
    last_step = None
    last_step_time = None
    started = time.time()
    last_activity = time.time()

    metrics_log.write_text("", encoding="utf-8")
    timing_log.write_text("", encoding="utf-8")
    eta_log.write_text("", encoding="utf-8")
    ckpt_log.write_text("", encoding="utf-8")
    seen_ckpt_steps: set[int] = set()

    while True:
        if stdout_log.exists():
            txt = stdout_log.read_text(encoding="utf-8", errors="replace")
            chunk = txt[last_offset:]
            if chunk:
                last_offset = len(txt)
                now = time.time()
                for line in chunk.splitlines():
                    m = STEP_RE.search(line)
                    if not m:
                        continue
                    step = int(m.group(1))
                    info = m.group(2).strip()
                    last_activity = now

                    with metrics_log.open("a", encoding="utf-8") as f:
                        f.write(f"{int(now)} step={step} {info}\n")

                    if last_step is not None and last_step_time is not None and step > last_step:
                        dt = now - last_step_time
                        ds = step - last_step
                        avg_step_sec = dt / ds
                        steps_per_sec = ds / dt if dt > 0 else 0.0
                        steps_to_ckpt = max(0, args.save_interval - (step % args.save_interval))
                        eta_ckpt = steps_to_ckpt * avg_step_sec
                        eta_end = max(0, args.total_steps - step) * avg_step_sec
                        with timing_log.open("a", encoding="utf-8") as f:
                            f.write(
                                f"{int(now)} step={step} elapsed_sec_last_100={dt:.3f} "
                                f"avg_step_sec_last_100={avg_step_sec:.6f} steps_per_sec_last_100={steps_per_sec:.6f} "
                                f"window_steps={ds}\n"
                            )
                        with eta_log.open("a", encoding="utf-8") as f:
                            f.write(
                                f"{int(now)} step={step} eta_to_next_ckpt_sec={eta_ckpt:.1f} "
                                f"eta_to_train_end_sec={eta_end:.1f}\n"
                            )
                    last_step = step
                    last_step_time = now

                    if step >= args.total_steps - 1:
                        return

        if args.checkpoint_dir is not None and args.checkpoint_dir.exists():
            now = time.time()
            for p in args.checkpoint_dir.iterdir():
                if not p.is_dir() or not p.name.isdigit():
                    continue
                s = int(p.name)
                if s in seen_ckpt_steps:
                    continue
                seen_ckpt_steps.add(s)
                with ckpt_log.open("a", encoding="utf-8") as f:
                    f.write(f"{int(now)} step={s} path={p}\n")

        if (time.time() - last_activity) > args.idle_timeout_sec and (time.time() - started) > args.idle_timeout_sec:
            # No new step logs for a long time: exit monitor to avoid hanging forever.
            return
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()

