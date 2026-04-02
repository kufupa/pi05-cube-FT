#!/usr/bin/env python3
"""Select best checkpoint by success rate from eval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    search_root = args.search_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for rp in sorted(search_root.glob("**/eval_report.json")):
        try:
            r = _load_report(rp)
            r["_path"] = str(rp)
            reports.append(r)
        except Exception:
            continue

    if not reports:
        raise RuntimeError(f"No eval_report.json files found under {search_root}")

    reports_sorted = sorted(
        reports,
        key=lambda r: (float(r.get("success_rate", 0.0)), int(r.get("episodes_found", 0))),
        reverse=True,
    )
    best = reports_sorted[0]

    (out_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Handoff Summary",
        "",
        f"- Search root: `{search_root}`",
        f"- Selected checkpoint: `{best.get('checkpoint')}`",
        f"- Success rate: `{best.get('success_rate')}`",
        f"- Episodes: `{best.get('episodes_found')}/{best.get('episodes_requested')}`",
        "",
        "## Candidates (sorted)",
    ]
    for r in reports_sorted:
        md.append(
            f"- `{r.get('checkpoint')}`  sr={r.get('success_rate')} "
            f"episodes={r.get('episodes_found')}/{r.get('episodes_requested')} "
            f"report=`{r.get('_path')}`"
        )
    (out_dir / "handoff_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {(out_dir / 'best_checkpoint.json')}")
    print(f"Wrote {(out_dir / 'handoff_summary.md')}")


if __name__ == "__main__":
    main()

