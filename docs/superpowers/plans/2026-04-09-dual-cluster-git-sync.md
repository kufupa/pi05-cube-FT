# Dual-cluster Git sync (HPC ↔ GitHub)

This repo is used from **two HPC clusters**. GitHub should carry **source, configs, docs, and small run metadata** only. **Checkpoints, datasets, videos, and large run trees stay local** (see root `.gitignore`).

## What belongs on GitHub

| Track | Do not track |
|--------|----------------|
| Python/shell sources, configs, small docs | `datasets/`, bulk `artifacts/`, most `runs/*.json` |
| Curated per-run folders under `reports/run_<run-id>/` | Ad-hoc `reports/*.md` at repo root |
| Allowlisted tiny files (see `.gitignore` negation rules) | `*.pt`, `*.safetensors`, `*.mp4`, archives, caches |

When you add a **new** run’s metadata to git, update `.gitignore` **allowlist** entries for that run (e.g. one `runs/workflow_<run>.json` path and `artifacts/.../eval_info.json` if needed), then commit intentionally.

## Branch naming

- **Feature / experiment work:** `cluster/<cluster-name>/run_<run-id>`  
  Example: `cluster/linnet/run_20260408_025416`
- **`main`:** integration branch; should stay fast-forwardable from reviewed work.

Work on a **run branch**; merge to `main` after checks (PR or controlled fast-forward).

## Sync cycle (each cluster)

1. `git fetch origin`
2. Start from latest shared base: `git checkout main && git pull --ff-only` (or `git rebase origin/main` on your run branch after updating `main`)
3. `git checkout -b cluster/<host>/run_<id>` (or reuse existing)
4. Commit **small** changes often; **never** `git add datasets/` or bulk `artifacts/`
5. **Before every push:** run `./scripts/smolvla_vggflow/git_prepush_size_check.sh`
6. `git push -u origin <branch>`
7. Merge to `main` on GitHub (or locally after review) using **merge** or **rebase + fast-forward**, not force-pushes to `main`

## Conflicts

- Prefer **`git fetch --all`** then **`git rebase origin/main`** on your branch (or merge `origin/main` into your branch if your team standard is merge).
- **Do not `git push --force` to `main`.**
- If `.gitignore` or allowlists conflict, resolve by keeping **broad ignores** and **explicit negations** for new small metadata only.

## Pre-push size gate

Run from repo root:

```bash
./scripts/smolvla_vggflow/git_prepush_size_check.sh
```

Optional environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `GIT_PREPUSH_MAX_BYTES` | `5242880` (5 MiB) | Fail if any **blob** at `HEAD` exceeds this |
| `GIT_PREPUSH_WARN_BYTES` | `524288` (512 KiB) | Warn-only threshold |

Install as a Git hook (optional):

```bash
ln -sf ../../scripts/smolvla_vggflow/git_prepush_size_check.sh .git/hooks/pre-push
```

## Quick checklist

- [ ] `git status` shows only intended paths
- [ ] No new large binaries under `artifacts/` / `datasets/`
- [ ] Size check script passes
- [ ] Pushing a **run branch**, not a dirty `main` with unreviewed bulk files
