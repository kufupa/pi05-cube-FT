#!/usr/bin/env bash
# Pre-push / pre-commit guardrail: fail if HEAD contains oversized blobs.
# See: docs/superpowers/plans/2026-04-09-dual-cluster-git-sync.md
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "git_prepush_size_check: not inside a git repository" >&2
  exit 2
}
cd "$ROOT"

MAX_BYTES="${GIT_PREPUSH_MAX_BYTES:-$((5 * 1024 * 1024))}"
WARN_BYTES="${GIT_PREPUSH_WARN_BYTES:-$((512 * 1024))}"

REF="${1:-HEAD}"

fail=0
while read -r mode type object path; do
  [[ "$type" == "blob" ]] || continue
  size="$(git cat-file -s "$object")"
  if (( size > MAX_BYTES )); then
    printf 'ERROR: blob too large (%d bytes > %d): %s\n' "$size" "$MAX_BYTES" "$path" >&2
    fail=1
  elif (( size > WARN_BYTES )); then
    printf 'WARN: large blob (%d bytes): %s\n' "$size" "$path" >&2
  fi
done < <(git ls-tree -r "$REF")

if (( fail )); then
  echo "git_prepush_size_check: failed (set GIT_PREPUSH_MAX_BYTES to override)" >&2
  exit 1
fi

echo "git_prepush_size_check: OK (ref=$REF max=${MAX_BYTES}B warn=${WARN_BYTES}B)"
exit 0
