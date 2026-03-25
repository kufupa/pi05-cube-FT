# VLAW Audit Report Template

## Scope
- Commit:
- Date:
- Auditor:
- Environment:

## Severity Summary
- Critical:
- High:
- Medium:
- Low:

## Findings
### [Severity] Finding title
- Evidence:
- Impact:
- Root cause:
- Proposed fix:
- Verification plan:

## Dependency Root-Cause Validation
- OpenPI install chain:
- Python headers/toolchain check:
- Chosen mitigation path:

## PBS Reliability
- `run_smoke_gpu.pbs` result:
- `run_openpi_real_eval.pbs` result:
- `run_vlaw_loop.pbs` result:
- Metadata completeness:

## Metrics Contract
- `results_base.json` schema:
- `results_base_real.json` schema:
- `results_vlaw.json` schema:
- Plot policy (Base vs Base-Real):

## Artifact Trace
- Produced artifacts:
- Missing artifacts:
- Consistency mismatches:

## Paper-Fidelity Matrix
| Stage | Status (Faithful/Approximate/Placeholder) | Notes |
|---|---|---|
| Real rollout collection |  |  |
| World-model post-training |  |  |
| Reward-model filtering |  |  |
| Synthetic generation |  |  |
| Policy post-training |  |  |

## Acceptance Gate
- [ ] Login preflight passed
- [ ] Login smoke passed
- [ ] Interface contracts passed
- [ ] GPU smoke passed
- [ ] OpenPI real baseline produced
- [ ] VLAW loop produced
- [ ] Artifact verification passed
