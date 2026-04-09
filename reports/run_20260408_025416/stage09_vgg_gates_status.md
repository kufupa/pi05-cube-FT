# phase09_vgg_gates
Executed: 2026-04-09T09:30:52Z

## SmolVLA velocity/value-gate validation
```
Loading  HuggingFaceTB/SmolVLM2-500M-Instruct weights ...
{
  "schema_version": "smolvla_gate_v1",
  "emit_utc": "2026-04-09T09:35:52.189430+00:00",
  "init_checkpoint": "jadechoghari/smolvla_metaworld",
  "slurm_job_id": "229848",
  "baseline_status": "baseline_model_copied",
  "velocity_trace_ok": true,
  "velocity_trace_skipped": false,
  "contract_ok": true,
  "contract_reasons": [],
  "velocity_shape": [
    2,
    50,
    32
  ],
  "base_flow_diff_max": 0.0,
  "base_flow_diff_mean": 0.0,
  "value_head_ok": true,
  "value_head_grad_norm": 4.732537854579277e-05,
  "value_head_value": -0.0005586810293607414,
  "device": "cuda",
  "value_head_grad_ok": true,
  "base_flow_ok": true,
  "gate_ok": true,
  "gate_reasons": [],
  "trace_path": "/vol/bitbucket/aa6622/pi05-cube-FT/reports/run_20260408_025416/vgg_velocity_trace.json"
}
```

- [PASS] VGG gate evaluation passed
  - 2026-04-09T09:35:53Z
All VGG gates passed.
