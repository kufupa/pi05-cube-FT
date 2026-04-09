# phase07_jepa_setup
Task: push-v3
Executed: 2026-04-09T00:47:50Z

- [PASS] JEPA smoke unroll skipped (single hub load via trajectory export)
  - 2026-04-09T00:48:02Z

## JEPA smoke status
- status: SKIPPED_SINGLE_HUB_LOAD_VIA_EXPORT
- message: phase07 export loads JEPA-WM once; smoke check omitted when SMOLVLA_JEPA_SKIP_SMOKE_WHEN_EXPORT=1.
- checked_at: 2026-04-09T00:48:02Z

- [PASS] JEPA-WMS environment prepared
  - 2026-04-09T00:48:02Z

## JEPA CEM paired push-v3 trajectory export
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
`torch_dtype` is deprecated! Use `dtype` instead!
[cem_paired_export] policy load_vlm raw='1' parsed=True
[cem_paired_export] policy device raw='default' resolved='cuda'
[cem_paired_export] policy config override: load_vlm_weights=True device=cuda
[cem_paired_export] policy config override applied=True
Loading  HuggingFaceTB/SmolVLM2-500M-Instruct weights ...
Loading weights:   0%|          | 0/489 [00:00<?, ?it/s]Loading weights:  56%|█████▌    | 272/489 [00:00<00:00, 2699.45it/s]Loading weights: 100%|██████████| 489/489 [00:00<00:00, 3433.98it/s]
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/pygame/pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
WARNING:Cluster utils:JEPAWM_DSET environment variable not set. Dataset paths will need to be provided manually or set the environment variable. See README.md for setup instructions.
Using cache found in /vol/bitbucket/aa6622/.cache/torch/hub/facebookresearch_dinov2_main
[cem_paired_export] loaded policy config: load_vlm_weights=True
[cem_paired_export] wm action_dim candidates=[4, 20], planner_action_dim=20 (env uses 4-D MT1 actions)
[cem_paired_export][rss] ep=0 step=0 rss_gb=2.682
[cem_paired_export][rss] ep=0 step=25 rss_gb=3.050
[cem_paired_export][rss] ep=0 step=50 rss_gb=3.066
[cem_paired_export][rss] ep=0 step=75 rss_gb=3.084
[cem_paired_export][rss] ep=0 step=100 rss_gb=3.101
[cem_paired_export][rss] ep=0 step=125 rss_gb=3.117
[cem_paired_export][rss] ep=0 step=150 rss_gb=3.133
[cem_paired_export][rss] ep=0 step=175 rss_gb=3.151
[cem_paired_export][rss] ep=1 step=0 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=25 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=50 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=75 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=100 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=125 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=150 rss_gb=3.643
[cem_paired_export][rss] ep=1 step=175 rss_gb=3.644
[cem_paired_export][rss] ep=2 step=0 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=25 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=50 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=75 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=100 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=125 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=150 rss_gb=3.734
[cem_paired_export][rss] ep=2 step=175 rss_gb=3.734
[cem_paired_export][rss] ep=3 step=0 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=25 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=50 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=75 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=100 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=125 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=150 rss_gb=3.756
[cem_paired_export][rss] ep=3 step=175 rss_gb=3.756
[cem_paired_export][rss] ep=4 step=0 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=25 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=50 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=75 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=100 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=125 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=150 rss_gb=3.762
[cem_paired_export][rss] ep=4 step=175 rss_gb=3.762
[cem_paired_export][rss] ep=5 step=0 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=25 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=50 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=75 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=100 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=125 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=150 rss_gb=3.764
[cem_paired_export][rss] ep=5 step=175 rss_gb=3.764
[cem_paired_export][rss] ep=6 step=0 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=25 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=50 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=75 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=100 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=125 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=150 rss_gb=3.786
[cem_paired_export][rss] ep=6 step=175 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=0 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=25 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=50 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=75 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=100 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=125 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=150 rss_gb=3.786
[cem_paired_export][rss] ep=7 step=175 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=0 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=25 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=50 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=75 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=100 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=125 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=150 rss_gb=3.786
[cem_paired_export][rss] ep=8 step=175 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=0 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=25 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=50 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=75 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=100 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=125 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=150 rss_gb=3.786
[cem_paired_export][rss] ep=9 step=175 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=0 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=25 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=50 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=75 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=100 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=125 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=150 rss_gb=3.786
[cem_paired_export][rss] ep=10 step=175 rss_gb=3.786
[cem_paired_export][rss] ep=11 step=0 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=25 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=50 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=75 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=100 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=125 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=150 rss_gb=3.787
[cem_paired_export][rss] ep=11 step=175 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=0 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=25 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=50 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=75 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=100 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=125 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=150 rss_gb=3.787
[cem_paired_export][rss] ep=12 step=175 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=0 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=25 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=50 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=75 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=100 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=125 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=150 rss_gb=3.787
[cem_paired_export][rss] ep=13 step=175 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=0 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=25 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=50 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=75 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=100 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=125 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=150 rss_gb=3.787
[cem_paired_export][rss] ep=14 step=175 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=0 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=25 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=50 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=75 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=100 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=125 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=150 rss_gb=3.787
[cem_paired_export][rss] ep=15 step=175 rss_gb=3.787
[cem_paired_export] wrote 16 episodes -> /vol/bitbucket/aa6622/.cache/jepa_exports/run_20260408_025416/episodes
```

- [PASS] JEPA rollout export written under /vol/bitbucket/aa6622/.cache/jepa_exports/run_20260408_025416
  - 2026-04-09T01:54:41Z
