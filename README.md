# PI0.5 VLA Project

## HPC Bootstrap (Clone-to-Run)
- Start here for new cluster setup: `BOOTSTRAP_HPC.md`
- One-shot environment setup: `bash scripts/bootstrap_hpc_env.sh`
- One-shot verification:
  - Quick: `bash scripts/verify_hpc_bootstrap.sh --mode quick`
  - Full (requires OpenPI checkpoint load): `bash scripts/verify_hpc_bootstrap.sh --mode full`

## PI0.5 LIBERO Checkpoint
- Checkpoint location: `checkpoints/pi05/libero_base/`
- Download instructions: Run `bash scripts/download_pi05.sh` (downloads ~2GB)
- Verification: Run `python -m src.vla.smoke_test_pi05` to verify.

## World and Reward Models
- Checkpoint locations: `checkpoints/world_model/ctrl_world_base/` and `checkpoints/reward_model/qwen2vl7b_base/`
- Download instructions: Run `bash scripts/download_world_reward.sh` (downloads ~10GB total)
- Verification: Run `python -m src.world_model.smoke_test` and `python -m src.reward_model.smoke_test` to verify.
