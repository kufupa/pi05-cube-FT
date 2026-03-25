#!/bin/bash
source ~/miniforge3/bin/activate robocasa_env
pip install -q huggingface_hub[cli]
huggingface-cli download --repo-type model lerobot/pi05_libero_base \
  --local-dir checkpoints/pi05/libero_base \
  --local-dir-use-symlinks False
