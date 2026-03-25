#!/bin/bash
# Minimal setup script for VLAW DROID pipeline using `uv`

# Create uv environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv .venv
fi

# Activate the environment
source .venv/bin/activate

# Install core dependencies.
if [ -f "pyproject.toml" ]; then
    uv pip install -e .
fi

# Install openpi and its specific dependencies (including our patched local version)
uv pip install -e "external/openpi" \
    "tensorflow-cpu==2.15.0" \
    "tensorflow-datasets==4.9.9" \
    "git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"

uv pip install transformers diffusers torch torchvision accelerate qwen-vl-utils pyyaml
uv pip install h5py matplotlib

export PYTHONPATH=$PWD:$PYTHONPATH

echo "Environment setup complete."
