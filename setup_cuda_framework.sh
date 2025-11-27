#!/bin/bash
set -e

if [ ! -f "setup.py" ]; then
    echo "Error: Run from mamba-optimized-cuda directory"
    exit 1
fi

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir -q
pip install triton einops transformers packaging ninja --no-cache-dir -q
pip install -e . --no-build-isolation --no-deps
python3 -c "import torch; import triton; import mamba_ssm; print('Setup complete')"
