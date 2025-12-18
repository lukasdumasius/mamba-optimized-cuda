#!/bin/bash
set -e

if [ ! -f "setup.py" ]; then
    echo "Error: Run from mamba-optimized-cuda directory"
    exit 1
fi

# Create and activate virtual environment
if [ ! -d ".venv" ]; then
    echo "=== Creating virtual environment ==="
    python3 -m venv .venv
fi

echo "=== Activating virtual environment ==="
source .venv/bin/activate

# Set up CUDA paths if available
if [ -d "/usr/local/cuda" ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Force build from source
export MAMBA_FORCE_BUILD="TRUE"

# Install build dependencies first
echo "=== Installing build dependencies ==="
pip install --upgrade pip
pip install wheel setuptools --no-cache-dir

echo "=== Installing PyTorch and dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install triton einops transformers packaging ninja --no-cache-dir

# Print diagnostic information
echo "=== Build Diagnostics ==="
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
python3 -c "from torch.utils import cpp_extension; print('CUDA_HOME:', cpp_extension.CUDA_HOME)"
which nvcc && nvcc --version || echo "nvcc not found in PATH"

# Try to build with verbose output
echo "=== Building CUDA extensions ==="
pip install -e . --no-build-isolation --no-deps -v 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Build successful! ==="
    python3 -c "import torch; import triton; import mamba_ssm; print('✓ All imports successful')"
    echo ""
    echo "=== Setup Complete ==="
    echo "To activate the environment, run: source .venv/bin/activate"
else
    echo ""
    echo "=== Build failed. Last 100 lines of build.log ==="
    tail -100 build.log
    echo ""
    echo "=== Checking for common errors ==="
    grep -i "error:" build.log | tail -10 || echo "No specific errors found in log"
    exit 1
fi
