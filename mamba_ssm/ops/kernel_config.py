import json
import os
from pathlib import Path

_config_cache = None

def get_kernel_config():
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    config_path = Path(__file__).parent.parent.parent / "kernels_config.json"
    if not config_path.exists():
        _config_cache = {}
        return _config_cache
    
    with open(config_path) as f:
        _config_cache = json.load(f)
    return _config_cache

def use_cuda_kernel(kernel_name):
    config = get_kernel_config()
    return config.get(kernel_name, {}).get("use_cuda", False)

